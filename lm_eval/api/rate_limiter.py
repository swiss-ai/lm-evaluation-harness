"""Thread-, async-, and optionally process-safe request rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


JUDGE_REQUESTS_PER_MINUTE_ENV = "JUDGE_REQUESTS_PER_MINUTE"
RATE_LIMIT_STATE_DIR_ENV = "LM_EVAL_RATE_LIMIT_STATE_DIR"


def _parse_requests_per_minute(value: float | str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"requests_per_minute must be a positive number, got {value!r}"
        ) from exc
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError(
            f"requests_per_minute must be a positive number, got {value!r}"
        )
    return rate


def _state_path(state_dir: str | os.PathLike[str], scope: str) -> Path:
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()
    return Path(state_dir) / f"{digest}.limit"


class RateLimiter:
    """Evenly space request starts to stay at or below a requests/minute limit.

    A limiter is safe to share across threads and asyncio event loops. When
    ``state_path`` is supplied, cooperating processes on machines that share
    that filesystem also reserve request slots under an advisory file lock.
    """

    def __init__(
        self,
        requests_per_minute: float | str,
        *,
        state_path: str | os.PathLike[str] | None = None,
        _clock: Callable[[], float] = time.time,
        _sleep: Callable[[float], None] = time.sleep,
        _async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        rate = _parse_requests_per_minute(requests_per_minute)
        assert rate is not None
        self.requests_per_minute = rate
        self._interval = 60.0 / rate
        self._state_path = Path(state_path) if state_path is not None else None
        self._clock = _clock
        self._sleep = _sleep
        self._async_sleep = _async_sleep
        self._lock = threading.Lock()
        self._next_request_at = 0.0

    def _reserve_local(self) -> float:
        with self._lock:
            now = self._clock()
            request_at = max(now, self._next_request_at)
            self._next_request_at = request_at + self._interval
        return max(0.0, request_at - now)

    def _reserve_shared(self) -> float:
        # fcntl is Unix-only. The shared-filesystem mode is opt-in and is used
        # by the CSCS Slurm launcher, whose compute environments are Linux.
        import fcntl

        assert self._state_path is not None
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with self._state_path.open("a+", encoding="utf-8") as state_file:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
            try:
                state_file.seek(0)
                raw_next_request_at = state_file.read().strip()
                try:
                    next_request_at = float(raw_next_request_at)
                except ValueError:
                    next_request_at = 0.0

                now = self._clock()
                request_at = max(now, next_request_at)
                state_file.seek(0)
                state_file.truncate()
                state_file.write(f"{request_at + self._interval:.9f}\n")
                state_file.flush()
            finally:
                fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)
        return max(0.0, request_at - now)

    def _reserve(self) -> float:
        if self._state_path is None:
            return self._reserve_local()
        return self._reserve_shared()

    def acquire(self) -> None:
        delay = self._reserve()
        if delay > 0:
            self._sleep(delay)

    async def acquire_async(self) -> None:
        delay = self._reserve()
        if delay > 0:
            await self._async_sleep(delay)


_LIMITERS: dict[tuple[str, float, str | None], RateLimiter] = {}
_LIMITERS_LOCK = threading.Lock()


def get_rate_limiter(
    requests_per_minute: float | str | None,
    *,
    scope: str,
    state_dir: str | os.PathLike[str] | None = None,
) -> RateLimiter | None:
    """Return a process-wide limiter for ``scope``, or ``None`` when disabled."""

    rate = _parse_requests_per_minute(requests_per_minute)
    if rate is None:
        return None

    resolved_state_dir = (
        os.fspath(state_dir)
        if state_dir is not None
        else os.getenv(RATE_LIMIT_STATE_DIR_ENV) or None
    )
    key = (scope, rate, resolved_state_dir)
    with _LIMITERS_LOCK:
        limiter = _LIMITERS.get(key)
        if limiter is None:
            limiter = RateLimiter(
                rate,
                state_path=(
                    _state_path(resolved_state_dir, scope)
                    if resolved_state_dir is not None
                    else None
                ),
            )
            _LIMITERS[key] = limiter
        return limiter


def get_judge_rate_limiter(scope: str) -> RateLimiter | None:
    """Return the shared limiter configured for an LLM judge endpoint/model."""

    return get_rate_limiter(
        os.getenv(JUDGE_REQUESTS_PER_MINUTE_ENV),
        scope=f"judge:{scope}",
    )


def acquire_judge_rate_limit(scope: str) -> None:
    """Wait for the next configured judge request slot."""

    limiter = get_judge_rate_limiter(scope)
    if limiter is not None:
        limiter.acquire()
