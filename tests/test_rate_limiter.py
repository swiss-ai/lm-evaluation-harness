import asyncio

import pytest

from lm_eval.api.rate_limiter import RateLimiter, get_rate_limiter


class _Clock:
    def __init__(self):
        self.now = 1000.0
        self.delays = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.delays.append(delay)
        self.now += delay

    async def async_sleep(self, delay):
        self.sleep(delay)


def test_rate_limiter_evenly_spaces_sync_requests():
    clock = _Clock()
    limiter = RateLimiter(
        30,
        _clock=clock.time,
        _sleep=clock.sleep,
        _async_sleep=clock.async_sleep,
    )

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert clock.delays == [2.0, 2.0]


def test_rate_limiter_evenly_spaces_async_requests():
    clock = _Clock()
    limiter = RateLimiter(
        20,
        _clock=clock.time,
        _sleep=clock.sleep,
        _async_sleep=clock.async_sleep,
    )

    async def run():
        await limiter.acquire_async()
        await limiter.acquire_async()

    asyncio.run(run())

    assert clock.delays == [3.0]


def test_shared_state_coordinates_distinct_limiter_instances(tmp_path):
    clock = _Clock()
    state_path = tmp_path / "shared.limit"
    first = RateLimiter(
        30,
        state_path=state_path,
        _clock=clock.time,
        _sleep=clock.sleep,
        _async_sleep=clock.async_sleep,
    )
    second = RateLimiter(
        30,
        state_path=state_path,
        _clock=clock.time,
        _sleep=clock.sleep,
        _async_sleep=clock.async_sleep,
    )

    first.acquire()
    second.acquire()

    assert clock.delays == [2.0]


@pytest.mark.parametrize("value", [0, -1, "zero", float("nan"), float("inf")])
def test_rate_limiter_rejects_invalid_rates(value):
    with pytest.raises(ValueError, match="positive number"):
        RateLimiter(value)


def test_get_rate_limiter_is_scoped_and_optional(monkeypatch):
    monkeypatch.delenv("LM_EVAL_RATE_LIMIT_STATE_DIR", raising=False)

    assert get_rate_limiter(None, scope="disabled") is None
    first = get_rate_limiter(30, scope="endpoint-a")
    second = get_rate_limiter("30", scope="endpoint-a")
    other = get_rate_limiter(30, scope="endpoint-b")

    assert first is second
    assert first is not other
