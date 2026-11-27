"""Stop a judge-scored task from reporting a score its judge never gave.

MultiJail and Aya red-teaming average only the verdicts their judge returned
and parsed. When the judge is unreachable (HTTP 503 "No provider found"),
rate-limited, or answers in a form the parser rejects, those verdicts are
dropped; with none left the score used to be 0, and the run "succeeded" with
safe = unsafe = invalid = 0 (evalsdev 2951cc29, 2026-11-25).

check_judge_failures() raises instead when no verdict was usable, or when more
than JUDGE_MAX_FAILED_FRACTION (default 0.05) of them failed.
"""

from __future__ import annotations

import os


MAX_FAILED_FRACTION_ENV = "JUDGE_MAX_FAILED_FRACTION"
DEFAULT_MAX_FAILED_FRACTION = 0.05


def max_failed_fraction() -> float:
    raw = os.getenv(MAX_FAILED_FRACTION_ENV, "").strip()
    if not raw:
        return DEFAULT_MAX_FAILED_FRACTION
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{MAX_FAILED_FRACTION_ENV} must be between 0 and 1 (got {raw!r})"
        )
    return value


def check_judge_failures(task: str, judge: str, n_valid: int, n_total: int) -> None:
    """Raise if the judge gave too few usable verdicts for the score to mean anything."""
    if n_total == 0:
        return
    failed = n_total - n_valid
    limit = max_failed_fraction()
    if n_valid == 0 or failed / n_total > limit:
        raise RuntimeError(
            f"{task}: the judge {judge} gave a usable verdict for only {n_valid} of {n_total} "
            f"items ({failed} failed: unreachable, rate-limited or unparsable answers); "
            f"more than {limit:.0%} failed, so no score is reported. "
            f"Check that the judge is served (see <TASK>_JUDGE_MODEL), or raise "
            f"{MAX_FAILED_FRACTION_ENV} to accept more failures."
        )


# A judge under load can return a garbled answer for a prompt it answers fine on its
# own (api.inference.cscs.ch's Gemma-4-31B-it, 2026-11-27): ask it again before
# counting the item as failed.
JUDGE_PARSE_ATTEMPTS_ENV = "JUDGE_PARSE_ATTEMPTS"
JUDGE_MAX_WORKERS_ENV = "JUDGE_MAX_WORKERS"


def parse_attempts(default: int = 5) -> int:
    return max(1, int(os.getenv(JUDGE_PARSE_ATTEMPTS_ENV) or default))


def max_workers(default: int = 16) -> int:
    """How many judge requests run at once: JUDGE_MAX_WORKERS, else the task's default."""
    return max(1, int(os.getenv(JUDGE_MAX_WORKERS_ENV) or default))


def judge_until_parsed(ask, parse, attempts=None):
    """parse(ask()) until it is not None, at most `attempts` times; None if never.
    ask() returning None (unreachable after its own retries) also counts as an attempt."""
    for _ in range(attempts or parse_attempts()):
        answer = ask()
        if answer is not None:
            parsed = parse(answer)
            if parsed is not None:
                return parsed
    return None
