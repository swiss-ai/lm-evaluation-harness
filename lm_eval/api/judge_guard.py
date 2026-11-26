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
