"""Judge-backed safety metrics must not report a score when the judge failed.

These aggregations run the judge calls themselves, so every test here stubs the
network out. The invariant under test is that an unusable judge yields NaN
rather than a number that reads as a clean result -- for `aya_redteaming.score`
and `multijail.unsafe`, 0 is the *best* possible score.
"""

import importlib.util
import math
from pathlib import Path

import pytest


TASKS = Path(__file__).parent.parent / "lm_eval" / "tasks"


def _load(module_path, monkeypatch):
    """Import a task metric module with the API key and network stubbed out."""
    monkeypatch.setenv("CSCS_SERVING_API", "test-key")

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "safe"}}]}

    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())

    spec = importlib.util.spec_from_file_location(f"_t_{module_path.stem}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def aya(monkeypatch):
    return _load(TASKS / "aya_redteaming" / "metric.py", monkeypatch)


@pytest.fixture
def multijail(monkeypatch):
    return _load(TASKS / "multijail" / "metric.py", monkeypatch)


def test_aya_all_judges_failing_is_nan_not_zero(aya, monkeypatch):
    monkeypatch.setattr(aya, "try_remote_generate", lambda *a, **k: None)
    assert math.isnan(aya.harmfulness_agg(["p1", "p2", "p3"], max_workers=1))


def test_aya_partial_failure_still_averages_usable_verdicts(aya, monkeypatch):
    replies = iter(["Yes", "No", None])
    monkeypatch.setattr(aya, "try_remote_generate", lambda *a, **k: next(replies))
    assert aya.harmfulness_agg(["p1", "p2", "p3"], max_workers=1) == 0.5


@pytest.mark.parametrize("agg", ["safe_agg", "unsafe_agg", "invalid_agg"])
def test_multijail_all_judges_failing_is_nan_not_zero(multijail, monkeypatch, agg):
    monkeypatch.setattr(
        multijail,
        "_run_all",
        lambda items, mw: [
            {"safe": None, "unsafe": None, "invalid": None} for _ in items
        ],
    )
    assert math.isnan(getattr(multijail, agg)([{"p": 1}, {"p": 2}]))


@pytest.mark.parametrize(
    "response,expected",
    [
        ("safe", 0),
        ("unsafe\nS2", 1),
        ("Safe", 0),
        # Negation used to match the "unsafe" substring and invert the verdict.
        ("This response is not unsafe.", None),
        ("Neither safe nor unsafe", None),
    ],
)
def test_llama_guard_verdict_is_anchored_not_substring(monkeypatch, response, expected):
    for name in ("realtoxicitypromptsllama", "polyglotoxicitypromptsllama"):
        module = _load(TASKS / name / "metrics.py", monkeypatch)
        result = module.evaluate_llama_guard(response)
        if expected is None:
            assert math.isnan(result), f"{name}: {response!r} -> {result}"
        else:
            assert result == expected, f"{name}: {response!r} -> {result}"
