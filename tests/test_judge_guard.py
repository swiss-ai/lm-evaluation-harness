"""Judge-scored safety tasks fail instead of reporting a score their judge never gave.

MultiJail averaged only the verdicts that parsed and returned 0 when none did, so
a run whose judge was down "succeeded" with safe = unsafe = invalid = 0 (evalsdev
2951cc29, 2026-11-25). Aya red-teaming did the same with a printed warning."""

import importlib
import sys
from unittest import mock

import pytest

from lm_eval.api import judge_guard
from lm_eval.api.judge_guard import check_judge_failures


def test_all_failed_raises():
    with pytest.raises(RuntimeError, match="only 0 of 10"):
        check_judge_failures("multijail", "j", 0, 10)


def test_a_few_failures_are_tolerated_but_not_many(monkeypatch):
    monkeypatch.delenv(judge_guard.MAX_FAILED_FRACTION_ENV, raising=False)
    check_judge_failures("t", "j", 96, 100)  # 4% failed
    with pytest.raises(RuntimeError, match="more than 5%"):
        check_judge_failures("t", "j", 90, 100)
    monkeypatch.setenv(judge_guard.MAX_FAILED_FRACTION_ENV, "0.2")
    check_judge_failures("t", "j", 90, 100)
    with pytest.raises(RuntimeError):
        check_judge_failures("t", "j", 0, 100)  # never with no verdict at all


def test_bad_threshold_is_rejected(monkeypatch):
    monkeypatch.setenv(judge_guard.MAX_FAILED_FRACTION_ENV, "1.5")
    with pytest.raises(ValueError):
        check_judge_failures("t", "j", 1, 2)


def test_empty_task_is_not_an_error():
    check_judge_failures("t", "j", 0, 0)


def test_multijail_raises_when_the_judge_failed_and_scores_when_it_answered(
    monkeypatch,
):
    from lm_eval.tasks.multijail import metric

    none = {"safe": None, "unsafe": None, "invalid": None}
    monkeypatch.setattr(metric, "_run_all", lambda items, workers: [none] * len(items))
    with pytest.raises(RuntimeError, match="multijail"):
        metric.safe_agg([{}] * 3)

    good = [{"safe": 1, "unsafe": 0, "invalid": 0}] * 3 + [
        {"safe": 0, "unsafe": 1, "invalid": 0}
    ]
    monkeypatch.setattr(metric, "_run_all", lambda items, workers: good)
    assert metric.safe_agg([{}] * 4) == 0.75
    assert metric.unsafe_agg([{}] * 4) == 0.25
    assert metric.invalid_agg([{}] * 4) == 0.0


def _import_aya(monkeypatch):
    # the module checks the judge with a real request when it is imported
    monkeypatch.setenv("CSCS_SERVING_API", "test-key")
    ok = mock.Mock(status_code=200)
    ok.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    with (
        mock.patch("requests.post", return_value=ok),
        mock.patch(
            "lm_eval.api.model_resolver.resolve_judge_model",
            side_effect=lambda m, **_: m,
        ),
    ):
        sys.modules.pop("lm_eval.tasks.aya_redteaming.metric", None)
        return importlib.import_module("lm_eval.tasks.aya_redteaming.metric")


def test_aya_raises_when_the_judge_failed_and_scores_when_it_answered(monkeypatch):
    aya = _import_aya(monkeypatch)
    monkeypatch.setattr(aya, "_call_judge", lambda prompt: None)
    with pytest.raises(RuntimeError, match="aya_redteaming"):
        aya.harmfulness_agg(["p"] * 5)
    answers = iter([1, 0, 0, 0])
    monkeypatch.setattr(aya, "_call_judge", lambda prompt: next(answers))
    assert aya.harmfulness_agg(["p"] * 4, max_workers=1) == 0.25


def test_an_unparsable_answer_is_asked_again(monkeypatch):
    from lm_eval.api.judge_guard import judge_until_parsed

    answers = iter(["yS. garbled", None, "Yes"])
    parse = lambda a: 1 if a.lower().startswith("yes") else None
    assert judge_until_parsed(lambda: next(answers), parse, attempts=3) == 1
    assert judge_until_parsed(lambda: "garbled", parse, attempts=4) is None


def test_judge_workers_follow_the_environment(monkeypatch):
    from lm_eval.api.judge_guard import max_workers

    monkeypatch.delenv("JUDGE_MAX_WORKERS", raising=False)
    assert max_workers(16) == 16
    monkeypatch.setenv("JUDGE_MAX_WORKERS", "1")
    assert max_workers(16) == 1
