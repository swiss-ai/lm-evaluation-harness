from argparse import Namespace

import datasets
import numpy as np
import pytest

from lm_eval.api.group import Group
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.config.group import AggMetricConfig
from lm_eval.config.task import TaskConfig
from lm_eval.evaluator import evaluate
from lm_eval.models.dummy import DummyLM
from lm_eval.repeats import configure_repeats


def grade(doc, responses):
    response = responses[0]
    return {
        "exact_match": np.int64(response == "A"),
        "math_verify": float(response == "B"),
        "weighted": (int(response == "A"), doc["weight"]),
    }


def weighted(items):
    return sum(x for x, _ in items) / sum(w for _, w in items)


@pytest.fixture
def task_factory(monkeypatch):
    def download(self, dataset_kwargs=None, **kwargs):
        self.dataset = {
            "test": datasets.Dataset.from_list(
                [
                    {"question": "one", "answer": "A", "weight": 1},
                    {"question": "two", "answer": "A", "weight": 3},
                ]
            )
        }

    monkeypatch.setattr(ConfigurableTask, "download", download)

    def make(name="example", **kwargs):
        return ConfigurableTask(
            config=TaskConfig(
                task=name,
                dataset_path="",
                test_split="test",
                num_fewshot=0,
                doc_to_text="question",
                doc_to_target="answer",
                process_results=grade,
                metric_list=[
                    {
                        "metric": k,
                        "aggregation": weighted if k == "weighted" else "mean",
                        "higher_is_better": True,
                    }
                    for k in ("exact_match", "math_verify", "weighted")
                ],
                repeat_support={
                    "exact_match": "binary",
                    "math_verify": "binary",
                    "weighted": "native",
                },
                **kwargs,
            )
        )

    return make


class Answers(DummyLM):
    def generate_until(self, requests, **kwargs):
        return [["A", "B", "wrong", "B", "B", "B"][i % 6] for i in range(len(requests))]


@pytest.mark.parametrize("log_samples", [False, True])
def test_evaluator_scores_every_attempt_and_grader(task_factory, log_samples):
    task = task_factory(
        repeats=3,
        repeat_scoring={
            "enabled": True,
            "pass_at_k": {"exact_match": [2, 3], "math_verify": [2, 3]},
        },
    )
    result = evaluate(
        Answers(),
        {"tasks": {"example": task}, "groups": {}},
        log_samples=log_samples,
        bootstrap_iters=0,
    )
    scores = result["results"]["example"]
    assert scores["exact_match_mean_repeats,none"] == pytest.approx(1 / 6)
    assert scores["math_verify_mean_repeats,none"] == pytest.approx(2 / 3)
    assert scores["exact_match_pass@2,none"] == pytest.approx(1 / 3)
    assert scores["math_verify_pass@2,none"] == pytest.approx(5 / 6)
    assert scores["exact_match_pass@3,none"] == 0.5
    assert scores["math_verify_pass@3,none"] == 1
    assert scores["weighted_mean_repeats,none"] == pytest.approx(1 / 12)
    assert result["repeat_results"][0]["results"]["example"]["weighted,none"] == 0.25
    assert result["n-samples"]["example"]["effective"] == 2
    if log_samples:
        assert result["samples"]["example"][0]["exact_match"] == [1, 0, 0]
    else:
        assert "samples" not in result


def test_precedence_clear_and_fail_before_generation(task_factory, tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text(
        "tasks: [a, b]\nrepeats: 3\nrepeat_scoring:\n  enabled: true\n  pass_at_k: {exact_match: [3]}\n"
    )
    cfg = EvaluatorConfig.from_cli(
        Namespace(config=str(config), repeat_scoring={"pass_at_k": {}})
    )
    a = task_factory("a", repeats=8, repeat_scoring={"pass_at_k": {"exact_match": [8]}})
    b = task_factory("b", repeats=9)
    configure_repeats(
        {"a": a, "b": b},
        cfg.repeats,
        cfg.repeat_scoring,
        {"a": {"repeats": 2, "repeat_scoring": {"pass_at_k": {"math_verify": [2]}}}},
    )
    assert a.config.repeats == 2 and b.config.repeats == 3
    assert a.config.repeat_scoring["pass_at_k"] == {"math_verify": [2]}
    assert b.config.repeat_scoring["pass_at_k"] == {}
    with pytest.raises(ValueError, match="binary"):
        configure_repeats({"b": b}, repeat_scoring={"pass_at_k": {"weighted": [2]}})
    with pytest.raises(ValueError, match="between"):
        configure_repeats({"b": b}, repeat_scoring={"pass_at_k": {"exact_match": [4]}})
    with pytest.raises(ValueError, match="leaf"):
        configure_repeats({"b": b}, task_overrides={"typo": {"repeats": 2}})


def test_legacy_and_single_repeat(task_factory):
    legacy = task_factory(repeats=3)
    result = evaluate(
        Answers(), {"tasks": {"example": legacy}, "groups": {}}, bootstrap_iters=0
    )
    assert result["results"]["example"]["exact_match,none"] == 0.5
    assert "repeat_results" not in result
    single = task_factory(repeats=1, repeat_scoring={"enabled": True})
    result = evaluate(
        Answers(), {"tasks": {"example": single}, "groups": {}}, bootstrap_iters=0
    )
    assert result["results"]["example"]["exact_match_mean_repeats,none"] == 0.5


@pytest.mark.parametrize("mixed", [False, True])
def test_group_never_reports_a_partial_repeat_score(task_factory, mixed):
    a = task_factory("a", repeats=3, repeat_scoring={"enabled": True})
    b = task_factory("b", repeats=1, repeat_scoring={"enabled": not mixed})
    group = Group(
        "suite", aggregate_metric_list=[AggMetricConfig(metric="exact_match")]
    )
    group.add(a)
    group.add(b)
    result = evaluate(
        Answers(),
        {"tasks": {"a": a, "b": b}, "groups": {"suite": group}},
        bootstrap_iters=0,
    )
    metrics = result["results"]["suite"]
    assert "exact_match,none" not in metrics
    if mixed:
        assert "exact_match_mean_repeats,none" not in metrics
    else:
        assert metrics["exact_match_mean_repeats,none"] == pytest.approx(1 / 3)
        assert "b" not in result["repeat_results"][1]["results"]


def test_missing_response_is_not_an_incorrect_attempt(task_factory):
    class Missing(Answers):
        def generate_until(self, requests, **kwargs):
            return [None] * len(requests)

    task = task_factory(repeats=2, repeat_scoring={"enabled": True})
    with pytest.raises(RuntimeError, match="missing attempt"):
        evaluate(
            Missing(), {"tasks": {"example": task}, "groups": {}}, bootstrap_iters=0
        )


def test_native_unavailable_score_keeps_all_repeats(task_factory):
    from math import isnan

    task = task_factory(repeats=2, repeat_scoring={"enabled": True})
    task._aggregation_list["weighted"] = lambda items: float("nan")
    result = evaluate(
        Answers(), {"tasks": {"example": task}, "groups": {}}, bootstrap_iters=0
    )
    assert isnan(result["results"]["example"]["weighted_mean_repeats,none"])
    assert len(result["repeat_results"]) == 2


def test_request_cache_changes_with_rendering_policy(task_factory, monkeypatch):
    from types import SimpleNamespace

    keys = []
    monkeypatch.setattr(
        "lm_eval.api.task.load_from_cache",
        lambda file_name, cache: keys.append(file_name),
    )
    monkeypatch.setattr("lm_eval.api.task.save_to_cache", lambda **kwargs: None)

    class Renderer:
        tokenizer = SimpleNamespace(chat_template="same-template")

        def render(self, messages, **kwargs):
            return str(self.chat_template_args) + str(messages)

    renderer = Renderer()
    prompts = []
    for thinking in [False, True]:
        renderer.chat_template_args = {"enable_thinking": thinking}
        task = task_factory()
        task.set_fewshot_seed(1234)
        task.build_all_requests(
            cache_requests=True, apply_chat_template=True, chat_template=renderer.render
        )
        prompts.append(task.instances[0].args[0])
    assert prompts[0] != prompts[1]
    assert keys[0] != keys[1]


def test_math500_legacy_pass_keeps_each_graders_success(monkeypatch):
    from lm_eval.tasks.minerva_math import utils

    monkeypatch.setattr(
        utils, "_score_doc", lambda answer, doc: {"A": (1, 0), "B": (0, 1)}[answer]
    )
    filtered = utils.PassAtKFilter().apply([["A", "B"]], [{}])[0]
    assert utils.process_results_sc({}, filtered) == {
        "exact_match": 1,
        "math_verify": 1,
    }


@pytest.mark.parametrize("overlap", [False, True])
def test_groups_preserve_declared_filter_coverage(task_factory, overlap):
    tasks = {}
    for name, selected in [("a", "f"), ("b", "g")]:
        filters = ["f", "g"] if overlap else [selected]
        tasks[name] = task_factory(
            name,
            repeats=2,
            repeat_filter=selected,
            repeat_scoring={"enabled": True},
            filter_list=[
                {"name": f, "filter": [{"function": "take_first"}]} for f in filters
            ],
        )
    group = Group(
        "suite",
        aggregate_metric_list=[
            AggMetricConfig(metric="exact_match", filter_list=["f", "g"])
        ],
    )
    for task in tasks.values():
        group.add(task)
    result = evaluate(
        Answers(), {"tasks": tasks, "groups": {"suite": group}}, bootstrap_iters=0
    )
    metrics = result["results"]["suite"]
    assert ("exact_match_mean_repeats,f" in metrics) is not overlap
    assert ("exact_match_mean_repeats,g" in metrics) is not overlap


def test_cli_writes_repeated_results_from_run_file(tmp_path):
    import json
    import os
    import subprocess
    import sys

    (tmp_path / "data.jsonl").write_text(
        '{"question":"one","answer":"never"}\n{"question":"two","answer":"never"}\n'
    )
    (tmp_path / "grader.py").write_text(
        'def grade(doc, responses):\n    return {"correctness": float(responses[0] == doc["answer"])}\n'
    )
    task = {
        "task": "repeat_cli_smoke",
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(tmp_path / "data.jsonl")}},
        "test_split": "test",
        "output_type": "generate_until",
        "num_fewshot": 0,
        "doc_to_text": "question",
        "doc_to_target": "answer",
        "metric_list": [
            {"metric": "correctness", "aggregation": "mean", "higher_is_better": True}
        ],
        "repeat_support": {"correctness": "binary"},
    }
    import yaml

    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(task) + "process_results: !function grader.grade\n"
    )
    config = {
        "model": "dummy",
        "include_path": str(tmp_path),
        "tasks": ["repeat_cli_smoke"],
        "repeats": 3,
        "repeat_scoring": {"enabled": True, "pass_at_k": {"correctness": [2]}},
        "log_samples": True,
        "output_path": str(tmp_path / "results"),
    }
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(config))
    completed = subprocess.run(  # noqa: S603 - Local CLI with synthetic inputs.
        [sys.executable, "-m", "lm_eval", "run", "--config", str(config_path)],
        capture_output=True,
        check=False,
        text=True,
        env=os.environ
        | {"HF_DATASETS_CACHE": str(tmp_path / "datasets"), "HF_HUB_OFFLINE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(
        next((tmp_path / "results").rglob("results_*.json")).read_text()
    )
    assert result["results"]["repeat_cli_smoke"]["correctness_mean_repeats,none"] == 0
    assert result["results"]["repeat_cli_smoke"]["correctness_pass@2,none"] == 0
    assert result["repeat_protocols"]["repeat_cli_smoke"]["completed_attempts"] == 6
    assert len(result["repeat_results"]) == 3
