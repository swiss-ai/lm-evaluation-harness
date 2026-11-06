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
from lm_eval.repeats import configure_repeats, enabled


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
                **{
                    "repeat_support": {
                        "exact_match": "binary",
                        "math_verify": "binary",
                        "weighted": "native",
                    },
                    **kwargs,
                },
            )
        )

    return make


class Answers(DummyLM):
    supports_repeat_scoring = True

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
        configure_repeats(
            {"b": b},
            task_overrides={"b": {"repeat_scoring": {"pass_at_k": {"weighted": [2]}}}},
        )
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

    class NoGeneration(Answers):
        def generate_until(self, requests, **kwargs):
            raise AssertionError("checked before any request")

    if mixed:
        with pytest.raises(ValueError, match="all of them or none"):
            evaluate(
                NoGeneration(),
                {"tasks": {"a": a, "b": b}, "groups": {"suite": group}},
                bootstrap_iters=0,
            )
        return
    result = evaluate(
        Answers(),
        {"tasks": {"a": a, "b": b}, "groups": {"suite": group}},
        bootstrap_iters=0,
    )
    metrics = result["results"]["suite"]
    assert "exact_match,none" not in metrics
    assert metrics["exact_match_mean_repeats,none"] == pytest.approx(1 / 3)
    assert "b" not in result["repeat_results"][1]["results"]


def test_group_aggregates_repeat_metrics_over_the_tasks_that_report_them(
    task_factory,
):
    a = task_factory("a", repeats=3, repeat_scoring={"enabled": True})
    b = task_factory(
        "b",
        repeats=3,
        repeat_scoring={"enabled": True},
        repeat_support={"math_verify": "binary"},
    )
    group = Group(
        "suite", aggregate_metric_list=[AggMetricConfig(metric="exact_match")]
    )
    group.add(a)
    group.add(b)
    result = evaluate(
        Answers(),
        {"tasks": {"a": a, "b": b}, "groups": {"suite": group}},
        bootstrap_iters=0,
    )["results"]
    key = "exact_match_mean_repeats,none"
    assert key not in result["b"]
    assert result["suite"][key] == result["a"][key]


def test_run_level_pass_at_k_covers_the_metrics_each_task_declares(task_factory):
    scored = task_factory("scored")
    other = task_factory("other", repeat_support={"math_verify": "binary"})
    configure_repeats(
        {"scored": scored, "other": other},
        repeats=2,
        repeat_scoring={"enabled": True, "pass_at_k": {"exact_match": [2]}},
    )
    assert scored.config.repeat_scoring["pass_at_k"] == {"exact_match": [2]}
    assert other.config.repeat_scoring["pass_at_k"] == {}


def test_cli_overrides_update_run_file_fields(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text(
        "tasks: [a]\ntask_overrides:\n  a: {repeats: 2, repeat_scoring: {enabled: true}}\n"
    )
    cfg = EvaluatorConfig.from_cli(
        Namespace(
            config=str(config),
            task_overrides={"a": {"repeat_scoring": {"mean": False}}, "b": 5},
        )
    )
    assert cfg.task_overrides == {
        "a": {"repeats": 2, "repeat_scoring": {"enabled": True, "mean": False}},
        "b": 5,
    }


def test_repeat_scoring_rejects_models_without_attempt_seeds(task_factory, tmp_path):
    from lm_eval.api.model import CachingLM

    class Local(Answers):
        supports_repeat_scoring = False

    task = task_factory(repeats=2, repeat_scoring={"enabled": True})
    for model in (Local(), CachingLM(Local(), str(tmp_path / "cache.db"))):
        with pytest.raises(ValueError, match="API models only"):
            evaluate(model, {"tasks": {"example": task}, "groups": {}})
    legacy = task_factory(repeats=2)
    result = evaluate(Local(), {"tasks": {"example": legacy}, "groups": {}})
    assert "exact_match,none" in result["results"]["example"]


def test_run_level_repeats_leave_loglikelihood_tasks_alone(task_factory):
    generative = task_factory("gen")
    choice = task_factory(
        "choice", output_type="multiple_choice", doc_to_choice=["A", "B"]
    )
    configure_repeats({"gen": generative, "choice": choice}, repeats=5)
    assert generative.config.repeats == 5 and choice.config.repeats == 1


def test_run_level_policy_covers_only_tasks_that_can_repeat_score(task_factory):
    def choice(name):
        return task_factory(
            name, output_type="multiple_choice", doc_to_choice=["A", "B"]
        )

    tasks = {"gen": task_factory("gen"), "choice": choice("choice")}
    tasks |= {"member": task_factory("member"), "mc_member": choice("mc_member")}
    group = Group("mixed", aggregate_metric_list=[AggMetricConfig(metric="acc")])
    group.add(tasks["member"])
    group.add(tasks["mc_member"])
    configure_repeats(tasks, repeat_scoring={"enabled": True}, groups=[group])
    assert [name for name, task in tasks.items() if enabled(task)] == ["gen"]


def test_run_level_policy_skips_multi_sample_filters_and_rejects_non_objects(
    task_factory,
):
    voting = task_factory(
        "voting",
        filter_list=[
            {"name": "maj@8", "filter": [{"function": "take_first_k", "k": 8}]}
        ],
    )
    configure_repeats({"voting": voting}, repeat_scoring={"enabled": True})
    assert not enabled(voting)
    with pytest.raises(ValueError, match="must be an object"):
        configure_repeats({"voting": voting}, repeat_scoring=True)
    with pytest.raises(ValueError, match="requests no metrics"):
        configure_repeats(
            {"voting": voting}, repeat_scoring={"enabled": True, "mean": False}
        )


def test_single_attempts_reach_the_model_uncopied(task_factory):
    task = task_factory()
    sent = []

    class Recorder(Answers):
        def generate_until(self, requests, **kwargs):
            sent.extend(requests)
            return super().generate_until(requests)

    evaluate(Recorder(), {"tasks": {"example": task}, "groups": {}})
    assert all(any(r is i for i in task.instances) for r in sent)


def test_pass_at_k_keeps_the_metric_direction(task_factory):
    task = task_factory(
        repeats=2,
        repeat_scoring={"enabled": True, "pass_at_k": {"exact_match": [2]}},
    )
    task._higher_is_better["exact_match"] = False
    result = evaluate(Answers(), {"tasks": {"example": task}, "groups": {}})
    assert result["higher_is_better"]["example"]["exact_match_pass@2"] is False


def test_override_on_an_unscored_task_warns(task_factory, caplog):
    task = task_factory("plain")
    configure_repeats(
        {"plain": task},
        task_overrides={
            "plain": {"repeat_scoring": {"pass_at_k": {"exact_match": [4]}}}
        },
    )
    assert not enabled(task) and "override is ignored" in caplog.text


def test_overlapping_groups_exclude_each_others_tasks(task_factory):
    tasks = {name: task_factory(name) for name in "ab"}
    tasks["c"] = task_factory(
        "c", output_type="multiple_choice", doc_to_choice=["A", "B"]
    )
    groups = []
    for name, members in (("first", "ab"), ("second", "bc")):
        group = Group(name, aggregate_metric_list=[AggMetricConfig(metric="acc")])
        for member in members:
            group.add(tasks[member])
        groups.append(group)
    configure_repeats(tasks, repeat_scoring={"enabled": True}, groups=groups)
    assert not any(enabled(task) for task in tasks.values())


def test_run_level_repeats_skip_tasks_that_opt_out_of_repeat_scoring(task_factory):
    scored, opted_out = task_factory("scored"), task_factory("opted_out", repeats=2)
    configure_repeats(
        {"scored": scored, "opted_out": opted_out},
        repeats=5,
        repeat_scoring={"enabled": True},
        task_overrides={"opted_out": {"repeat_scoring": {"enabled": False}}},
    )
    assert scored.config.repeats == 5 and opted_out.config.repeats == 2


def test_predict_only_rejects_repeat_scoring(task_factory):
    from lm_eval.evaluator import simple_evaluate

    with pytest.raises(ValueError, match="predict_only"):
        simple_evaluate(
            model=Answers(),
            tasks=[task_factory(repeats=2)],
            repeat_scoring={"enabled": True},
            predict_only=True,
        )


def test_group_protocol_lists_each_k_once(task_factory):
    tasks = {
        name: task_factory(
            name,
            repeats=2,
            repeat_scoring={"enabled": True, "pass_at_k": {"exact_match": [2]}},
            filter_list=[
                {"name": f, "filter": [{"function": "take_first"}]} for f in ("f", "g")
            ],
        )
        for name in ("a", "b")
    }
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
    assert "exact_match_pass@2,f" in result["results"]["suite"]
    assert "exact_match_pass@2,g" in result["results"]["suite"]
    policy = result["repeat_protocols"]["suite"]["policy"]
    assert policy["pass_at_k"] == {"exact_match": [2]}


def test_caching_lm_keeps_greedy_generations_for_other_models(tmp_path):
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import CachingLM

    calls = []

    class Local(DummyLM):
        supports_repeat_scoring = False

        def generate_until(self, requests, disable_tqdm=False):
            calls.append(len(requests))
            return ["text"] * len(requests)

    model = CachingLM(Local(), str(tmp_path / "cache.db"), cache_namespace="local")
    request = Instance(
        "generate_until", {}, ("prompt", {"until": ["\n"]}), 0, repeat_index=0
    )
    assert model.generate_until([request]) == ["text"]
    assert model.generate_until([request]) == ["text"]
    assert calls == [1]


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


def test_request_cache_key_includes_repeats(task_factory, monkeypatch):
    keys = []
    monkeypatch.setattr(
        "lm_eval.api.task.load_from_cache",
        lambda file_name, cache: keys.append(file_name),
    )
    monkeypatch.setattr("lm_eval.api.task.save_to_cache", lambda **kwargs: None)
    for repeats in [1, 3]:
        task_factory(repeats=repeats).build_all_requests(cache_requests=True)
    assert keys[0] != keys[1]


@pytest.mark.parametrize("case", ["same", "crossed", "disjoint"])
def test_group_filters_cover_the_tasks_that_define_them(task_factory, caplog, case):
    defined = {"same": ("fg", "fg"), "crossed": ("fg", "fg"), "disjoint": ("f", "g")}
    chosen = {"same": ("f", "f"), "crossed": ("f", "g"), "disjoint": ("f", "g")}
    tasks = {
        name: task_factory(
            name,
            repeats=2,
            repeat_filter=selected,
            repeat_scoring={"enabled": True},
            filter_list=[
                {"name": f, "filter": [{"function": "take_first"}]} for f in filters
            ],
        )
        for name, filters, selected in zip(
            "ab", defined[case], chosen[case], strict=True
        )
    }
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
    assert ("exact_match_mean_repeats,f" in metrics) is (case != "crossed")
    assert ("exact_match_mean_repeats,g" in metrics) is (case == "disjoint")
    assert ("the group omits it" in caplog.text) is (case == "crossed")


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
    # The dummy model stands in for an API model.
    cli = (
        "import sys; from lm_eval.models.dummy import DummyLM; "
        "DummyLM.supports_repeat_scoring = True; "
        "from lm_eval.__main__ import cli_evaluate; "
        "sys.argv = ['lm_eval', 'run', '--config', sys.argv[1]]; cli_evaluate()"
    )
    completed = subprocess.run(  # noqa: S603 - Local CLI with synthetic inputs.
        [sys.executable, "-c", cli, str(config_path)],
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
    assert result["repeat_protocols"]["repeat_cli_smoke"]["documents"] == 2
    assert len(result["repeat_results"]) == 3


def test_api_protocol_reports_temperature_not_ignored_sampling_flag(
    task_factory, monkeypatch
):
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    model = LocalCompletionsAPI(
        model="test", base_url="http://example.invalid", tokenizer_backend=None
    )
    monkeypatch.setattr(model, "generate_until", Answers().generate_until)
    task = task_factory(
        repeats=2,
        repeat_scoring={"enabled": True},
        generation_kwargs={"temperature": 0, "do_sample": True},
    )
    result = evaluate(
        model, {"tasks": {"example": task}, "groups": {}}, bootstrap_iters=0
    )
    assert result["repeat_protocols"]["example"]["sampling_enabled"] is False


def test_missing_attempts_are_not_cached(tmp_path):
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import CachingLM

    sent = []

    class Api(DummyLM):
        supports_repeat_scoring = True
        cache_generation = True

        def generate_until(self, requests, disable_tqdm=False):
            sent.append(len(requests))
            return [None if r.repeat_index == 0 else "" for r in requests]

    model = CachingLM(Api(), str(tmp_path / "cache.db"), cache_namespace="api")
    attempts = [
        Instance("generate_until", {}, ("q", {"until": []}), 0, repeat_index=i)
        for i in range(2)
    ]
    assert model.generate_until(attempts) == [None, ""]
    assert model.generate_until(attempts) == [None, ""]
    assert sent == [2, 1]


def test_resumed_runs_keep_cached_attempts_across_request_settings(
    task_factory, tmp_path, monkeypatch
):
    from lm_eval.api import model as model_module
    from lm_eval.evaluator import simple_evaluate

    namespaces = []
    init = model_module.CachingLM.__init__

    def record(self, lm, cache_db, cache_namespace=None):
        namespaces.append(cache_namespace)
        init(self, lm, cache_db, cache_namespace)

    monkeypatch.setattr(model_module.CachingLM, "__init__", record)
    for args in (
        "num_concurrent=2,timeout=10",
        "num_concurrent=8,requests_per_minute=60,auth_token=new",
        "max_gen_toks=9",
    ):
        simple_evaluate(
            model="dummy",
            model_args=args,
            tasks=[task_factory()],
            use_cache=str(tmp_path / "cache"),
        )
    assert namespaces[0] == namespaces[1] != namespaces[2]


def test_pass_at_k_only_run_policy_leaves_other_tasks_unscored(task_factory, caplog):
    scored = task_factory("scored")
    other = task_factory("other", repeat_support={"weighted": "native"})
    policy = {
        "enabled": True,
        "mean": False,
        "pass_at_k": {"exact_match": [2], "typo": [2]},
    }
    configure_repeats(
        {"scored": scored, "other": other}, repeats=2, repeat_scoring=policy
    )
    assert enabled(scored) and not enabled(other)
    assert scored.config.repeat_scoring["pass_at_k"] == {"exact_match": [2]}
    assert "typo" in caplog.text
