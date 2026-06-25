import math

from lm_eval.api.model import LM
from lm_eval.evaluator import evaluate
from lm_eval.tasks import TaskManager
from lm_eval.tasks.mt_bench import metric
from lm_eval.tasks.mt_bench.task import MTBenchTask


class ScriptedLM(LM):
    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        return [f"answer for {request.arguments[0]}" for request in requests]


def _make_task():
    return MTBenchTask(
        config={
            "task": "mt_bench",
            "output_type": "multi_turn_generate",
            "test_split": "test",
            "num_fewshot": 0,
            "metric_list": [
                {
                    "metric": "mt_bench_score",
                    "aggregation": metric.mt_bench_score,
                    "higher_is_better": True,
                },
                {
                    "metric": "mt_bench_turn_1_score",
                    "aggregation": metric.mt_bench_turn_1_score,
                    "higher_is_better": True,
                },
                {
                    "metric": "mt_bench_turn_2_score",
                    "aggregation": metric.mt_bench_turn_2_score,
                    "higher_is_better": True,
                },
                {
                    "metric": "mt_bench_judge_success_rate",
                    "aggregation": metric.mt_bench_judge_success_rate,
                    "higher_is_better": True,
                },
                {
                    "metric": "avg_word_count",
                    "aggregation": "mean",
                    "higher_is_better": False,
                },
            ],
            "generation_kwargs": {"until": [], "max_gen_toks": 1024},
            "metadata": {
                "version": 1,
                "use_reference_temperature_config": True,
            },
        }
    )


def test_mt_bench_task_manager_loads_registered_task():
    manager = TaskManager(include_path=None)

    assert "mt_bench" in manager.all_tasks
    task = manager.load_task_or_group(["mt_bench"])["mt_bench"]
    assert len(task.test_docs()) == 80


def test_mt_bench_default_judge_model_matches_hosted_qwen(monkeypatch):
    monkeypatch.delenv("MT_BENCH_JUDGE_MODEL", raising=False)

    assert metric._judge_model() == "Qwen/Qwen3.5-27B"


def test_mt_bench_multiturn_uses_chat_template_when_enabled():
    task = _make_task()
    doc = task.test_docs()[0]

    def chat_template(messages, add_generation_prompt=True):
        rendered = "|".join(
            f"{message['role']}={message['content']}" for message in messages
        )
        return f"templated:{rendered}:gen={add_generation_prompt}"

    state = task.init_multiturn_state(
        doc,
        "",
        {"until": []},
        apply_chat_template=True,
        chat_template=chat_template,
    )
    prompt, _ = task.multiturn_next_request(state)

    assert prompt.startswith("templated:user=")
    assert prompt.endswith(":gen=True")
    assert doc["turns"][0] in prompt


def test_mt_bench_multiturn_collects_two_responses(monkeypatch):
    monkeypatch.setattr(
        metric,
        "_run_judging",
        lambda items: [
            {"question_id": items[0]["question_id"], "turn": 1, "score": 8.0},
            {"question_id": items[0]["question_id"], "turn": 2, "score": 6.0},
        ],
    )

    task = _make_task()
    task.build_all_requests(rank=0, world_size=1, limit=1)
    results = evaluate(
        lm=ScriptedLM(),
        task_dict={"mt_bench": task},
        limit=1,
        bootstrap_iters=0,
    )

    assert results["results"]["mt_bench"]["mt_bench_score,none"] == 7.0
    assert results["results"]["mt_bench"]["mt_bench_turn_1_score,none"] == 8.0
    assert results["results"]["mt_bench"]["mt_bench_turn_2_score,none"] == 6.0


def test_mt_bench_score_ignores_failed_judge_rows(monkeypatch):
    monkeypatch.setattr(
        metric,
        "_run_judging",
        lambda items: [
            {"turn": 1, "score": 9.0},
            {"turn": 2, "score": math.nan},
        ],
    )

    assert metric.mt_bench_score([{"question_id": 1}]) == 9.0
    assert metric.mt_bench_turn_1_score([{"question_id": 1}]) == 9.0
    assert math.isnan(metric.mt_bench_turn_2_score([{"question_id": 1}]))
    assert metric.mt_bench_judge_success_rate([{"question_id": 1}]) == 0.5
