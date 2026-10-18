"""Tests for Multi-IF single-turn (`multi-if`) and multi-turn (`multi-if-multiturn`)."""

import json
from pathlib import Path

from datasets import Dataset

from lm_eval.api.model import LM
from lm_eval.api.task import ConfigurableTask
from lm_eval.evaluator import evaluate
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.multi_if import utils


def _turn(content, ids, kwargs):
    return {
        "prompt": json.dumps({"role": "user", "content": content}),
        "ids": json.dumps(ids),
        # raw format: JSON list of JSON-encoded dicts, with null for unused args
        "kwargs": json.dumps([json.dumps(k) for k in kwargs]),
    }


def _raw_row(key, turns, language="English"):
    row = {"key": key, "language": language, "turns": None, "responses": None}
    for t in (1, 2, 3):
        turn = turns[t - 1] if t <= len(turns) else None
        row[f"turn_{t}_prompt"] = turn["prompt"] if turn else None
        row[f"turn_{t}_instruction_id_list"] = turn["ids"] if turn else None
        row[f"turn_{t}_kwargs"] = turn["kwargs"] if turn else None
    return row


# Instructions accumulate across turns, as in the real dataset.
T1 = _turn("Q1. Do not use commas.", ["punctuation:no_comma"], [{"end_phrase": None}])
T2 = _turn(
    "End with 'bye.'",
    ["punctuation:no_comma", "startend:end_checker"],
    [{"end_phrase": None}, {"end_phrase": "bye."}],
)
T3 = _turn(
    "Mention apple.",
    ["punctuation:no_comma", "startend:end_checker", "keywords:existence"],
    [{"end_phrase": None}, {"end_phrase": "bye."}, {"keywords": ["apple"]}],
)
RAW_ROWS = [_raw_row("three", [T1, T2, T3]), _raw_row("two", [T1, T2])]


def _metric_list(yaml_name):
    """metric_list from the real task YAML, so the tests stay in sync with it."""
    return load_yaml(Path(utils.__file__).parent / yaml_name)["metric_list"]


def _docs():
    return list(utils.process_docs(Dataset.from_list(RAW_ROWS)))


# ---------------------------------------------------------------------------
# process_docs
# ---------------------------------------------------------------------------


def test_process_docs_keeps_all_non_empty_turns():
    three, two = _docs()
    assert len(three["prompts"]) == 3
    assert len(two["prompts"]) == 2
    assert three["prompt"] == "Q1. Do not use commas."
    assert three["instruction_ids"][2][-1] == "keywords:existence"


# ---------------------------------------------------------------------------
# process_results
# ---------------------------------------------------------------------------


def test_single_turn_result_keeps_original_metric_names():
    out = utils.process_results(_docs()[0], ["no commas here"])
    assert out["prompt_level_strict_acc"] is True
    assert out["inst_level_strict_acc"] == [True]
    assert not any(k.startswith("turn_") for k in out)


def test_multi_turn_constraints_are_cumulative():
    # Turn 3 adds "apple" but breaks the turn-1 no-comma rule -> turn 3 fails.
    out = utils.process_results(_docs()[0], [["fine", "ok bye.", "apple, bye."]])
    assert out["turn_1_prompt_level_strict_acc"] is True
    assert out["turn_2_prompt_level_strict_acc"] is True
    assert out["turn_3_prompt_level_strict_acc"] is False
    assert out["turn_3_inst_level_strict_acc"] == [False, True, True]


def test_unplayed_turns_emit_no_metrics():
    out = utils.process_results(_docs()[1], [["fine", "ok bye."]])
    assert "turn_2_prompt_level_strict_acc" in out
    assert not any(k.startswith("turn_3_") for k in out)


def test_overall_and_per_language_keys():
    single = utils.process_results(_docs()[0], ["no commas here"])
    assert single["overall"] == single["overall_english"]
    assert not any(k.startswith("overall_") and k != "overall_english" for k in single)

    multi = utils.process_results(_docs()[0], [["fine", "ok bye.", "apple bye."]])
    for t in (1, 2, 3):
        assert f"turn_{t}_overall" in multi and f"turn_{t}_overall_english" in multi


def test_agg_overall_is_mean_of_four_aggregated_scores():
    items = [
        {
            "prompt_level_strict_acc": True,
            "inst_level_strict_acc": [True, True],
            "prompt_level_loose_acc": True,
            "inst_level_loose_acc": [True, True],
        },
        {
            "prompt_level_strict_acc": False,
            "inst_level_strict_acc": [True, False, False, False],
            "prompt_level_loose_acc": True,
            "inst_level_loose_acc": [True, True, True, True],
        },
    ]
    # strict: prompt 1/2, inst 3/6; loose: prompt 2/2, inst 6/6
    assert utils.agg_overall(items) == (0.5 + 0.5 + 1.0 + 1.0) / 4


# ---------------------------------------------------------------------------
# End-to-end rollout through the evaluator
# ---------------------------------------------------------------------------


class _FakeChatLM(LM):
    """Replies based on how many user turns the rendered prompt contains."""

    REPLIES = ("fine", "ok bye.", "apple bye.")

    def __init__(self):
        super().__init__()
        self.prompts = []

    @property
    def tokenizer_name(self):
        return "fake"

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return "\n".join(f"{m['role']}: {m['content']}" for m in chat_history)

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        prompts = [r.arguments[0] for r in requests]
        self.prompts.extend(prompts)
        return [self.REPLIES[p.count("user: ") - 1] for p in prompts]


class _LocalMultiIFTask(utils.MultiIFTask):
    rows = RAW_ROWS

    def download(self, *args, **kwargs):
        self.dataset = {"train": Dataset.from_list(self.rows)}


def _multiturn_task(rows=RAW_ROWS):
    task_cls = type("_Task", (_LocalMultiIFTask,), {"rows": rows})
    return task_cls(
        config={
            "task": "multi-if-multiturn-test",
            "class": _LocalMultiIFTask,
            "output_type": "multi_turn_generate",
            "test_split": "train",
            "num_fewshot": 0,
            "process_docs": utils.process_docs,
            "process_results": utils.process_results,
            "doc_to_text": "prompt",
            "doc_to_target": 0,
            "generation_kwargs": {"until": [], "do_sample": False},
            "metric_list": _metric_list("multi-if-multiturn.yaml"),
            "metadata": {"version": 1.0},
        }
    )


def test_rollout_feeds_back_history_and_scores_each_turn():
    lm = _FakeChatLM()
    results = evaluate(
        lm=lm,
        task_dict={"multi-if-multiturn-test": _multiturn_task()},
        bootstrap_iters=0,
        apply_chat_template=True,
        log_samples=True,
    )

    # 3 turns for the 3-turn doc, 2 for the 2-turn doc.
    assert len(lm.prompts) == 5
    # Turn-3 prompt contains the model's own earlier replies.
    last = max(lm.prompts, key=len)
    assert "assistant: fine" in last and "assistant: ok bye." in last
    assert last.endswith("user: Mention apple.")

    res = results["results"]["multi-if-multiturn-test"]
    assert res["turn_1_prompt_level_strict_acc,none"] == 1.0
    assert res["turn_2_prompt_level_strict_acc,none"] == 1.0
    # Only the 3-turn doc contributes to turn 3.
    assert res["turn_3_prompt_level_strict_acc,none"] == 1.0
    assert res["turn_3_inst_level_strict_acc,none"] == 1.0
    assert res["turn_3_overall,none"] == 1.0
    assert res["turn_3_overall_english,none"] == 1.0
    # languages with no docs report nothing
    assert "turn_1_overall_french,none" not in res


def test_max_turns_caps_rollout(monkeypatch):
    monkeypatch.setattr(utils, "MAX_TURNS", 1)
    lm = _FakeChatLM()
    results = evaluate(
        lm=lm,
        task_dict={"multi-if-multiturn-test": _multiturn_task()},
        bootstrap_iters=0,
        apply_chat_template=True,
    )
    assert len(lm.prompts) == 2  # one turn per doc
    res = results["results"]["multi-if-multiturn-test"]
    assert res["turn_1_prompt_level_strict_acc,none"] == 1.0
    assert "turn_2_prompt_level_strict_acc,none" not in res


class _LocalSingleTurnTask(ConfigurableTask):
    def download(self, *args, **kwargs):
        self.dataset = {"train": Dataset.from_list(RAW_ROWS)}


def test_single_turn_task_end_to_end():
    """`multi-if` (generate_until) still runs: one generation per doc, turn 1 only."""
    task = _LocalSingleTurnTask(
        config={
            "task": "multi-if-test",
            "output_type": "generate_until",
            "test_split": "train",
            "num_fewshot": 0,
            "process_docs": utils.process_docs,
            "process_results": utils.process_results,
            "doc_to_text": "prompt",
            "doc_to_target": 0,
            "generation_kwargs": {"until": [], "do_sample": False},
            "metric_list": _metric_list("multi-if.yaml"),
            "metadata": {"version": 1.0},
        }
    )
    lm = _FakeChatLM()
    results = evaluate(lm=lm, task_dict={"multi-if-test": task}, bootstrap_iters=0)

    assert len(lm.prompts) == 2  # one generation per doc
    assert all(p.startswith("Q1.") for p in lm.prompts)  # turn-1 prompt only
    res = results["results"]["multi-if-test"]
    assert res["prompt_level_strict_acc,none"] == 1.0
    assert res["overall,none"] == res["overall_english,none"] == 1.0
    assert not any(k.startswith("turn_") for k in res)


def test_per_language_scores_only_use_that_language():
    # 1-turn French conversation the fake model fails (it never says "banane").
    fr = _turn("Q1 fr.", ["keywords:existence"], [{"keywords": ["banane"]}])
    rows = RAW_ROWS + [_raw_row("fr", [fr], language="French")]
    results = evaluate(
        lm=_FakeChatLM(),
        task_dict={"multi-if-multiturn-test": _multiturn_task(rows)},
        bootstrap_iters=0,
        apply_chat_template=True,
    )
    res = results["results"]["multi-if-multiturn-test"]
    assert res["turn_1_overall_english,none"] == 1.0  # 2 English convs, all pass
    assert res["turn_1_overall_french,none"] == 0.0  # 1 French conv, fails
    # pooled: 2 of 3 convs and 2 of 3 instructions pass, strict and loose
    assert abs(res["turn_1_overall,none"] - 2 / 3) < 1e-9
    # the French conversation has 1 turn, so no French turn-2 score
    assert "turn_2_overall_french,none" not in res
    assert res["turn_2_overall_english,none"] == 1.0
