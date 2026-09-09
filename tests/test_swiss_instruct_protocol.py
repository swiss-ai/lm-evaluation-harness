"""Regressions from the full Apertus 8B run, using real task helpers."""

from pathlib import Path

import pytest

from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig
from lm_eval.filters.extraction import OrderedRegexFilter
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template


ROOT = Path(__file__).resolve().parents[1] / "lm_eval/tasks"


def test_mmlu_subjects_request_concise_final_answers():
    paths = list((ROOT / "mmlu/flan_cot_zeroshot").glob("mmlu_*.yaml"))
    assert len(paths) == 57
    for path in paths:
        cfg = load_yaml(path, resolve_func=True)
        assert "The answer is (X)" in cfg["description"]
        assert "concise" in cfg["description"]
        assert "\nA: Let's think" not in cfg["doc_to_text"]


@pytest.mark.parametrize(
    "path",
    [
        "mmlu/flan_cot_zeroshot/_mmlu_flan_cot_zeroshot_template_yaml",
        "mmlu_pro/_default_template_yaml",
    ],
)
def test_mmlu_ordered_extractor_prefers_explicit_final_answer(path):
    cfg = load_yaml(ROOT / path, resolve_func=True)
    selected = next(
        (f for f in cfg["filter_list"] if f["name"] == "ordered-extract"), None
    )
    assert selected is not None
    params = dict(selected["filter"][0])
    assert params.pop("function") == "ordered_regex"
    assert OrderedRegexFilter(**params).apply(
        [["Option (A) fails. The final answer is (B)."]], [{}]
    ) == [["B"]]


def test_mmlu_pro_keeps_fewshot_answer_in_assistant_role():
    cfg = load_yaml(ROOT / "mmlu_pro/_default_template_yaml", resolve_func=True)
    doc = {
        "question": "Which?",
        "options": [" one ", "two"],
        "cot_content": "A: Let's think step by step. Unique reasoning. The answer is (B).",
    }
    question = cfg["fewshot_config"]["doc_to_text"](doc)
    answer = cfg["fewshot_config"]["doc_to_target"](doc)
    assert "Unique reasoning" not in question
    assert "Unique reasoning" in answer
    assert "(A) one" in question


@pytest.mark.parametrize(
    "path,doc",
    [
        ("mathqa/mathqa.yaml", {"Problem": "Two plus two?"}),
        (
            "drop/default.yaml",
            {"passage": "Ada scored four points.", "question": "How many points?"},
        ),
    ],
)
def test_answer_prefix_is_an_assistant_message(path, doc):
    cfg = load_yaml(ROOT / path, resolve_func=True)
    task = object.__new__(ConfigurableTask)
    task._config = TaskConfig(**cfg)
    task.fewshot_cfg = task.config.fewshot_config
    prompt = apply_template(cfg["doc_to_text"], doc)
    messages = task.build_qa_turn(q=prompt, gen_prefix=cfg.get("gen_prefix"))
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[-1].content.strip() == "Answer:"
    assert "Answer:" not in messages[0].content


def test_drop_demonstrations_use_actual_answers_and_keep_scorer():
    cfg = load_yaml(ROOT / "drop/default.yaml", resolve_func=True)
    assert cfg["num_fewshot"] == 3
    assert cfg["description"] == "Return only the answer. No explanation."
    doc = {
        "answers": [("Ada", "Bob")],
        "answer": {"number": "", "spans": ["Ada", "Bob"]},
    }
    assert cfg["doc_to_target"](doc) == "Ada, Bob"
    assert cfg["process_results"](doc, ["Ada", "Bob"])["f1"] == 1.0


@pytest.mark.parametrize("unknown_position", [0, 1, 2])
def test_bbq_unknown_target_never_accepts_concrete_answers(unknown_position):
    from lm_eval.tasks.bbq.utils import doc_to_choice, doc_to_targets

    answers = ["Ada", "Bob"]
    answers.insert(unknown_position, "Unknown")
    doc = {
        **{f"ans{i}": answer for i, answer in enumerate(answers)},
        "label": unknown_position,
    }
    choices = doc_to_choice(doc)
    accepted = doc_to_targets(doc)
    assert all((i in accepted) == (i >= 2) for i in range(len(choices)))
