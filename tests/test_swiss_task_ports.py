import json
from pathlib import Path

import pytest
from datasets import Dataset

from lm_eval.tasks._index import Kind, TaskIndex
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template


TASK_ROOT = Path(__file__).resolve().parents[1] / "lm_eval" / "tasks"
PORT_DIRS = [
    "global_mmlu/gen_0shot",
    "include/gen_0shot",
    "include_new/gen_0shot",
    "switzerland_qa/gen_0shot",
    "blend",
    "cultural_bench",
    "multi_if",
    "okapi/truthfulqa_multilingual",
]
NAMES = [
    "global_mmlu_gen_0shot",
    "include_base_44_gen_0shot",
    "include_base_new_45_gen_0shot",
    "switzerland_qa_0shot",
    "blend_sample",
    "cultural_bench",
    "multi-if",
    "truthfulqa_multilingual_mc2",
]


def test_imported_groups_resolve_every_child_and_yaml_function():
    index = TaskIndex.build([TASK_ROOT / name for name in PORT_DIRS])
    assert set(NAMES) <= index.keys()
    visited = set()

    def visit(name):
        assert name in index, f"Unresolved imported group member: {name}"
        if name in visited:
            return
        visited.add(name)
        entry = index[name]
        config = load_yaml(entry.yaml_path, resolve_func=True)
        if entry.kind == Kind.GROUP:
            for child in config["task"]:
                visit(child if isinstance(child, str) else child["task"])
        else:
            assert config["dataset_path"]
            assert config["output_type"] in {"generate_until", "multiple_choice"}
            for field in ["process_docs", "process_results"]:
                if field in config:
                    assert callable(config[field])

    for name in NAMES:
        visit(name)


@pytest.mark.parametrize(("answer", "letter"), [(0, "A"), (1, "B"), (2, "C"), (3, "D")])
def test_include44_converts_numeric_targets_to_generated_letters(answer, letter):
    config = load_yaml(
        TASK_ROOT / "include/gen_0shot/_base_template_yaml", resolve_func=True
    )
    assert apply_template(config["doc_to_target"], {"answer": answer}) == letter


def test_blend_filters_country_and_parses_choices_without_losing_question():
    from lm_eval.tasks.blend.utils import process_us

    dataset = Dataset.from_list(
        [
            {
                "country": country,
                "choices": json.dumps(
                    {"A": "one", "B": "two", "C": "three", "D": "four"}
                ),
                "prompt": 'Which food?\nProvide as JSON format {"answer": "A"}',
            }
            for country in ["US", "UK"]
        ]
    )
    rows = list(process_us(dataset))
    assert len(rows) == 1
    assert rows[0]["country"] == "US"
    assert rows[0]["choice_B"] == "two"
    assert rows[0]["clean_prompt"] == "Which food?"


def test_switzerland_topic_filter_preserves_only_requested_topic():
    from lm_eval.tasks.switzerland_qa.gen_0shot.utils import process_history

    rows = list(
        process_history(
            Dataset.from_list(
                [
                    {"topic": "history", "question": "When?"},
                    {"topic": "geography", "question": "Where?"},
                ]
            )
        )
    )
    assert rows == [{"topic": "history", "question": "When?"}]


def test_multi_if_uses_first_turn_and_scores_known_constraint():
    from lm_eval.tasks.multi_if.utils import process_docs, process_results

    rows = list(
        process_docs(
            Dataset.from_list(
                [
                    {
                        "key": 7,
                        "language": "English",
                        "turn_1_prompt": json.dumps({"content": "Say hello."}),
                        "turn_1_instruction_id_list": json.dumps(
                            ["detectable_format:json_format"]
                        ),
                        "turn_1_kwargs": json.dumps([json.dumps({})]),
                        "turn_2_prompt": json.dumps(
                            {
                                "content": "This is outside the imported first-turn protocol."
                            }
                        ),
                    }
                ]
            )
        )
    )
    assert rows[0]["prompt"] == "Say hello."
    assert rows[0]["key"] == 7
    assert process_results(rows[0], ['{"hello": "world"}'])["prompt_level_strict_acc"]
    assert not process_results(rows[0], ["plain prose"])["prompt_level_strict_acc"]
