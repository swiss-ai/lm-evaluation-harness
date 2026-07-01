from lm_eval.tasks import TaskManager
from lm_eval.tasks.olympiadbench import utils


def test_olympiadbench_task_manager_indexes_text_only_tasks():
    manager = TaskManager(include_path=None)

    assert "olympiadbench" in manager.all_tasks
    assert "olympiadbench_text_only" in manager.all_tasks
    assert "olympiadbench_oe_to_maths_en_comp" in manager.all_tasks
    assert "olympiadbench_oe_to_physics_zh_cee" in manager.all_tasks


def test_olympiadbench_prompt_matches_english_reference_shape():
    doc = {
        "question": "Compute 1+1.",
        "answer_type": "Numerical",
        "is_multiple_answer": False,
        "unit": None,
    }

    prompt = utils.make_prompt(doc, "OE_TO_maths_en_COMP")

    assert "International Math competition" in prompt
    assert "So the final answer is \\boxed{answer}." in prompt


def test_olympiadbench_extracts_reference_final_answer_phrase():
    output = "We compute the expression. So the final answer is \\boxed{2}."

    assert utils.extract_answer(output, is_chinese=False) == "\\boxed{2}."


def test_olympiadbench_process_results_scores_boxed_answer():
    doc = {
        "final_answer": ["2"],
        "error": None,
        "answer_type": "Numerical",
    }
    result = utils.process_results(
        doc,
        "We compute carefully. So the final answer is \\boxed{2}.",
        subset="OE_TO_maths_en_COMP",
    )

    assert result["exact_match"] == 1
    assert result["degeneration"] == 0


def test_olympiadbench_process_results_rejects_wrong_answer():
    doc = {
        "final_answer": ["2"],
        "error": None,
        "answer_type": "Numerical",
    }
    result = utils.process_results(
        doc,
        "We compute carefully. So the final answer is \\boxed{3}.",
        subset="OE_TO_maths_en_COMP",
    )

    assert result["exact_match"] == 0
