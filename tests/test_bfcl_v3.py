from lm_eval.tasks import TaskManager
from lm_eval.tasks.bfcl_v3.task import BFCLV3Task


def _make_task(category="simple", language="Python"):
    return BFCLV3Task(
        config={
            "task": f"bfcl_v3_{category}",
            "output_type": "generate_until",
            "test_split": "test",
            "num_fewshot": 0,
            "metric_list": [
                {
                    "metric": "acc",
                    "aggregation": "mean",
                    "higher_is_better": True,
                }
            ],
            "generation_kwargs": {
                "until": ["\n\n"],
                "max_gen_toks": 512,
                "do_sample": False,
            },
            "metadata": {
                "version": 3,
                "bfcl_category": category,
                "bfcl_language": language,
            },
        }
    )


def test_bfcl_v3_simple_scores_reference_call():
    task = _make_task()
    doc = task.test_docs()[0]

    assert doc["id"] == "simple_0"
    assert task.process_results(doc, ["calculate_triangle_area(base=10, height=5)"])[
        "acc"
    ]
    assert not task.process_results(doc, ["wrong(base=10, height=5)"])["acc"]


def test_bfcl_v3_irrelevance_scores_absence_of_function_call():
    task = _make_task("irrelevance")
    doc = task.test_docs()[0]

    assert task.process_results(doc, ["I cannot use the provided functions."])["acc"]
    assert not task.process_results(doc, ["some_function(foo=1)"])["acc"]


def test_bfcl_v3_task_manager_loads_registered_task():
    manager = TaskManager(include_path=None)

    assert "bfcl_v3_simple" in manager.all_tasks
    assert "bfcl_v3_multi_turn_base" in manager.all_tasks
    task = manager.load_task_or_group(["bfcl_v3_simple"])["bfcl_v3_simple"]
    assert len(task.test_docs()) == 400


def test_bfcl_v3_multi_turn_scores_ground_truth_calls():
    task = _make_task("multi_turn_base")
    task.config.output_type = "generate_until_multiturn"
    task.OUTPUT_TYPE = "generate_until_multiturn"
    doc = task.test_docs()[0]
    result = {
        "decoded_responses": [[turn] for turn in doc["ground_truth"]],
        "raw_responses": doc["ground_truth"],
        "execution_results": [],
    }

    assert doc["id"] == "multi_turn_base_0"
    assert task.process_results(doc, [result])["acc"]
