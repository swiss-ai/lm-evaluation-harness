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


def _make_apertus_task(category="simple", language="Python"):
    return BFCLV3Task(
        config={
            "task": f"bfcl_v3_apertus_{category}",
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
                "bfcl_tool_format": "apertus",
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
    assert "bfcl_v3_apertus" in manager.all_groups
    assert "bfcl_v3_apertus_simple" in manager.all_tasks
    assert "bfcl_v3_apertus_java" in manager.all_tasks
    assert "bfcl_v3_apertus_javascript" in manager.all_tasks
    task = manager.load_task_or_group(["bfcl_v3_simple"])["bfcl_v3_simple"]
    assert len(task.test_docs()) == 400


def test_bfcl_v3_apertus_prompt_and_scores_reference_call():
    task = _make_apertus_task()
    doc = task.test_docs()[0]

    prompt = task.doc_to_text(doc)
    assert "<|tools_prefix|>" in prompt
    assert "<|developer_start|>" in prompt
    assert "calculate_triangle_area" in prompt
    assert task.process_results(
        doc,
        [
            '<|tools_prefix|>[{"calculate_triangle_area": '
            '{"base": 10, "height": 5}}]<|tools_suffix|>'
        ],
    )["acc"]
    assert not task.process_results(
        doc,
        ['<|tools_prefix|>[{"wrong": {"base": 10, "height": 5}}]<|tools_suffix|>'],
    )["acc"]


def test_bfcl_v3_apertus_fewshot_context_bypasses_outer_chat_template():
    task = _make_apertus_task()
    doc = task.test_docs()[0]

    prompt = task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **_: "WRAPPED:" + repr(messages),
    )

    assert prompt.startswith("<s><|system_start|>")
    assert "WRAPPED:" not in prompt
    assert "<|user_start|><s><|system_start|>" not in prompt
    assert prompt.count("<|assistant_start|>") == 1


def test_bfcl_v3_apertus_scores_nested_dict_call():
    task = _make_apertus_task()
    doc = {
        "function": [
            {
                "name": "update_user_info",
                "description": "Update user information in the database.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "update_info": {
                            "type": "dict",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        },
                        "database": {"type": "string", "default": "CustomerInfo"},
                    },
                    "required": ["user_id", "update_info"],
                },
            }
        ],
        "ground_truth": [
            {
                "update_user_info": {
                    "user_id": [43523],
                    "update_info": [
                        {"name": ["John Doe"], "email": ["johndoe@email.com"]}
                    ],
                    "database": ["CustomerInfo", ""],
                }
            }
        ],
    }

    assert task.process_results(
        doc,
        [
            '<|tools_prefix|>[{"update_user_info": {"user_id": 43523, '
            '"update_info": {"name": "John Doe", "email": "johndoe@email.com"}, '
            '"database": "CustomerInfo"}}]<|tools_suffix|>'
        ],
    )["acc"]


def test_bfcl_v3_apertus_irrelevance_scores_absence_of_tool_block():
    task = _make_apertus_task("irrelevance")
    doc = task.test_docs()[0]

    assert task.process_results(doc, ["I cannot use the provided functions."])["acc"]
    assert not task.process_results(
        doc,
        ['<|tools_prefix|>[{"some_function": {"foo": 1}}]<|tools_suffix|>'],
    )["acc"]


def test_bfcl_v3_apertus_java_scores_reference_call():
    task = _make_apertus_task("java", language="Java")
    doc = task.test_docs()[0]

    assert task.process_results(
        doc,
        [
            '<|tools_prefix|>[{"GeometryPresentation.createPresentation": '
            '{"controller": "mapController", "parent": "mapArea"}}]<|tools_suffix|>'
        ],
    )["acc"]


def test_bfcl_v3_apertus_javascript_scores_reference_call():
    task = _make_apertus_task("javascript", language="JavaScript")
    doc = task.test_docs()[0]

    assert task.process_results(
        doc,
        [
            '<|tools_prefix|>[{"validateUserInput": '
            '{"inputField": "userInputField", "isComplete": true}}]<|tools_suffix|>'
        ],
    )["acc"]


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


def test_bfcl_v3_apertus_multi_turn_decodes_to_executable_calls():
    task = _make_apertus_task("multi_turn_base")

    decoded = task._decode_multiturn_calls(
        '<|tools_prefix|>[{"cd": {"folder": "document"}}, '
        '{"mkdir": {"dir_name": "temp"}}]<|tools_suffix|>'
    )

    assert decoded == ["cd(folder='document')", "mkdir(dir_name='temp')"]
