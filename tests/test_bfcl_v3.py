import pytest

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
                },
                {
                    "metric": "acc_lenient",
                    "aggregation": "mean",
                    "higher_is_better": True,
                },
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
                },
                {
                    "metric": "acc_lenient",
                    "aggregation": "mean",
                    "higher_is_better": True,
                },
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
    assert "<|tools_prefix|>" not in prompt
    assert "<|developer_start|>" not in prompt
    assert "Find the area of a triangle" in prompt
    results = task.process_results(
        doc,
        [
            '<|tools_prefix|>[{"calculate_triangle_area": '
            '{"base": 10, "height": 5}}]<|tools_suffix|>'
        ],
    )
    assert results["acc"]
    assert results["acc_lenient"]
    assert not task.process_results(
        doc,
        ['<|tools_prefix|>[{"wrong": {"base": 10, "height": 5}}]<|tools_suffix|>'],
    )["acc"]


def test_bfcl_v3_apertus_fewshot_context_uses_tool_chat_template():
    task = _make_apertus_task()
    doc = task.test_docs()[0]
    seen = {}

    prompt = task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(
            {"messages": messages, "kwargs": kwargs}
        )
        or "templated",
    )

    assert prompt == "templated"
    assert seen["kwargs"]["tools"][0]["name"] == "calculate_triangle_area"
    assert seen["kwargs"]["tools"][0]["parameters"]["type"] == "object"
    assert seen["messages"][0]["role"] == "system"
    assert "<available_function_name>" not in seen["messages"][0]["content"]["text"]
    assert "<parameter_name>" not in seen["messages"][0]["content"]["text"]
    assert seen["messages"][1]["content"]["parts"][0]["type"] == "text"


def test_bfcl_v3_apertus_fewshot_context_strips_null_tool_fields():
    task = _make_apertus_task("live_simple")
    doc = next(doc for doc in task.test_docs() if doc["id"] == "live_simple_70-34-0")
    seen = {}

    task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(
            {"messages": messages, "kwargs": kwargs}
        )
        or "templated",
    )

    properties = seen["kwargs"]["tools"][0]["parameters"]["properties"]
    assert "default" not in properties["startingAfter"]
    assert "default" not in properties["endingBefore"]
    assert properties["timespan"]["default"] == "86400"

    task = _make_apertus_task("live_multiple")
    doc = next(doc for doc in task.test_docs() if doc["id"] == "live_multiple_146-58-0")
    task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(
            {"messages": messages, "kwargs": kwargs}
        )
        or "templated",
    )
    properties = seen["kwargs"]["tools"][0]["parameters"]["properties"]
    assert properties["networkId"]["default"] == "[]"

    task = _make_apertus_task("live_irrelevance")
    doc = next(
        doc for doc in task.test_docs() if doc["id"] == "live_irrelevance_7-0-7"
    )
    task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(
            {"messages": messages, "kwargs": kwargs}
        )
        or "templated",
    )
    properties = seen["kwargs"]["tools"][0]["parameters"]["properties"]
    assert properties["auth"]["default"] == "[]"


def test_bfcl_v3_apertus_fewshot_context_requires_chat_template():
    task = _make_apertus_task()
    doc = task.test_docs()[0]

    with pytest.raises(ValueError, match="require --apply_chat_template"):
        task.fewshot_context(doc, num_fewshot=0, apply_chat_template=False)


def test_bfcl_v3_apertus_multi_turn_uses_tool_chat_template():
    task = _make_apertus_task("multi_turn_base")
    doc = task.test_docs()[0]
    seen = {}
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(
            {"messages": messages, "kwargs": kwargs}
        )
        or "templated-multiturn",
    )

    prompt, _ = task.multiturn_next_request(state)

    assert prompt == "templated-multiturn"
    assert seen["kwargs"]["tools"]
    assert seen["messages"][0]["role"] == "system"
    assert any(
        part["type"] == "text"
        for message in seen["messages"]
        for part in message.get("content", {}).get("parts", [])
    )


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


def test_bfcl_v3_apertus_scores_raw_json_call_without_tool_markers():
    task = _make_apertus_task()
    doc = next(doc for doc in task.test_docs() if doc["id"] == "simple_158")

    results = task.process_results(
        doc,
        [
            '{"get_criminal_records": {"name": "Mr. X", '
            '"location": "New York, NY", "from_year": 2012, "to_year": 2015}}}\n'
        ],
    )

    assert not results["acc"]
    assert results["acc_lenient"]


def test_bfcl_v3_apertus_scores_tool_block_without_suffix():
    task = _make_apertus_task()
    doc = next(doc for doc in task.test_docs() if doc["id"] == "simple_5")

    results = task.process_results(
        doc,
        [
            "I'll call the tool now."
            '<|tools_prefix|>[{"solve_quadratic": '
            '{"a": 3, "b": -11, "c": -4, "root_type": "all"}}]'
        ],
    )

    assert not results["acc"]
    assert results["acc_lenient"]


def test_bfcl_v3_apertus_scores_function_calls_tag():
    task = _make_apertus_task()
    doc = {
        "function": [
            {
                "name": "court_case.search",
                "description": "Search court cases.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "docket_number": {"type": "string"},
                        "location": {"type": "string"},
                        "full_text": {"type": "boolean"},
                    },
                    "required": ["docket_number", "location"],
                },
            }
        ],
        "ground_truth": [
            {
                "court_case.search": {
                    "docket_number": ["123456"],
                    "location": ["Texas"],
                    "full_text": [False],
                }
            }
        ],
    }

    results = task.process_results(
        doc,
        [
            "<function_calls>{\n"
            '"court_case.search": {\n'
            '"docket_number": "123456",\n'
            '"location": "Texas",\n'
            '"full_text": false\n'
            "}\n"
            "}</function_calls>"
        ],
    )

    assert not results["acc"]
    assert results["acc_lenient"]


def test_bfcl_v3_apertus_scores_json_call_after_thought_text():
    task = _make_apertus_task()
    doc = next(doc for doc in task.test_docs() if doc["id"] == "simple_158")

    results = task.process_results(
        doc,
        [
            "<think>Okay, let's tackle this request.</think>\n"
            '{"get_criminal_records": {"name": "Mr. X", '
            '"location": "New York, NY", "from_year": 2012, "to_year": 2015}}'
        ],
    )

    assert not results["acc"]
    assert results["acc_lenient"]


def test_bfcl_v3_apertus_construct_requests_uses_larger_generation_budget():
    task = _make_apertus_task()
    doc = task.test_docs()[0]

    request = task.construct_requests(doc, ctx="prompt")

    assert request.arguments[1]["max_gen_toks"] == 2048


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
