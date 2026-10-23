import itertools
import json

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
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs}) or "templated"
        ),
    )

    assert prompt == "templated"
    assert seen["kwargs"]["tools"][0]["name"] == "calculate_triangle_area"
    assert seen["kwargs"]["tools"][0]["parameters"]["type"] == "object"
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][0]["content"]["text"] == ""
    assert "<available_function_name>" not in seen["messages"][0]["content"]["text"]
    assert "<parameter_name>" not in seen["messages"][0]["content"]["text"]
    assert seen["messages"][1]["content"]["parts"][0]["type"] == "text"


def test_bfcl_v3_apertus_fewshot_context_uses_custom_system_instruction():
    task = _make_apertus_task()
    doc = task.test_docs()[0]
    seen = {}

    task.fewshot_context(
        doc,
        num_fewshot=0,
        system_instruction="Use tools carefully.",
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs}) or "templated"
        ),
    )

    assert seen["messages"][0]["content"]["text"] == "Use tools carefully."


def test_bfcl_v3_apertus_fewshot_context_strips_null_tool_fields():
    task = _make_apertus_task("live_simple")
    doc = next(doc for doc in task.test_docs() if doc["id"] == "live_simple_70-34-0")
    seen = {}

    task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs}) or "templated"
        ),
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
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs}) or "templated"
        ),
    )
    properties = seen["kwargs"]["tools"][0]["parameters"]["properties"]
    assert properties["networkId"]["default"] == "[]"

    task = _make_apertus_task("live_irrelevance")
    doc = next(doc for doc in task.test_docs() if doc["id"] == "live_irrelevance_7-0-7")
    task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs}) or "templated"
        ),
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
        chat_template=lambda messages, **kwargs: (
            seen.update({"messages": messages, "kwargs": kwargs})
            or "templated-multiturn"
        ),
    )

    prompt, _ = task.multiturn_next_request(state)

    assert prompt == "templated-multiturn"
    assert seen["kwargs"]["tools"]
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][0]["content"]["text"] == ""
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
    assert "\n\n" not in request.arguments[1]["until"]
    assert request.arguments[1]["until"][0] == "<|assistant_end|>"


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
    task.config.output_type = "multi_turn_generate"
    task.OUTPUT_TYPE = "multi_turn_generate"
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


def test_bfcl_v3_multi_turn_uses_chat_template_on_every_step():
    task = _make_task("multi_turn_base")
    doc = task.test_docs()[0]
    seen = []
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.append(messages) or "templated-multiturn"
        ),
    )

    prompt, _ = task.multiturn_next_request(state)
    assert prompt == "templated-multiturn"
    assert [m["role"] for m in seen[-1]] == ["system", "user"]
    assert "Here is a list of functions" in seen[-1][0]["content"]

    task.multiturn_consume_response(state, "[pwd()]")
    prompt, _ = task.multiturn_next_request(state)
    assert prompt == "templated-multiturn"
    assert [m["role"] for m in seen[-1]] == ["system", "user", "assistant", "user"]
    assert seen[-1][2]["content"] == "[pwd()]"
    assert seen[-1][3]["content"].startswith("Execution results:")


def test_bfcl_v3_multi_turn_merges_back_to_back_user_messages():
    task = _make_task("multi_turn_base")
    doc = task.test_docs()[0]
    seen = []
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.append(messages) or "",
    )
    task.multiturn_next_request(state)
    state["history"].append({"role": "user", "content": "next turn"})

    task.multiturn_next_request(state)

    roles = [m["role"] for m in seen[-1]]
    assert all(a != b for a, b in itertools.pairwise(roles))
    assert seen[-1][-1]["content"].endswith("\n\nnext turn")


def test_bfcl_v3_multi_turn_without_chat_template_keeps_transcript():
    task = _make_task("multi_turn_base")
    doc = task.test_docs()[0]
    state = task.init_multiturn_state(doc, ctx="", gen_kwargs={})

    prompt, _ = task.multiturn_next_request(state)

    assert prompt.startswith("You are an expert in composing functions.")
    assert "\n\nUser: " in prompt
    assert prompt.endswith("\nAssistant:")


def test_bfcl_v3_construct_requests_stops_before_next_user_message():
    task = _make_task("multi_turn_base")
    doc = task.test_docs()[0]

    _, gen_kwargs = task.construct_requests(doc, "ctx").arguments

    assert gen_kwargs["until"] == ["\n\n", "\nUser:"]


def test_bfcl_v3_apertus_construct_requests_has_no_user_stop():
    task = _make_apertus_task("multi_turn_base")
    doc = task.test_docs()[0]

    _, gen_kwargs = task.construct_requests(doc, "ctx").arguments

    assert "\nUser:" not in gen_kwargs["until"]


def test_bfcl_v3_multi_turn_drops_empty_assistant_replies_for_chat_template():
    task = _make_task("multi_turn_base")
    doc = task.test_docs()[0]
    seen = []
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.append(messages) or "",
    )
    task.multiturn_next_request(state)
    task.multiturn_consume_response(state, "")

    task.multiturn_next_request(state)

    assert all(m["content"].strip() for m in seen[-1] if m["role"] == "assistant")
    roles = [m["role"] for m in seen[-1]]
    assert all(a != b for a, b in itertools.pairwise(roles))


def test_bfcl_v3_apertus_folds_doc_system_message_into_system_prompt():
    task = _make_apertus_task("live_simple")
    doc = {
        "id": "live_simple_x",
        "question": [
            [
                {"role": "system", "content": "Doc system rules."},
                {"role": "user", "content": "hi"},
            ]
        ],
        "function": [],
    }
    seen = {}

    task.fewshot_context(
        doc,
        num_fewshot=0,
        system_instruction="Base instruction.",
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: seen.update(messages=messages) or "",
    )

    assert [m["role"] for m in seen["messages"]] == ["system", "user"]
    assert seen["messages"][0]["content"]["text"] == (
        "Base instruction.\n\nDoc system rules."
    )


def _make_fc_task(category="simple", language="Python"):
    task = _make_apertus_task(category, language)
    task.bfcl_tool_format = "fc"
    return task


class _ChatPrompt(str):
    """Stands in for JsonChatStr: what a chat-completions model's template returns."""

    @property
    def prompt(self):
        return str(self)


def _fc_response(*calls, content=""):
    from lm_eval.tasks.bfcl_v3.task import TOOL_CALLS_MARKER

    tool_calls = [
        {
            "id": f"call{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(calls)
    ]
    return content + TOOL_CALLS_MARKER + json.dumps(tool_calls)


def test_bfcl_v3_fc_marker_matches_chat_model():
    from lm_eval.models.openai_completions import (
        TOOL_CALLS_MARKER as MODEL_MARKER,
    )
    from lm_eval.tasks.bfcl_v3.task import TOOL_CALLS_MARKER

    assert TOOL_CALLS_MARKER == MODEL_MARKER


def test_bfcl_v3_fc_fewshot_context_sends_messages_only():
    task = _make_fc_task("simple")
    doc = task.test_docs()[0]
    seen = {}

    prompt = task.fewshot_context(
        doc,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.update(messages=messages, kwargs=kwargs) or _ChatPrompt("msgs")
        ),
    )

    assert prompt == "msgs"
    assert seen["messages"] == doc["question"][0]
    assert seen["kwargs"] == {}


def test_bfcl_v3_fc_refuses_a_completions_model():
    task = _make_fc_task("simple")
    doc = task.test_docs()[0]

    with pytest.raises(ValueError, match="chat-completions model"):
        task.fewshot_context(
            doc,
            num_fewshot=0,
            apply_chat_template=True,
            chat_template=lambda messages, **kwargs: "<s>rendered text",
        )


def test_bfcl_v3_fc_construct_requests_sends_tools_with_api_safe_names():
    task = _make_fc_task("parallel")
    doc = task.test_docs()[0]

    _, gen_kwargs = task.construct_requests(doc, "ctx").arguments

    tools = json.loads(gen_kwargs["tools"])
    assert [tool["function"]["name"] for tool in tools] == ["spotify_play"]
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "\n\n" not in gen_kwargs["until"]


def test_bfcl_v3_fc_scores_tool_calls_and_maps_names_back():
    task = _make_fc_task("parallel")
    doc = task.test_docs()[0]
    response = _fc_response(
        ("spotify_play", {"artist": "Taylor Swift", "duration": 20}),
        ("spotify_play", {"artist": "Maroon 5", "duration": 15}),
    )

    assert task.process_results(doc, [response])["acc"] == 1


def test_bfcl_v3_fc_scores_simple_call():
    task = _make_fc_task("simple")
    doc = task.test_docs()[0]
    response = _fc_response(("calculate_triangle_area", {"base": 10, "height": 5}))

    assert task.process_results(doc, [response])["acc"] == 1


def test_bfcl_v3_fc_text_without_tool_calls_scores_zero():
    task = _make_fc_task("simple")
    doc = task.test_docs()[0]

    assert (
        task.process_results(doc, ["[calculate_triangle_area(base=10, height=5)]"])[
            "acc"
        ]
        == 0
    )


def test_bfcl_v3_fc_irrelevance_scores_absence_of_tool_calls():
    task = _make_fc_task("irrelevance")
    doc = task.test_docs()[0]

    assert task.process_results(doc, ["I can't help with that."])["acc"] == 1


def test_bfcl_v3_fc_java_scores_reference_call():
    task = _make_fc_task("java", language="Java")
    doc = task.test_docs()[0]
    response = _fc_response(
        (
            "GeometryPresentation_createPresentation",
            {"controller": "mapController", "parent": "mapArea"},
        )
    )

    assert task.process_results(doc, [response])["acc"] == 1


def test_bfcl_v3_fc_multi_turn_executes_calls_and_links_tool_messages():
    task = _make_fc_task("multi_turn_base")
    doc = task.test_docs()[0]
    seen = []
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={"until": [], "max_gen_toks": 2048},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.append(messages) or _ChatPrompt("msgs")
        ),
    )

    _, gen_kwargs = task.multiturn_next_request(state)
    names = [tool["function"]["name"] for tool in json.loads(gen_kwargs["tools"])]
    assert "cd" in names
    assert [m["role"] for m in seen[-1]] == ["user"]

    task.multiturn_consume_response(
        state, _fc_response(("cd", {"folder": "document"}), content="Going there.")
    )
    task.multiturn_next_request(state)

    assert [m["role"] for m in seen[-1]] == ["user", "assistant", "tool"]
    assert seen[-1][1]["tool_calls"][0]["id"] == "call0"
    assert seen[-1][2]["tool_call_id"] == "call0"
    assert "document" in seen[-1][2]["content"]
    assert state["decoded_responses"][0] == [["cd(folder='document')"]]

    task.multiturn_consume_response(state, "Done.")
    assert state["turn_idx"] == 1


def test_bfcl_v3_fc_miss_func_sends_held_out_functions_as_tools():
    task = _make_fc_task("multi_turn_miss_func")
    doc = task.test_docs()[0]
    held_out_turn = int(next(iter(doc["missed_function"])))
    seen = []
    state = task.init_multiturn_state(
        doc,
        ctx="",
        gen_kwargs={},
        apply_chat_template=True,
        chat_template=lambda messages, **kwargs: (
            seen.append(messages) or _ChatPrompt("msgs")
        ),
    )
    for _ in range(held_out_turn):
        task.multiturn_next_request(state)
        task.multiturn_consume_response(state, "No calls.")

    _, gen_kwargs = task.multiturn_next_request(state)

    held_out = doc["missed_function"][str(held_out_turn)]
    names = {tool["function"]["name"] for tool in json.loads(gen_kwargs["tools"])}
    assert {f["name"] for f in held_out} <= names
    assert seen[-1][-1]["content"].startswith("I have updated some more functions")
