from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.bfcl_v3.ast_checker import ast_checker
from lm_eval.tasks.bfcl_v3.multi_turn_checker import multi_turn_checker
from lm_eval.tasks.bfcl_v3.multi_turn_utils import (
    execute_multi_turn_func_call,
    extract_function_call_strings,
    instantiate_classes,
)


VERSION_PREFIX = "BFCL_v3"
DATA_DIR = Path(__file__).parent / "data"
POSSIBLE_ANSWER_DIR = DATA_DIR / "possible_answer"
MULTI_TURN_FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"
MAXIMUM_STEP_LIMIT = 20

MULTI_TURN_FUNC_DOC_FILE_MAPPING = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
"""

DEFAULT_MULTI_TURN_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + "\nAt each turn, you should try your best to complete the tasks requested by "
    "the user within the current turn. Continue to output functions to call until "
    "you have fulfilled the user's request to the best of your ability. Once you "
    "have no more functions to call, the system will consider the current turn "
    "complete and proceed to the next turn or task.\n"
)

DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC = (
    "I have updated some more functions you can choose from. What about now?"
)


class _ListDocs(list):
    @property
    def features(self) -> dict[str, Any]:
        if not self:
            return {}
        return {key: None for key in self[0]}


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _sort_key(entry: dict) -> tuple[str, int]:
    category, index = entry["id"].rsplit("_", 1)
    if "-" in index:
        index = index.split("-")[0]
    return category, int(index)


def _category_file(category: str, answers: bool = False) -> Path:
    base = POSSIBLE_ANSWER_DIR if answers else DATA_DIR
    return base / f"{VERSION_PREFIX}_{category}.json"


def _is_multi_turn(category: str) -> bool:
    return category.startswith("multi_turn")


def _flatten_messages(messages: list[dict]) -> str:
    rendered = []
    for message in messages:
        role = message.get("role", "user").capitalize()
        rendered.append(f"{role}: {message.get('content', '')}")
    return "\n".join(rendered)


def _language_hint(category: str) -> str:
    if category == "java":
        return " Note that the provided function is in Java 8 SDK syntax."
    if category == "javascript":
        return " Note that the provided function is in JavaScript syntax."
    return " Note that the provided function is in Python 3 syntax."


def _prompt_functions(functions: list[dict], category: str) -> list[dict]:
    functions = deepcopy(functions)
    hint = _language_hint(category)
    for function in functions:
        function["description"] = function.get("description", "") + hint
    return functions


def _load_multi_turn_function_docs(involved_classes: list[str]) -> list[dict]:
    functions = []
    for class_name in involved_classes:
        functions.extend(
            _load_jsonl(
                MULTI_TURN_FUNC_DOC_DIR / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]
            )
        )
    return functions


def _prepare_multi_turn_doc(doc: dict) -> dict:
    doc["function"] = _load_multi_turn_function_docs(doc["involved_classes"])
    if "missed_function" not in doc:
        return doc

    for turn_index, missed_func_names in doc["missed_function"].items():
        doc["missed_function"][turn_index] = []
        for missed_func_name in missed_func_names:
            for func_index, func_doc in enumerate(doc["function"]):
                if func_doc["name"] == missed_func_name:
                    doc["missed_function"][turn_index].append(func_doc)
                    doc["function"].pop(func_index)
                    break
    return doc


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip("`\n ")


def _parse_python_calls(text: str) -> list[dict]:
    text = _strip_code_fence(text)
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text += "]"
    parsed = ast.parse(text, mode="eval")
    calls = parsed.body.elts if isinstance(parsed.body, ast.List) else [parsed.body]
    return [_resolve_ast_call(call) for call in calls if isinstance(call, ast.Call)]


def _parse_java_calls(text: str) -> list[dict]:
    from lm_eval.tasks.bfcl_v3.java_parser import parse_java_function_call

    text = _strip_code_fence(text)
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text += "]"
    return parse_java_function_call(text[1:-1])


def _parse_js_calls(text: str) -> list[dict]:
    from lm_eval.tasks.bfcl_v3.js_parser import parse_javascript_function_call

    text = _strip_code_fence(text)
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text += "]"
    return parse_javascript_function_call(text[1:-1])


def _resolve_ast_call(elem: ast.Call) -> dict:
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    return {
        func_name: {arg.arg: _resolve_ast_value(arg.value) for arg in elem.keywords}
    }


def _resolve_ast_value(value: ast.AST):
    if isinstance(value, ast.Constant):
        return "..." if value.value is Ellipsis else value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        operand = _resolve_ast_value(value.operand)
        return -operand if isinstance(operand, (int, float)) else ast.unparse(value)
    if isinstance(value, ast.List):
        return [_resolve_ast_value(v) for v in value.elts]
    if isinstance(value, ast.Tuple):
        return tuple(_resolve_ast_value(v) for v in value.elts)
    if isinstance(value, ast.Dict):
        return {
            _resolve_ast_value(k): _resolve_ast_value(v)
            for k, v in zip(value.keys, value.values, strict=True)
        }
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        return ast.unparse(value) if not value.keywords else _resolve_ast_call(value)
    if isinstance(value, ast.Ellipsis):
        return "..."
    if isinstance(value, ast.Subscript):
        return ast.unparse(value)
    raise TypeError(f"Unsupported AST type: {type(value)}")


def _is_function_calling_format(decoded_output) -> bool:
    if not isinstance(decoded_output, list):
        return False
    for item in decoded_output:
        if not isinstance(item, dict) or len(item) != 1:
            return False
        if not isinstance(next(iter(item.values())), dict):
            return False
    return True


def _is_empty_output(decoded_output) -> bool:
    if not _is_function_calling_format(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    return len(decoded_output) == 1 and len(decoded_output[0]) == 0


class BFCLV3Task(ConfigurableTask):
    VERSION = 3
    MAX_MULTITURN_STEPS = 128

    def __init__(self, config: dict | None = None, **kwargs):
        config = dict(config or {})
        config.pop("class", None)
        metadata = dict(config.get("metadata") or {})
        self.bfcl_category = metadata["bfcl_category"]
        self.bfcl_language = metadata.get("bfcl_language", "Python")
        super().__init__(config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        prompts = sorted(_load_jsonl(_category_file(self.bfcl_category)), key=_sort_key)
        answer_path = _category_file(self.bfcl_category, answers=True)
        answers = (
            sorted(_load_jsonl(answer_path), key=_sort_key)
            if answer_path.exists()
            else []
        )
        answer_by_id = {entry["id"]: entry for entry in answers}
        docs = []
        for prompt in prompts:
            doc = dict(prompt)
            if _is_multi_turn(self.bfcl_category):
                doc = _prepare_multi_turn_doc(doc)
            if prompt["id"] in answer_by_id:
                doc["ground_truth"] = answer_by_id[prompt["id"]]["ground_truth"]
            docs.append(doc)
        self.dataset = {"test": _ListDocs(docs)}

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        functions = _prompt_functions(doc.get("function", []), self.bfcl_category)
        function_text = json.dumps(functions, ensure_ascii=False)
        messages = doc["question"][0]
        system_prompt = (
            DEFAULT_MULTI_TURN_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
            if _is_multi_turn(self.bfcl_category)
            else DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
        )
        return (
            f"{system_prompt}\n"
            f"Here is a list of functions in JSON format that you can invoke.\n"
            f"{function_text}\n\n"
            f"{_flatten_messages(messages)}\nAssistant:"
        )

    def doc_to_target(self, doc):
        return ""

    def construct_requests(
        self, doc, ctx, chat_template=None, apply_chat_template=False, **kwargs
    ):
        arguments = deepcopy(self.config.generation_kwargs)
        arguments.setdefault("until", ["\n\n"])
        arguments.setdefault("max_gen_toks", 512)
        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(ctx, arguments),
            idx=0,
            **kwargs,
        )

    def init_multiturn_state(self, doc: dict, ctx: str, gen_kwargs: dict) -> dict:
        return {
            "doc": doc,
            "gen_kwargs": gen_kwargs,
            "functions": deepcopy(doc["function"]),
            "history": [],
            "turn_idx": 0,
            "turn_started": False,
            "done": False,
            "raw_responses": [],
            "decoded_responses": [],
            "execution_results": [],
            "instances": instantiate_classes(
                doc["initial_config"],
                doc["involved_classes"],
                long_context="long_context" in self.bfcl_category,
            ),
        }

    def multiturn_is_done(self, state: dict) -> bool:
        return state["done"]

    def multiturn_next_request(self, state: dict) -> tuple[str, dict] | None:
        if state["done"]:
            return None
        self._ensure_turn_started(state)
        if state["done"]:
            return None
        return self._render_multiturn_prompt(state), deepcopy(state["gen_kwargs"])

    def multiturn_consume_response(self, state: dict, response: str) -> None:
        if state["done"]:
            return
        state["history"].append({"role": "assistant", "content": response})
        state["raw_responses"][-1].append(response)

        try:
            decoded = extract_function_call_strings(response)
        except Exception:  # noqa: BLE001 - unparsable model text is a failed turn.
            decoded = []

        if not decoded:
            self._finish_turn(state)
            return

        state["decoded_responses"][-1].append(decoded)
        execution_results, _ = execute_multi_turn_func_call(
            decoded,
            instances=state["instances"],
            long_context="long_context" in self.bfcl_category,
        )
        state["execution_results"][-1].append(execution_results)
        state["history"].append(
            {
                "role": "user",
                "content": "Execution results:\n"
                + json.dumps(execution_results, ensure_ascii=False),
            }
        )

        if len(state["raw_responses"][-1]) >= MAXIMUM_STEP_LIMIT:
            self._finish_turn(state)

    def multiturn_result(self, state: dict) -> dict:
        return {
            "raw_responses": state["raw_responses"],
            "decoded_responses": state["decoded_responses"],
            "execution_results": state["execution_results"],
        }

    def _ensure_turn_started(self, state: dict) -> None:
        while not state["turn_started"]:
            if state["turn_idx"] >= len(state["doc"]["question"]):
                state["done"] = True
                return
            messages = state["doc"]["question"][state["turn_idx"]]
            holdout_functions = (
                state["doc"].get("missed_function", {}).get(str(state["turn_idx"]))
            )
            if holdout_functions is not None:
                state["functions"].extend(holdout_functions)
                messages = [
                    {
                        "role": "user",
                        "content": json.dumps(holdout_functions, ensure_ascii=False)
                        + "\n"
                        + DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]
            if not messages:
                state["turn_idx"] += 1
                continue
            state["history"].extend(messages)
            state["raw_responses"].append([])
            state["decoded_responses"].append([])
            state["execution_results"].append([])
            state["turn_started"] = True

    def _finish_turn(self, state: dict) -> None:
        state["turn_idx"] += 1
        state["turn_started"] = False
        if state["turn_idx"] >= len(state["doc"]["question"]):
            state["done"] = True

    def _render_multiturn_prompt(self, state: dict) -> str:
        functions = _prompt_functions(state["functions"], self.bfcl_category)
        function_text = json.dumps(functions, ensure_ascii=False)
        return (
            f"{DEFAULT_MULTI_TURN_SYSTEM_PROMPT_WITHOUT_FUNC_DOC}\n"
            f"Here is a list of functions in JSON format that you can invoke.\n"
            f"{function_text}\n\n"
            f"{_flatten_messages(state['history'])}\nAssistant:"
        )

    def _decode(self, text: str) -> list[dict]:
        if self.bfcl_language == "Java":
            return _parse_java_calls(text)
        if self.bfcl_language == "JavaScript":
            return _parse_js_calls(text)
        return _parse_python_calls(text)

    def process_results(self, doc, results):
        raw = results[0]
        if _is_multi_turn(self.bfcl_category):
            checker_result = multi_turn_checker(
                raw["decoded_responses"],
                doc["ground_truth"],
                doc,
                self.bfcl_category,
                model_name="prompt",
            )
            return {"acc": int(checker_result["valid"])}

        try:
            decoded = self._decode(raw)
        except Exception:  # noqa: BLE001 - unparsable model text scores as incorrect.
            decoded = None

        if "relevance" in self.bfcl_category or "irrelevance" in self.bfcl_category:
            contains_call = decoded is not None and not _is_empty_output(decoded)
            if "irrelevance" in self.bfcl_category:
                return {"acc": int(not contains_call)}
            return {"acc": int(contains_call)}

        if decoded is None or not _is_function_calling_format(decoded):
            return {"acc": 0}

        checker_result = ast_checker(
            doc["function"],
            decoded,
            doc["ground_truth"],
            self.bfcl_language,
            self.bfcl_category,
            model_name="prompt",
        )
        return {"acc": int(checker_result["valid"])}

    def aggregation(self):
        return {"acc": np.mean}

    def higher_is_better(self):
        return {"acc": True}
