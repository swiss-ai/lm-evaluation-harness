from __future__ import annotations

import ast
import copy
import importlib
import inspect
import json
import re
from typing import Any


CLASS_FILE_PATH_MAPPING = {
    "GorillaFileSystem": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.gorilla_file_system",
    "MathAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.math_api",
    "MessageAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.message_api",
    "TwitterAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.posting_api",
    "TicketAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.ticket_api",
    "TradingBot": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.trading_bot",
    "TravelAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.travel_booking",
    "VehicleControlAPI": "lm_eval.tasks.bfcl_v3.multi_turn_func_source.vehicle_control",
}

STATELESS_CLASSES = {"MathAPI"}
BLOCKED_METHODS = {
    "kill",
    "exit",
    "quit",
    "remove",
    "unlink",
    "popen",
    "Popen",
    "run",
}


def instantiate_classes(
    initial_config: dict, involved_classes: list[str], long_context: bool = False
) -> dict[str, Any]:
    instances = {}
    for class_name in involved_classes:
        module = importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name])
        class_ = getattr(module, class_name)
        instance = class_()
        if class_name not in STATELESS_CLASSES:
            class_initial_config = copy.deepcopy(initial_config.get(class_name, {}))
            instance._load_scenario(class_initial_config, long_context=long_context)
        instances[class_name] = instance
    return instances


def execute_multi_turn_func_call(
    func_call_list: list[str],
    initial_config: dict | None = None,
    involved_classes: list[str] | None = None,
    model_name: str | None = None,
    test_entry_id: str | None = None,
    long_context: bool = False,
    is_evaL_run: bool = False,
    instances: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    if instances is None:
        instances = instantiate_classes(
            initial_config or {}, involved_classes or [], long_context=long_context
        )

    method_mapping = _method_mapping(instances)
    execution_results = []
    for func_call in func_call_list:
        try:
            result = _execute_call_string(func_call, method_mapping)
            if isinstance(result, str):
                execution_results.append(result)
            elif isinstance(result, dict):
                try:
                    execution_results.append(json.dumps(result))
                except TypeError:
                    execution_results.append(str(result))
            else:
                execution_results.append(str(result))
        except Exception as exc:  # noqa: BLE001 - BFCL records tool failures as observations.
            execution_results.append(f"Error during execution: {exc}")
    return execution_results, instances


def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    return len(input_list) == 1 and len(input_list[0]) == 0


def extract_function_call_strings(text: str) -> list[str]:
    text = _strip_code_fence(text)
    if not text:
        return []
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text += "]"
    parsed = ast.parse(text, mode="eval")
    calls = parsed.body.elts if isinstance(parsed.body, ast.List) else [parsed.body]
    return [ast.unparse(call) for call in calls if isinstance(call, ast.Call)]


def _method_mapping(instances: dict[str, Any]) -> dict[str, Any]:
    mapping = {}
    for class_name, instance in instances.items():
        for method_name, method in inspect.getmembers(
            instance, predicate=inspect.ismethod
        ):
            if method_name.startswith("_"):
                continue
            mapping[method_name] = method
            mapping[f"{class_name}.{method_name}"] = method
    return mapping


def _execute_call_string(func_call: str, method_mapping: dict[str, Any]) -> Any:
    node = ast.parse(func_call, mode="eval").body
    if not isinstance(node, ast.Call):
        raise TypeError(f"Expected a function call, got {type(node).__name__}")
    return _execute_call(node, method_mapping)


def _execute_call(node: ast.Call, method_mapping: dict[str, Any]) -> Any:
    func_name = _function_name(node.func)
    short_name = func_name.rsplit(".", 1)[-1]
    if short_name in BLOCKED_METHODS:
        raise ValueError(f"Function call {short_name} is not allowed.")
    if func_name not in method_mapping and short_name not in method_mapping:
        raise ValueError(f"Unknown function call {func_name}.")

    method = method_mapping.get(func_name, method_mapping[short_name])
    args = [_literal_or_call(arg, method_mapping) for arg in node.args]
    kwargs = {
        kw.arg: _literal_or_call(kw.value, method_mapping)
        for kw in node.keywords
        if kw.arg is not None
    }
    return method(*args, **kwargs)


def _literal_or_call(node: ast.AST, method_mapping: dict[str, Any]) -> Any:
    if isinstance(node, ast.Call):
        return _execute_call(node, method_mapping)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_or_call(item, method_mapping) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_or_call(item, method_mapping) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_or_call(key, method_mapping): _literal_or_call(
                value, method_mapping
            )
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = _literal_or_call(node.operand, method_mapping)
        return -operand if isinstance(operand, (int, float)) else ast.unparse(node)
    return ast.literal_eval(node)


def _function_name(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    if not parts:
        raise ValueError(f"Unsupported function expression: {ast.unparse(node)}")
    return ".".join(reversed(parts))


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip("`\n ")
