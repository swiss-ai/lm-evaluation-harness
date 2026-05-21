from __future__ import annotations

from lm_eval.tasks.bfcl_v3.multi_turn_utils import (
    execute_multi_turn_func_call,
    instantiate_classes,
    is_empty_execute_response,
)


def multi_turn_checker(
    multi_turn_model_result_list_decoded: list[list[list[str]]],
    multi_turn_ground_truth_list: list[list[str]],
    test_entry: dict,
    test_category: str,
    model_name: str,
) -> dict:
    initial_config = test_entry["initial_config"]
    involved_classes = test_entry["involved_classes"]
    test_entry_id = test_entry["id"]
    test_category = test_entry_id.rsplit("_", 1)[0]
    execution_results = []
    all_turn_model_execution_results = []
    long_context = "long_context" in test_category or "composite" in test_category
    model_instances = instantiate_classes(
        initial_config, involved_classes, long_context=long_context
    )
    ground_truth_instances = instantiate_classes(
        initial_config, involved_classes, long_context=long_context
    )

    for turn_index, single_turn_ground_truth_list in enumerate(
        multi_turn_ground_truth_list
    ):
        if turn_index < len(multi_turn_model_result_list_decoded):
            single_turn_model_response_list = multi_turn_model_result_list_decoded[
                turn_index
            ]
        else:
            single_turn_model_response_list = []

        single_turn_model_execution_results = []
        single_turn_model_execution_results_uncombined = []
        for single_step_model_response in single_turn_model_response_list:
            single_step_model_execution_results, _ = (
                execute_multi_turn_func_call(
                    func_call_list=single_step_model_response,
                    instances=model_instances,
                    long_context=long_context,
                )
            )
            single_turn_model_execution_results.extend(
                single_step_model_execution_results
            )
            single_turn_model_execution_results_uncombined.append(
                single_step_model_execution_results
            )

        single_turn_ground_truth_execution_results, _ = (
            execute_multi_turn_func_call(
                func_call_list=single_turn_ground_truth_list,
                instances=ground_truth_instances,
                long_context=long_context,
            )
        )

        all_turn_model_execution_results.extend(single_turn_model_execution_results)
        execution_results.append(
            {
                "model": single_turn_model_execution_results_uncombined,
                "ground_truth": single_turn_ground_truth_execution_results,
            }
        )

        if len(single_turn_ground_truth_list) > 0:
            if not single_turn_model_response_list or is_empty_execute_response(
                single_turn_model_response_list
            ):
                return {
                    "valid": False,
                    "error_message": f"Model response list is empty for turn {turn_index}",
                    "error_type": "multi_turn:empty_turn_model_response",
                    "details": {"execution_result": execution_results},
                }

        if not single_turn_ground_truth_list:
            continue

        if len(model_instances) != len(ground_truth_instances) or set(
            model_instances
        ) != set(ground_truth_instances):
            return {
                "valid": False,
                "error_message": "Model and ground-truth instances do not match.",
                "error_type": "multi_turn:instance_set_mismatch",
                "details": {
                    "model_instances": sorted(model_instances),
                    "ground_truth_instances": sorted(ground_truth_instances),
                },
            }

        state_check_result = state_checker(model_instances, ground_truth_instances)
        if not state_check_result["valid"]:
            state_check_result["execution_result"] = execution_results
            return state_check_result

        response_check_result = response_checker(
            all_turn_model_execution_results,
            single_turn_ground_truth_execution_results,
            turn_index,
        )
        if not response_check_result["valid"]:
            return response_check_result

    return {"valid": True}


def state_checker(model_instances: dict, ground_truth_instances: dict):
    for class_name, ground_truth_instance in ground_truth_instances.items():
        model_instance = model_instances[class_name]
        valid, differences = _compare_instances(model_instance, ground_truth_instance)
        if not valid:
            return {
                "valid": False,
                "error_message": (
                    f"Model instance for {class_name} does not match the "
                    "ground-truth state."
                ),
                "error_type": "multi_turn:instance_state_mismatch",
                "details": {
                    "differences": differences,
                    "model_instance_state": _public_attrs(model_instance),
                    "ground_truth_instance_state": _public_attrs(ground_truth_instance),
                },
            }
    return {"valid": True}


def response_checker(model_response_list: list, ground_truth_response_list: list, turn_index: int):
    is_subsequence, missing_items = _is_subsequence_unordered(
        ground_truth_response_list, model_response_list
    )
    if not is_subsequence:
        return {
            "valid": False,
            "error_message": (
                "Model response execution results so far do not contain all "
                f"ground-truth execution results for turn {turn_index}."
            ),
            "error_type": "multi_turn:execution_response_mismatch",
            "details": {
                "missing_items": missing_items,
                "model_response (including all previous turns)": model_response_list,
                "ground_truth_response (only the current turn)": ground_truth_response_list,
            },
        }
    return {"valid": True}


def _compare_instances(instance_a, instance_b):
    attrs_a = _public_attrs(instance_a)
    attrs_b = _public_attrs(instance_b)
    differences = {}
    for key in set(attrs_a) | set(attrs_b):
        if attrs_a.get(key) != attrs_b.get(key):
            differences[key] = {
                "model": attrs_a.get(key),
                "ground_truth": attrs_b.get(key),
            }
    return not differences, differences


def _public_attrs(instance):
    return {
        key: value for key, value in vars(instance).items() if not key.startswith("_")
    }


def _is_subsequence_unordered(ground_truth: list, model: list):
    unmatched = list(model)
    missing = []
    for item in ground_truth:
        if item in unmatched:
            unmatched.remove(item)
        else:
            missing.append(item)
    return len(missing) == 0, missing
