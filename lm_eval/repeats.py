"""Opt-in repeated-attempt scoring using each task's extraction and grader."""

from copy import copy
from dataclasses import replace
from math import comb, isfinite, isnan
from numbers import Real
from statistics import fmean as mean


def configure_repeats(tasks, repeats=None, repeat_scoring=None, task_overrides=None):
    overrides = {} if task_overrides is None else task_overrides
    if not isinstance(overrides, dict) or set(overrides) - tasks.keys():
        raise ValueError("task_overrides must name selected leaf tasks")
    for name, task in tasks.items():
        override = overrides.get(name, {})
        if not isinstance(override, dict) or set(override) - {
            "repeats",
            "repeat_scoring",
        }:
            raise ValueError(
                f"{name}: only repeats and repeat_scoring can be overridden"
            )
        count = override.get("repeats", repeats)
        if count is None:
            count = task.get_config("repeats")
        if type(count) is not int or count < 1:
            raise ValueError(f"{name}: repeats must be a positive integer")
        policy = {"enabled": False, "mean": True, "pass_at_k": {}}
        for layer in (
            task.get_config("repeat_scoring"),
            repeat_scoring,
            override.get("repeat_scoring"),
        ):
            if layer is not None:
                if not isinstance(layer, dict) or set(layer) - policy.keys():
                    raise ValueError(f"{name}: invalid repeat_scoring policy")
                policy.update(layer)
        if any(type(policy[key]) is not bool for key in ("enabled", "mean")):
            raise ValueError(f"{name}: enabled and mean must be booleans")
        if not isinstance(policy["pass_at_k"], dict):
            raise TypeError(f"{name}: pass_at_k must map metric names to k lists")
        support = task.get_config("repeat_support") or {}
        if policy["enabled"]:
            if (
                task.OUTPUT_TYPE not in {"generate_until", "multi_turn_generate"}
                or not support
            ):
                raise ValueError(f"{name}: repeat scoring is not supported")
            if not isinstance(support, dict) or any(
                kind not in {"binary", "mean", "native"} for kind in support.values()
            ):
                raise ValueError(f"{name}: invalid repeat_support declaration")
            missing = {
                metric for metric, kind in support.items() if kind == "native"
            } - task.aggregation().keys()
            if missing:
                raise ValueError(
                    f"{name}: repeat metrics missing aggregation: {sorted(missing)}"
                )
            if not policy["mean"] and not policy["pass_at_k"]:
                raise ValueError(f"{name}: repeat scoring requests no metrics")
            for metric, ks in policy["pass_at_k"].items():
                if support.get(metric) != "binary":
                    raise ValueError(
                        f"{name}: pass@k requires a declared binary metric: {metric}"
                    )
                if (
                    not isinstance(ks, list)
                    or not ks
                    or any(type(k) is not int or not 1 <= k <= count for k in ks)
                ):
                    raise ValueError(
                        f"{name}: pass@k needs integer k values between 1 and repeats"
                    )
            filters = repeat_filters(task)
            # Multi-sample filters need an explicitly selected single-attempt path.
            if any(
                getattr(factory(), "k", 1) > 1 for f in filters for factory in f.filters
            ):
                raise ValueError(
                    f"{name}: repeat_filter must select a single-attempt filter"
                )
        task.set_config("repeats", count)
        task.set_config("repeat_scoring", policy)


def enabled(task):
    return (getattr(getattr(task, "config", None), "repeat_scoring", None) or {}).get(
        "enabled", False
    )


def repeat_filters(task):
    selected = task.get_config("repeat_filter")
    filters = [f for f in task._filters if selected is None or f.name == selected]
    if not filters:
        raise ValueError(f"{task.task_name}: unknown repeat_filter {selected!r}")
    return filters


def apply_repeat_filters(task):
    """Keep each attempt separate while reusing the ordinary filter pipeline."""
    count = task.get_config("repeats")
    for inst in task.instances:
        if len(inst.resps) != count:
            raise ValueError(
                f"{task.task_name}: incomplete attempts for document {inst.doc_id}"
            )
        inst.filtered_resps = {}
    for index in range(count):
        attempts = []
        for inst in task.instances:
            attempt = copy(inst)
            attempt.resps = [inst.resps[index]]
            attempt.filtered_resps = {}
            attempts.append(attempt)
        for ensemble in repeat_filters(task):
            ensemble.apply(attempts)
            for inst, attempt in zip(task.instances, attempts, strict=True):
                inst.filtered_resps.setdefault(ensemble.name, []).append(
                    attempt.filtered_resps[ensemble.name]
                )


def score_attempts(task, doc, requests, filter_key):
    scores = [
        task.process_results(
            doc, [req.filtered_resps[filter_key][i] for req in requests]
        )
        for i in range(task.get_config("repeats"))
    ]
    support = task.get_config("repeat_support")
    for score in scores:
        for metric, kind in support.items():
            if metric not in score:
                raise ValueError(f"{task.task_name}: grader did not return {metric}")
            value = score[metric]
            if kind != "native" and (
                not isinstance(value, Real) or not isfinite(value)
            ):
                raise ValueError(
                    f"{task.task_name}: {metric} must return a finite number"
                )
            if kind == "binary" and value not in (0, 1):
                raise ValueError(
                    f"{task.task_name}: {metric} must return binary correctness"
                )
    return {metric: [score[metric] for score in scores] for metric in support}


def aggregate_metric(task, metric, items):
    """Aggregate across documents within each repeat before reducing repeats."""
    policy = task.get_config("repeat_scoring")
    kind = task.get_config("repeat_support")[metric]
    agg = task.aggregation()[metric] if kind == "native" else mean
    repeat_scores = [agg(list(values)) for values in zip(*items, strict=True)]
    if any(
        not isinstance(value, Real)
        or (not isfinite(value) and not (kind == "native" and isnan(value)))
        for value in repeat_scores
    ):
        raise ValueError(
            f"{task.task_name}: {metric} aggregation must return a scalar (NaN only for unavailable native scores)"
        )
    result = {}
    if policy["mean"]:
        result[f"{metric}_mean_repeats"] = mean(repeat_scores)
    count = task.get_config("repeats")
    for k in policy["pass_at_k"].get(metric, []):
        result[f"{metric}_pass@{k}"] = mean(
            1 - comb(count - int(sum(values)), k) / comb(count, k) for values in items
        )
    return result, repeat_scores


def scoring_group(group):
    """Only combine a repeat metric when every leaf supplies that protocol."""
    tasks = group.get_all_tasks()
    if not any(enabled(task) for task in tasks):
        return group
    aggregations = []
    for config in group.aggregate_metric_list or []:
        if not all(
            enabled(task) and config.metric in task.get_config("repeat_support")
            for task in tasks
        ):
            continue
        policies = [task.get_config("repeat_scoring") for task in tasks]
        metrics = []
        if all(policy["mean"] for policy in policies):
            metrics.append(f"{config.metric}_mean_repeats")
        ks = set.intersection(
            *(set(policy["pass_at_k"].get(config.metric, [])) for policy in policies)
        )
        metrics.extend(f"{config.metric}_pass@{k}" for k in sorted(ks))
        selected = {
            task.task_name: {f.name for f in repeat_filters(task)} for task in tasks
        }
        common_filters = set.union(*selected.values())
        if config.filter_list is not None:
            common_filters.intersection_update(config.filter_list)
        common_filters = {
            name
            for name in common_filters
            if all(
                name in selected[task.task_name]
                for task in tasks
                if name in {f.name for f in task._filters}
            )
        }
        aggregations.extend(
            replace(config, metric=metric, filter_list=sorted(common_filters))
            for metric in metrics
        )
    result = copy(group)
    result.aggregate_metric_list = aggregations
    return result
