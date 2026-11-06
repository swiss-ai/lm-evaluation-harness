"""Opt-in repeated-attempt scoring using each task's extraction and grader."""

import logging
from copy import copy
from dataclasses import replace
from math import comb, isfinite, isnan
from numbers import Real
from statistics import fmean as mean


eval_logger = logging.getLogger(__name__)

GENERATIVE = {"generate_until", "multi_turn_generate"}


def configure_repeats(
    tasks, repeats=None, repeat_scoring=None, task_overrides=None, groups=()
):
    overrides = {} if task_overrides is None else task_overrides
    if not isinstance(overrides, dict) or set(overrides) - tasks.keys():
        raise ValueError("task_overrides must name selected leaf tasks")
    if repeat_scoring is not None and not isinstance(repeat_scoring, dict):
        raise ValueError("repeat_scoring must be an object")
    run = repeat_scoring or {}
    run_ks = run.get("pass_at_k") if isinstance(run.get("pass_at_k"), dict) else {}
    if run.get("enabled") and run.get("mean") is False and not run_ks:
        raise ValueError("repeat_scoring requests no metrics")
    # The run policy covers tasks that can repeat-score what it requests, except
    # those of a group that aggregates over one that cannot.
    eligible = {
        name
        for name, task in tasks.items()
        if repeat_scoring is not None
        and task.OUTPUT_TYPE in GENERATIVE
        and task.get_config("repeat_support")
        and not _multi_sample(task)
        and (run.get("mean", True) or _binary(task) & run_ks.keys())
    }
    changed = True
    while changed:  # overlapping groups can exclude each other's tasks
        changed = False
        for group in groups:
            leaves = {task.task_name for task in group.get_all_tasks()}
            if (
                group.aggregate_metric_list
                and leaves & eligible
                and not leaves <= eligible
            ):
                eligible -= leaves
                changed = True
    unmatched = run_ks.keys() - set().union(*(_binary(tasks[n]) for n in eligible))
    if run.get("enabled") and unmatched:
        eval_logger.warning(
            f"No repeat-scored task declares binary {sorted(unmatched)}: no pass@k"
        )
    for name, task in tasks.items():
        override = overrides.get(name, {})
        if not isinstance(override, dict) or set(override) - {
            "repeats",
            "repeat_scoring",
        }:
            raise ValueError(
                f"{name}: only repeats and repeat_scoring can be overridden"
            )
        policy = {"enabled": False, "mean": True, "pass_at_k": {}}
        run_layer = repeat_scoring if name in eligible else None
        if run_layer is not None and run_ks:
            run_layer = run_layer | {
                "pass_at_k": {m: ks for m, ks in run_ks.items() if m in _binary(task)}
            }
        support = task.get_config("repeat_support") or {}
        for layer in (
            task.get_config("repeat_scoring"),
            run_layer,
            override.get("repeat_scoring"),
        ):
            if layer is not None:
                if not isinstance(layer, dict) or set(layer) - policy.keys():
                    raise ValueError(f"{name}: invalid repeat_scoring policy")
                policy.update(layer)
        if any(type(policy[key]) is not bool for key in ("enabled", "mean")):
            raise ValueError(f"{name}: enabled and mean must be booleans")
        if not policy["enabled"] and set(override.get("repeat_scoring") or {}) - {
            "enabled"
        }:
            eval_logger.warning(
                f"{name}: repeat scoring is off, so its repeat_scoring override is ignored"
            )
        count = override.get("repeats")
        if count is None:
            # A run-level count repeats generative tasks: with run-level repeat
            # scoring only those that use it, otherwise all (legacy repeats).
            applies = task.OUTPUT_TYPE in GENERATIVE and (
                policy["enabled"] or not run.get("enabled", False)
            )
            count = (
                repeats
                if repeats is not None and applies
                else task.get_config("repeats")
            )
        if type(count) is not int or count < 1:
            raise ValueError(f"{name}: repeats must be a positive integer")
        if not isinstance(policy["pass_at_k"], dict):
            raise TypeError(f"{name}: pass_at_k must map metric names to k lists")
        if policy["enabled"]:
            if task.OUTPUT_TYPE not in GENERATIVE or not support:
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
            if _multi_sample(task):
                raise ValueError(
                    f"{name}: repeat_filter must select a single-attempt filter"
                )
        task.set_config("repeats", count)
        task.set_config("repeat_scoring", policy)


def _multi_sample(task):
    return any(
        factory.keywords.get("k", 1) > 1
        for f in repeat_filters(task)
        for factory in f.filters
    )


def _binary(task):
    support = task.get_config("repeat_support")
    if not isinstance(support, dict):
        return set()
    return {metric for metric, kind in support.items() if kind == "binary"}


def check_scope(tasks, groups, lm):
    """Fail before any request when repeat scoring cannot be honoured.

    Only API models send each attempt as its own request, with an offset seed
    where the provider takes one. A group's aggregate needs every leaf scored.
    """
    scored = sorted(name for name, task in tasks.items() if enabled(task))
    if scored and not lm.supports_repeat_scoring:
        raise ValueError(
            "Repeat scoring supports API models only (local-completions, "
            "local-chat-completions, apertus2-completions and other TemplateAPI "
            f"models); requested for {', '.join(scored[:5])}"
        )
    for group in groups.values():
        if not group.aggregate_metric_list or not any(
            enabled(task) for task in group.get_all_tasks()
        ):
            continue
        reported = {
            config.metric.removesuffix("_mean_repeats").partition("_pass@")[0]
            for config in scoring_group(group).aggregate_metric_list
        }
        for config in group.aggregate_metric_list:
            if config.metric not in reported:
                eval_logger.warning(
                    f"{group.name}: its tasks share no repeat-scored "
                    f"{config.metric} on a common filter; the group omits it"
                )


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
        if len(inst.resps) != count or None in inst.resps:
            raise RuntimeError(
                f"{task.task_name}: missing attempt for document {inst.doc_id}"
            )
        inst.filtered_resps = {}
    filters = repeat_filters(task)
    for index in range(count):
        attempts = []
        for inst in task.instances:
            attempt = copy(inst)
            attempt.resps = [inst.resps[index]]
            attempt.filtered_resps = {}
            attempts.append(attempt)
        for ensemble in filters:
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
    """Aggregate repeat metrics over the leaves that report them, as ordinary groups do."""
    tasks = group.get_all_tasks()
    scored = [task for task in tasks if enabled(task)]
    if not scored:
        return group
    if group.aggregate_metric_list and len(scored) != len(tasks):
        missing = sorted(task.task_name for task in tasks if not enabled(task))
        raise ValueError(
            f"{group.name}: repeat scoring is enabled for some of its tasks only "
            f"(not {', '.join(missing)}); enable it for all of them or none"
        )
    aggregations = []
    for config in group.aggregate_metric_list or []:
        contributors = [
            t for t in scored if config.metric in t.get_config("repeat_support")
        ]
        if not contributors:
            continue
        policies = [task.get_config("repeat_scoring") for task in contributors]
        metrics = []
        if all(policy["mean"] for policy in policies):
            metrics.append(f"{config.metric}_mean_repeats")
        ks = set.intersection(
            *(set(policy["pass_at_k"].get(config.metric, [])) for policy in policies)
        )
        metrics.extend(f"{config.metric}_pass@{k}" for k in sorted(ks))
        # As in ordinary groups, a filter covers the tasks that define it; each of
        # them must score repeats with it.
        selected = {
            task.task_name: {f.name for f in repeat_filters(task)}
            for task in contributors
        }
        common_filters = {
            name
            for name in set.union(*selected.values())
            if (config.filter_list is None or name in config.filter_list)
            and all(
                name in selected[task.task_name]
                for task in contributors
                if name in {f.name for f in task._filters}
            )
        }
        if common_filters:
            aggregations.extend(
                replace(config, metric=metric, filter_list=sorted(common_filters))
                for metric in metrics
            )
    result = copy(group)
    result.aggregate_metric_list = aggregations
    return result
