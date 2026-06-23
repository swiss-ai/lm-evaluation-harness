from __future__ import annotations

import itertools
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

import lm_eval.api.model
import lm_eval.api.registry
from lm_eval.caching.cache import delete_cache
from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED
from lm_eval.evaluator_utils import (
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    handle_non_serializable,
    hash_dict_images,
    hash_string,
    positional_deprecated,
    set_torch_seed,
    setup_logging,
    simple_parse_args_string,
    wrap_text,
)


if TYPE_CHECKING:
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task
    from lm_eval.loggers import EvaluationTracker

eval_logger = logging.getLogger(__name__)


def _warn_if_system_prompt_authority_inactive(lm, task_dict) -> list[str]:
    """Warn when a task needs an authoritative system prompt but the model's
    system-prompt authority check was not activated (it is OFF by default).

    A task opts in with ``metadata: {requires_system_prompt_authority: true}``.
    HF/vLLM set ``lm.system_prompt_authority_handled`` to ``True`` when any of
    ``check_system_prompt_authority`` / ``strip_system_boilerplate`` /
    ``allow_system_boilerplate`` is passed; backends without the concept lack
    the attribute (treated as handled → no warning). Returns the offending task
    names (``[]`` if none), so the behaviour is unit-testable.
    """
    if getattr(lm, "system_prompt_authority_handled", True):
        return []
    needy = []
    for to in get_task_list(task_dict):
        # Group placeholders can carry a falsy/None task; read defensively.
        cfg = getattr(getattr(to, "task", None), "config", None)
        metadata = getattr(cfg, "metadata", None)
        if isinstance(metadata, dict) and metadata.get(
            "requires_system_prompt_authority"
        ):
            needy.append(to.task_name)
    if needy:
        eval_logger.warning(
            "Task(s) %s expect an authoritative system prompt, but the "
            "system-prompt authority check is not active for this model (OFF by "
            "default). Chat-template boilerplate (e.g. Llama 3.x date headers) "
            "can silently skew these scores. Pass `check_system_prompt_authority=true` to "
            "verify, `strip_system_boilerplate=true` to auto-fix Llama 3.x "
            "boilerplate, or `allow_system_boilerplate=true` to acknowledge and "
            "silence this warning.",
            needy,
        )
    return needy


@positional_deprecated
def simple_evaluate(
    model: str | LM,
    model_args: str | dict[str, str | int | float] | None = None,
    tasks: list[str | dict | Task] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str | None = None,
    max_batch_size: int | None = None,
    device: str | None = None,
    use_cache: str | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: int | float | None = None,
    samples: dict | None = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: EvaluationTracker | None = None,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = True,
    gen_kwargs: str | dict | None = None,
    task_manager: TaskManager | None = None,
    verbosity=None,
    predict_only: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    numpy_random_seed: int = DEFAULT_OTHER_SEED,
    torch_random_seed: int = DEFAULT_OTHER_SEED,
    fewshot_random_seed: int = DEFAULT_OTHER_SEED,
    confirm_run_unsafe_code: bool = False,
    metadata: dict | None = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        model (str | LM): Name of model or LM object. See
            lm_eval.models.__init__.py for available aliases.
        model_args (str | dict | None): String or dict arguments for each model
            class, see LM.create_from_arg_string and LM.create_from_arg_object.
            Ignored if `model` argument is a LM object.
        tasks (list[str | dict | Task]): List of task names or Task objects.
            Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined
            and type(task).__name__ otherwise.
        num_fewshot (int): Number of examples in few-shot context.
        batch_size (int | str | None): Batch size for model.
        max_batch_size (int | None): Maximal batch size to try with automatic
            batch size detection.
        device (str | None): PyTorch device (e.g. "cpu" or "cuda:0") for running
            models.
        use_cache (str | None): A path to a sqlite db file for caching model
            responses. `None` if not caching.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests. `None` if not caching.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`. `None` if not desired.
        delete_requests_cache (bool): Deletes all the request cache if set to
            `True`. `None` if not desired.
        limit (int | float | None): Limit the number of examples per task (only
            use this for testing). If <1, limit is a percentage of the total
            number of examples.
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
        bootstrap_iters (int): Number of iterations for bootstrap statistics, used
            when calculating stderrs. Set to 0 for no stderr calculations to be
            performed.
        check_integrity (bool): Whether to run the relevant part of the test suite
            for the tasks.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents for
            per-sample measurement and post-hoc analysis.
        evaluation_tracker (EvaluationTracker | None): Tracker for logging
            experiment configuration and results.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        gen_kwargs (dict | str | None): Arguments for model generation. Ignored
            for all tasks with loglikelihood output_type.
        task_manager (TaskManager | None): Task manager instance to use.
        verbosity (str | None): Verbosity level for logging.
        predict_only (bool): If True, only model outputs will be generated and
            returned. Metrics will not be evaluated.
        random_seed (int): Random seed for python's random module. If set to None,
            the seed will not be set.
        numpy_random_seed (int): Random seed for numpy. If set to None, the seed
            will not be set.
        torch_random_seed (int): Random seed for torch. If set to None, the seed
            will not be set.
        fewshot_random_seed (int): Random seed for fewshot sampler random generator.
            If set to None, the seed of generator will be set to None.
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe.
        metadata (dict | None): Additional metadata to be added to the task
            manager. Will get passed to the download function of the task.

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """
    if verbosity is not None:
        setup_logging(verbosity=verbosity)
    start_date = time.time()

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )

    _NEEDS_CHAT_TEMPLATE = ("inst", "chat")
    if (
        (
            isinstance(model_args, str)
            and any(kw in model_args.lower() for kw in _NEEDS_CHAT_TEMPLATE)
        )
        or (
            isinstance(model_args, dict)
            and any(
                any(kw in str(v).lower() for kw in _NEEDS_CHAT_TEMPLATE)
                for v in model_args.values()
            )
        )
    ) and not apply_chat_template:
        eval_logger.warning(
            wrap_text(
                f"""pretrained={model_args.get("pretrained") if isinstance(model_args, dict) else model_args} appears to be an
                instruct or chat variant but chat template is not applied.
                Recommend setting `apply_chat_template` (optionally `fewshot_as_multiturn`).""",
            )
        )

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        set_torch_seed(torch_random_seed)

    if fewshot_random_seed is not None:
        seed_message.append(f"Setting fewshot manual seed to {fewshot_random_seed}")

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs:
        if isinstance(gen_kwargs, str):
            gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            f"generation_kwargs: {gen_kwargs} specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if not gen_kwargs:
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""

        if isinstance(model_args, dict):
            eval_logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            eval_logger.info(
                wrap_text(
                    f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
                )
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type {type(model)}, but is required to be a subclass of lm_eval.api.model.LM . This may be because you are passing an initialized Hugging Face PreTrainedModel without having wrapped it in `lm_eval.models.huggingface.HFLM(pretrained=my_model)` first."
            )
        eval_logger.info("Using pre-initialized model")
        lm = model

    if use_cache is not None:
        eval_logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        metadata = (
            simple_parse_args_string(model_args)
            if isinstance(model_args, str)
            else model_args
            if isinstance(model_args, dict)
            else {}
        ) | (metadata or {})
        task_manager = TaskManager(metadata=metadata)

    task_dict = get_task_dict(
        tasks,
        task_manager,
    )

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                if task_obj.get_config("output_type") in (
                    "generate_until",
                    "multi_turn_generate",
                ):
                    # multi_turn_generate also issues generate_until calls per
                    # turn (see run_multi_turn_rollout), reading the task's
                    # generation_kwargs — so the CLI override must reach it too,
                    # e.g. `--gen_kwargs until=<|im_end|>` to add a stop token.
                    if gen_kwargs is not None:
                        task_obj.set_config(
                            key="generation_kwargs", value=gen_kwargs, update=True
                        )
                    eval_logger.info(
                        f"{task_obj.config.task}: Using gen_kwargs: {task_obj.config.generation_kwargs}"
                    )

                if predict_only:
                    eval_logger.info(
                        f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                    )
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(
                            f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                        )
                    else:
                        eval_logger.warning(
                            f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                        )
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (
                        default_num_fewshot := task_obj.get_config("num_fewshot")
                    ) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    # Rank 0 only — `simple_evaluate` runs on every rank under accelerate/DDP,
    # so guard to avoid one identical warning per process.
    if getattr(lm, "rank", 0) == 0:
        _warn_if_system_prompt_authority_inactive(lm, task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model if isinstance(model, str) else "CUSTOM",
            model_args=model_args or "",
            system_instruction=system_instruction,
            chat_template=lm.chat_template(apply_chat_template)
            if apply_chat_template
            else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        samples=samples,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
    )
    if verbosity is not None:
        setup_logging(verbosity=verbosity)

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
        }
        # add more detailed model info if available
        if hasattr(lm, "get_model_info"):
            results["config"].update(lm.get_model_info())  # type: ignore
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []  # type: ignore
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    lm: LM,
    task_dict,
    limit: int | None = None,
    samples: dict | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: int | None = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    confirm_run_unsafe_code: bool = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        lm (LM): Language Model.
        task_dict (dict[str, Task]): Dictionary of tasks. Tasks will be taken to
            have name type(task).config.task.
        limit (int | None): Limit the number of examples per task (only use this
            for testing).
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`.
        bootstrap_iters (int | None): Number of iterations for bootstrap
            statistics, used when calculating stderr. Set to 0 for skipping all
            stderr calculations.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents
            for per-sample measurement and post-hoc analysis.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        verbosity (str): Verbosity level for logging. (no-op, deprecated)
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe.

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )
    if samples is not None:
        eval_logger.info(f"Evaluating examples for tasks {list(samples.keys())}")
    if apply_chat_template:
        eval_logger.warning(
            "Chat template formatting change affects loglikelihood and multiple-choice tasks. See docs/chat-template-readme.md for details."
        )
    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    if not log_samples and not all(
        "bypass" not in getattr(task_output.task, "_metric_fn_list", {})
        for task_output in eval_tasks
    ):
        raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    # validation checks:
    # 1.are we running multimodal task <-> non-multimodal model class, or vice-versa.
    # 2.are we running code that is marked as unsafe.
    incompatible_tasks = []
    for task_output in eval_tasks:
        task: Task = task_output.task

        if getattr(task, "MULTIMODAL", False) and not getattr(lm, "MULTIMODAL", False):
            incompatible_tasks.append(task_output.task_name)
        elif getattr(task, "UNSAFE_CODE", False) and not confirm_run_unsafe_code:
            raise ValueError(
                f"Attempted to run task: {task_output.task_name} which is marked as unsafe. Set confirm_run_unsafe_code=True to run this task."
            )
    if len(incompatible_tasks) > 0 and not getattr(lm, "MULTIMODAL", False):
        raise ValueError(
            f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, but the selected model type does not currently implement this. Multimodal support is currently restricted to the ['hf-multimodal', 'vllm-vlm'] model type."
        )
    # end validation check

    # Cache the limit arg.
    limit_arg = limit
    limits = []
    for task_output in eval_tasks:
        task: Task = task_output.task

        limit = get_sample_size(task, limit_arg)
        limits.append(limit)
        task.build_all_requests(
            limit=limit,
            samples=samples.get(task_output.task_name, None)
            if samples is not None
            else samples,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template", None)
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            import torch

            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

    ### Iterative multi-turn rollout (output_type: multi_turn_generate) ###
    # Runs BEFORE the standard one-shot dispatch so the turn-indexed
    # `Instance` objects it builds end up on each task's `_instances` list
    # and flow through the existing filter + process_results + sample-log
    # path unchanged. Tasks without multi_turn_generate are unaffected.
    _mt_task_outputs = [
        to for to in eval_tasks if to.task.OUTPUT_TYPE == "multi_turn_generate"
    ]
    if _mt_task_outputs:
        run_multi_turn_rollout(lm, _mt_task_outputs)

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs, strict=True):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output, limit in zip(eval_tasks, limits, strict=True):
        task = task_output.task

        # Skip post-processing for ranks with no instances (data parallel with
        # small datasets, or multi_turn_generate when world_size > docs-for-task,
        # leaving some ranks an empty shard). Must come BEFORE apply_filters():
        # Filter.apply does `zip(*(... for inst in instances))`, which raises
        # `ValueError: not enough values to unpack` on an empty instance list.
        # The gather will collect empty results from this rank.
        if len(task.instances) == 0:
            continue

        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps:
            indices = (
                samples.get(task_output.task_name, None)
                if samples is not None
                else None
            )
            doc_iterator = task.doc_iterator(
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
                samples=indices,
            )
            for doc_id, doc in doc_iterator:
                doc_id_true = indices[doc_id] if indices else doc_id
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id_true,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        # Per-response generation length measured before any
                        # thinking-strip (populated by generation models that support
                        # it, e.g. vLLM; empty list otherwise).
                        "length_info": [req.length_info for req in requests],
                        "filter": filter_key,
                        "metrics": list(metrics.keys()),
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        import torch

        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.logged_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks. Agree on a consistent key
            # set first: a rank with an empty shard (world_size > docs-for-task,
            # reachable for multi_turn_generate on small/limited tasks, or any DP
            # run with fewer docs than ranks) has no `sample_metrics` keys and
            # would otherwise issue fewer `gather_object` collectives than its
            # peers — desyncing the gather and deadlocking the job.
            local_keys = list(task_output.sample_metrics.keys())
            gathered_keys = [None] * WORLD_SIZE
            torch.distributed.all_gather_object(gathered_keys, local_keys)
            all_keys = sorted({k for keys in gathered_keys for k in keys})
            for metrics in all_keys:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.sample_metrics.get(metrics, []),
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(
                results, versions, task_dict
            )

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if (
                len(task_list) != 0
            ):  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better:
                            _higher_is_better[m] = h

                        if (
                            m in _higher_is_better
                            and _higher_is_better[m] is not None
                            and _higher_is_better[m] != h
                        ):
                            eval_logger.warning(
                                f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None."
                            )
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **(
                {"groups": dict(group_agg.items())}
                if (bool(group_agg) & show_group_table)
                else {}
            ),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output, limit in zip(eval_tasks, limits, strict=True)
            },
        }
        if log_samples:
            # default: hash images
            samples = (
                hash_dict_images(samples)
                if os.environ.get("LMEVAL_HASHMM", "1") != "0"
                and (hasattr(lm, "MULTIMODAL"))
                else samples
            )
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None


# Defensive cap on multi-turn rollout depth against a task whose
# ``next_user_turn`` never returns ``None``. No real benchmark exceeds ~10
# turns; 64 is comfortable headroom. Expose via task config if ever needed.
MAX_SAFETY_TURNS = 64


def _gather_ints(lm, value: int) -> list[int]:
    """Collective ``accelerator.gather`` of one per-rank integer, returned as
    a plain list (length == world_size). All ranks must call together. The
    caller applies the reduction (``any`` / ``max``).
    """
    import torch

    t = torch.tensor(int(value), device=lm.device)
    return lm.accelerator.gather(t).cpu().detach().numpy().tolist()


def _any_rank_has_pending(lm, pending) -> bool:
    """Collective entry guard. True iff any rank has docs."""
    return any(_gather_ints(lm, bool(pending)))


def _gather_max_live(lm, local_n: int) -> int:
    """Collective per-turn gather. Returns ``max(live_n across ranks)``,
    used both for global termination (``== 0``) and pad-to-global-max.
    """
    return max(_gather_ints(lm, local_n))


def run_multi_turn_rollout(lm, mt_task_outputs) -> None:
    """Iterative multi-turn rollout driver for ``output_type: multi_turn_generate``.

    Per turn, for each pending doc: ask the task for the next user turn
    via ``task.next_user_turn`` (None → mark done), render the growing
    history through ``lm.apply_chat_template``, build one
    ``Instance(idx=turn_idx)``, and batch all pending instances across
    all multi-turn tasks into a single ``lm.generate_until`` call. The
    response is appended to history and ``task._instances``, then
    ``task.should_stop`` decides early-exit. Loop stops when no docs
    are pending; the standard post-process loop in ``evaluate()`` then
    consumes the variable-length per-doc instance lists the same way
    it handles ``multiple_choice``'s N-instances-per-doc.

    Distributed (``accelerate launch``) is supported when the LM
    exposes ``lm.accelerator``. Three coordination points keep
    collectives even-batched: a collective entry guard, a per-turn
    ``gather`` of live counts (drives termination and pad size), and
    sentinel-dummy padding for ranks with fewer live docs than the
    global max. Backends with internal DP (vLLM ``data_parallel_size``,
    SGLang / openai ``num_concurrent``) expose ``world_size == 1`` and
    take the non-distributed branch.

    Full architecture, hook contract, backend matrix, and limits:
    ``docs/multi_turn_rollout.md``.
    """
    from copy import deepcopy

    from lm_eval.api.instance import Instance
    from lm_eval.api.utils import Message

    if not mt_task_outputs:
        return

    # Docs are already sharded across ranks by
    # `task.doc_iterator(rank, world_size)` in
    # `_build_initial_multi_turn_states`; the coordination rules live in
    # the docstring above.
    is_distributed = lm.world_size > 1
    if is_distributed and not hasattr(lm, "accelerator"):
        raise NotImplementedError(
            "multi_turn_generate with world_size > 1 requires an "
            "`accelerator` attribute on the LM (HuggingFace + `accelerate "
            "launch`). Backends that do their own data parallelism "
            "internally should expose world_size=1."
        )

    # Each (task_output, doc_id, state) so the response routes back to
    # the right task.
    pending: list[tuple] = []
    for to in mt_task_outputs:
        for doc_id, state in to.task._multi_turn_states.items():
            pending.append((to, doc_id, state))

    # `build_all_requests` already errored if --apply_chat_template was
    # forgotten; this catches backends that don't implement it at all.
    if not hasattr(lm, "apply_chat_template"):
        raise RuntimeError(
            "multi_turn_generate requires `lm.apply_chat_template`; "
            "backend does not provide it."
        )

    # Entry guard — collective so ranks with empty shards don't strand
    # peers at the next gather. Both early-return sites must clear
    # `_multi_turn_states` (otherwise the histories built by
    # `_build_initial_multi_turn_states` leak past short-circuit).
    if is_distributed:
        if not _any_rank_has_pending(lm, pending):
            _clear_multi_turn_states(mt_task_outputs)
            return
    elif not pending:
        _clear_multi_turn_states(mt_task_outputs)
        return

    # Pad sentinel: 1-token throwaway request used by ranks whose live
    # batch is below the global max. Response is discarded.
    dummy_instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=("<pad>", {"max_gen_toks": 1, "until": []}),
        idx=-1,
        metadata=("__multi_turn_dp_pad__", -1, 1),
    )

    turn_idx = 0
    while True:
        live: list[tuple] = []
        this_turn_batch: list[tuple] = []  # (task_output, state, Instance)

        for to, doc_id, state in pending:
            user_text = to.task.next_user_turn(state.doc, state.history, turn_idx)
            if user_text is None:
                state.done = True
                continue

            state.history.append(Message("user", user_text))

            prompt = lm.apply_chat_template(
                [m.to_dict() for m in state.history],
                add_generation_prompt=True,
            )
            gen_kwargs = deepcopy(to.task.config.generation_kwargs or {})

            inst = Instance(
                request_type="generate_until",
                doc=state.doc,
                arguments=(prompt, gen_kwargs),
                idx=turn_idx,
                metadata=(
                    to.task.config.task,
                    doc_id,
                    to.task.config.repeats or 1,
                ),
            )
            this_turn_batch.append((to, state, inst))
            live.append((to, doc_id, state))

        # Termination: distributed requires EVERY rank's batch to be
        # empty before breaking (else the next gather deadlocks).
        local_n = len(this_turn_batch)
        if is_distributed:
            global_max = _gather_max_live(lm, local_n)
            if global_max == 0:
                break
        else:
            if local_n == 0:
                break
            global_max = local_n

        # Safety cap deliberately checked AFTER the per-turn gather so a
        # future refactor that diverges `turn_idx` across ranks cannot
        # cause an asymmetric break that deadlocks the next gather.
        # Today `turn_idx` is identical on every rank — all ranks break
        # together — but the order matters for that invariant.
        if turn_idx >= MAX_SAFETY_TURNS:
            eval_logger.warning(
                "multi_turn_generate driver hit safety cap of %d turns; "
                "%d docs still pending — stopping.",
                MAX_SAFETY_TURNS,
                len(pending),
            )
            break

        cloned = [inst for _, _, inst in this_turn_batch]
        numpad = global_max - local_n
        if numpad > 0:
            cloned.extend([dummy_instance] * numpad)

        responses = lm.generate_until(cloned)

        if is_distributed:
            lm.accelerator.wait_for_everyone()

        for (to, state, inst), resp in zip(
            this_turn_batch, responses[:local_n], strict=True
        ):
            inst.resps.append(resp)
            state.history.append(Message("assistant", resp))
            to.task._instances.append(inst)

            if to.task.should_stop(state.doc, state.history, turn_idx):
                state.done = True

        pending = [(to, did, st) for to, did, st in live if not st.done]
        turn_idx += 1

    eval_logger.info(
        "multi_turn_generate driver completed: %d task(s), %d total turns "
        "across %d initial docs",
        len(mt_task_outputs),
        turn_idx,
        sum(len(to.task._multi_turn_states) for to in mt_task_outputs),
    )

    _clear_multi_turn_states(mt_task_outputs)


def _clear_multi_turn_states(mt_task_outputs) -> None:
    """Release per-doc conversation histories. Called at end-of-loop
    AND each early-return in :func:`run_multi_turn_rollout`; task
    objects can persist across ``simple_evaluate`` calls (notebook /
    test flows) and would otherwise retain full N-turn histories
    indefinitely.
    """
    for to in mt_task_outputs:
        if hasattr(to.task, "_multi_turn_states"):
            del to.task._multi_turn_states


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
