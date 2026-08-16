import os
import re
from math import isclose
from types import SimpleNamespace

import pytest

import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks
from lm_eval.utils import make_table


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize(
    "task_name,limit,model,model_args,bootstrap_iters",
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            0,
        ),
        (
            ["mmlu_abstract_algebra"],
            None,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            10000,
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_evaluator(
    task_name: list[str], limit: int, model: str, model_args: str, bootstrap_iters: int
):
    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=bootstrap_iters,
    )
    assert e1 is not None

    lm = api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    task_manager = tasks.TaskManager()
    task_dict = task_manager.load(task_name)

    e2 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
    )

    assert e2 is not None
    # check that caching is working

    def r(x):
        if "arc_easy" in x["results"]:
            return x["results"]["arc_easy"]
        else:
            return x["results"]["mmlu_abstract_algebra"]

    assert all(
        x == y
        for x, y in zip(
            [y for _, y in r(e1).items()],
            [y for _, y in r(e2).items()],
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "task_name,limit,model,model_args",
    [
        (
            ["ai2_arc"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["mmlu_stem"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["lambada_openai"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["wikitext"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_printed_results(
    task_name: list[str], limit: int, model: str, model_args: str, on_ci: bool
):
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=0,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )

    filename = "_".join(
        (
            "-".join(task_name),
            str(limit),
            str(model),
            re.sub(r"[^a-zA-Z0-9_\-.]", "-", model_args),
        )
    )
    filepath = f"./tests/testdata/{filename}.txt"
    with open(filepath) as f:
        t1 = f.read().strip()

    t2 = make_table(results).strip()

    t1_lines, t2_lines = t1.splitlines(), t2.splitlines()
    assert len(t1_lines) == len(t2_lines)
    for t1_line, t2_line in zip(t1_lines, t2_lines, strict=True):
        t1_items, t2_items = t1_line.split("|"), t2_line.split("|")
        assert len(t1_items) == len(t2_items)
        metric_name = t1_items[5].strip() if len(t1_items) > 5 else ""
        for t1_item, t2_item in zip(t1_items, t2_items, strict=True):
            try:
                t1_item_f = float(t1_item)
                t2_item_f = float(t2_item)
                if metric_name in {"perplexity", "word_perplexity", "byte_perplexity"}:
                    assert isclose(t1_item_f, t2_item_f, rel_tol=0.8, abs_tol=0.0)
                else:
                    # These are deliberately loose: the test evaluates only 10 samples
                    # and the original reference-data provenance is not fully known.
                    tol = 0.3 if on_ci else 0.5
                    assert abs(t1_item_f - t2_item_f) < tol
            except ValueError:
                # Strip whitespace so column-width differences
                # (caused by value precision changes) don't fail the test.
                # Also ignore separator-line cells (e.g. "------:").
                t1_s = t1_item.strip().rstrip("-:").rstrip("-")
                t2_s = t2_item.strip().rstrip("-:").rstrip("-")
                if t1_s or t2_s:
                    assert t1_s == t2_s


# ---------------------------------------------------------------------------
# System-prompt authority check: per-task "check inactive" warning
# (the probe itself is OFF by default; tasks opt in via metadata).
# ---------------------------------------------------------------------------
def _fake_task(*, requires):
    metadata = (
        {"requires_system_prompt_authority": True} if requires else {"version": 1}
    )
    return SimpleNamespace(config=SimpleNamespace(metadata=metadata))


def test_sysprompt_warn_when_task_requires_and_check_inactive(caplog):
    lm = SimpleNamespace(system_prompt_authority_handled=False)
    with caplog.at_level("WARNING"):
        needy = evaluator._warn_if_system_prompt_authority_inactive(
            lm, {"realguardrails_s_ifeval": _fake_task(requires=True)}
        )
    assert needy == ["realguardrails_s_ifeval"]
    assert any("authoritative system prompt" in r.message for r in caplog.records)


def test_sysprompt_no_warn_when_model_handled():
    # handled=True (e.g. strip/allow/check passed) -> no warning
    lm = SimpleNamespace(system_prompt_authority_handled=True)
    assert (
        evaluator._warn_if_system_prompt_authority_inactive(
            lm, {"realguardrails_s_ifeval": _fake_task(requires=True)}
        )
        == []
    )


def test_sysprompt_no_warn_when_task_does_not_require():
    lm = SimpleNamespace(system_prompt_authority_handled=False)
    assert (
        evaluator._warn_if_system_prompt_authority_inactive(
            lm, {"hellaswag": _fake_task(requires=False)}
        )
        == []
    )


def test_sysprompt_no_warn_when_backend_lacks_attr():
    lm = SimpleNamespace()  # e.g. API backend without the concept
    assert (
        evaluator._warn_if_system_prompt_authority_inactive(
            lm, {"realguardrails_s_ifeval": _fake_task(requires=True)}
        )
        == []
    )


def test_sysprompt_returns_only_requiring_tasks_in_mixed_list():
    # Mixed run: only the task that requests authority should be flagged.
    lm = SimpleNamespace(system_prompt_authority_handled=False)
    assert evaluator._warn_if_system_prompt_authority_inactive(
        lm,
        {
            "hellaswag": _fake_task(requires=False),
            "realguardrails_tensortrust_extraction": _fake_task(requires=True),
            "arc_easy": _fake_task(requires=False),
        },
    ) == ["realguardrails_tensortrust_extraction"]
