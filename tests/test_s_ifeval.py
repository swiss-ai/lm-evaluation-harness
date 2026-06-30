"""Unit tests for the S-IFEval scoring shim (realguardrails / s_ifeval).

The reused IFEval verifier registry is mocked out — its own correctness is
covered by the upstream IFEval tests; here we only validate the skip-list
and verifier-wiring semantics specific to S-IFEval.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from tests._sysp_utils import FakeDataset, load_task_utils


@pytest.fixture
def utils(monkeypatch):
    """Inject a stub ``lm_eval.tasks.ifeval.utils`` so the s_ifeval shim
    can be imported (and exercised) without the heavy verifier deps.
    """
    fake_pkg = types.ModuleType("lm_eval")
    fake_tasks = types.ModuleType("lm_eval.tasks")
    fake_ifeval = types.ModuleType("lm_eval.tasks.ifeval")
    fake_ifeval_utils = types.ModuleType("lm_eval.tasks.ifeval.utils")

    class _InputExample:
        def __init__(self, *, key, instruction_id_list, prompt, kwargs):
            self.key = key
            self.instruction_id_list = instruction_id_list
            self.prompt = prompt
            self.kwargs = kwargs

    class _OutputExample:
        def __init__(self, *, follow_all_instructions, follow_instruction_list):
            self.follow_all_instructions = follow_all_instructions
            self.follow_instruction_list = follow_instruction_list

    fake_ifeval_utils.InputExample = _InputExample
    fake_ifeval_utils.OutputExample = _OutputExample
    fake_ifeval_utils.test_instruction_following_strict = MagicMock()
    fake_ifeval_utils.test_instruction_following_loose = MagicMock()
    fake_ifeval_utils.agg_inst_level_acc = MagicMock(side_effect=lambda items: 0.5)

    monkeypatch.setitem(sys.modules, "lm_eval", fake_pkg)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.ifeval", fake_ifeval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks.ifeval.utils", fake_ifeval_utils)

    mod = load_task_utils(
        "lm_eval/tasks/realguardrails/s_ifeval/utils.py",
        module_name="s_ifeval_utils",
    )
    mod._fake_ifeval_utils = fake_ifeval_utils
    return mod


# ---------------------------------------------------------------------------
# Skip list filtering
# ---------------------------------------------------------------------------


def test_skip_list_drops_specified_ids(utils):
    doc = {
        "instruction_id_list": [
            "keywords:existence",
            "language:response_language",
            "punctuation:no_comma",
        ],
        "kwargs": [{"keywords": ["foo"]}, {"language": "en"}, {}],
    }
    eff_ids, eff_kwargs = utils._effective_instructions(doc)
    assert eff_ids == ["keywords:existence", "punctuation:no_comma"]
    assert eff_kwargs == [{"keywords": ["foo"]}, {}]


def test_process_docs_drops_any_row_intersecting_skip_set(utils):
    """Upstream evaluate.py drops a row when its instruction_id_list has
    *any* element in the skip set, not only when every element is skipped.
    The mixed row (id=2) must therefore be dropped — keeping it would
    silently score the model on a partial constraint set and yield 506
    effective rows instead of the paper's 470.
    """
    ds = FakeDataset(
        [
            {"id": 0, "instruction_id_list": ["language:response_language"]},
            {"id": 1, "instruction_id_list": ["combination:repeat_prompt"]},
            {
                "id": 2,
                "instruction_id_list": [
                    "language:response_language",
                    "keywords:existence",
                ],
            },
            {"id": 3, "instruction_id_list": ["keywords:existence"]},
            {
                "id": 4,
                "instruction_id_list": [
                    "keywords:existence",
                    "punctuation:no_comma",
                ],
            },
        ]
    )
    kept = utils.process_docs(ds)
    assert sorted(r["id"] for r in kept.rows) == [3, 4]


# ---------------------------------------------------------------------------
# process_results forwards filtered inputs to the verifier
# ---------------------------------------------------------------------------


def _set_verifier_returns(utils, strict_all, strict_list, loose_all, loose_list):
    OutputExample = utils._fake_ifeval_utils.OutputExample
    utils._fake_ifeval_utils.test_instruction_following_strict.return_value = (
        OutputExample(
            follow_all_instructions=strict_all,
            follow_instruction_list=strict_list,
        )
    )
    utils._fake_ifeval_utils.test_instruction_following_loose.return_value = (
        OutputExample(
            follow_all_instructions=loose_all,
            follow_instruction_list=loose_list,
        )
    )


def test_process_results_filters_skip_then_calls_verifier(utils):
    doc = {
        "key": 1005,
        "prompt": "original combined prompt",
        "instruction_id_list": [
            "keywords:existence",
            "combination:repeat_prompt",  # skipped
        ],
        "kwargs": [{"keywords": ["foo"]}, {"prompt_to_repeat": "..."}],
        "messages": [
            {"role": "system", "content": "constraint"},
            {"role": "user", "content": "base task"},
        ],
    }
    _set_verifier_returns(
        utils,
        strict_all=True,
        strict_list=[True],
        loose_all=True,
        loose_list=[True],
    )
    out = utils.process_results(doc, ["model output"])

    strict_call = utils._fake_ifeval_utils.test_instruction_following_strict.call_args
    inp = strict_call.args[0]
    assert inp.instruction_id_list == ["keywords:existence"]
    assert inp.kwargs == [{"keywords": ["foo"]}]
    assert inp.prompt == "original combined prompt"

    assert out == {
        "prompt_level_strict_acc": True,
        "inst_level_strict_acc": [True],
        "prompt_level_loose_acc": True,
        "inst_level_loose_acc": [True],
    }


def test_process_results_empty_after_skip_is_vacuous_pass(utils):
    doc = {
        "key": 1,
        "prompt": "x",
        "instruction_id_list": ["language:response_language"],
        "kwargs": [{}],
    }
    out = utils.process_results(doc, ["model output"])
    assert out == {
        "prompt_level_strict_acc": True,
        "inst_level_strict_acc": [],
        "prompt_level_loose_acc": True,
        "inst_level_loose_acc": [],
    }
    utils._fake_ifeval_utils.test_instruction_following_strict.assert_not_called()
    utils._fake_ifeval_utils.test_instruction_following_loose.assert_not_called()


def test_agg_inst_level_acc_delegates_to_ifeval(utils):
    result = utils.agg_inst_level_acc([[True, False], [True]])
    assert result == 0.5
    utils._fake_ifeval_utils.agg_inst_level_acc.assert_called_once()
