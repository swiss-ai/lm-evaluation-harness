"""S-IFEval (System-IFEval) scoring for RealGuardrails / SystemCheck.

Reuses the canonical Google IFEval verifier registry from
``lm_eval.tasks.ifeval`` without modification — the paper confirmed that
SystemCheck adds no new instruction types. The only S-IFEval-specific
behavior implemented here:

  1. **Skip list.** Skip the instructions ``language:response_language``
     and ``combination:repeat_prompt`` at scoring time, matching the
     ``--instruction_ids_to_skip`` flag in
     ``evals/ifeval/read.sh`` of github.com/normster/RealGuardrails.
  2. **Row drop.** ``process_docs`` removes rows whose
     ``instruction_id_list`` intersects the skip set, matching the
     upstream::

         if _INSTRUCTION_IDS_TO_SKIP.value and len(
             set(example["instruction_id_list"]).intersection(
                 set(_INSTRUCTION_IDS_TO_SKIP.value))):
             continue

     This produces the paper's 470-of-541 effective denominator. A
     per-instruction filter (keeping rows that mix skipped + non-skipped
     IDs) would leave 36 rows behind and silently score the model on a
     partial constraint set.

The per-doc system prompt (constraints) is supplied to the model via the
task YAML's ``description: "{{ messages[0]['content'] }}"`` together with
``--apply_chat_template``; the user message (base task) is the
``doc_to_text`` field.
"""

from __future__ import annotations

from typing import Any

from lm_eval.tasks.ifeval import utils as ifeval_utils


# Upstream read.sh:
#   --instruction_ids_to_skip language:response_language,combination:repeat_prompt
SKIP_INSTRUCTION_IDS: frozenset[str] = frozenset(
    {
        "language:response_language",
        "combination:repeat_prompt",
    }
)


def _has_skipped_id(doc: dict[str, Any]) -> bool:
    return bool(set(doc["instruction_id_list"]) & SKIP_INSTRUCTION_IDS)


def _effective_instructions(
    doc: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (instruction_ids, kwargs) with skip-listed entries removed."""
    eff_ids: list[str] = []
    eff_kwargs: list[dict[str, Any]] = []
    for inst_id, kw in zip(doc["instruction_id_list"], doc["kwargs"], strict=True):
        if inst_id in SKIP_INSTRUCTION_IDS:
            continue
        eff_ids.append(inst_id)
        eff_kwargs.append(kw)
    return eff_ids, eff_kwargs


def process_docs(dataset):
    """Drop rows whose ``instruction_id_list`` intersects the skip set.

    See the module docstring for the upstream rationale. After this hook,
    every reaching row has zero skipped instructions, so the
    per-instruction filtering in ``process_results`` is defensive only
    (kept in case a caller bypasses ``process_docs`` by overriding the
    task config).
    """
    return dataset.filter(lambda d: not _has_skipped_id(d))


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    response = next(iter(results))

    eff_ids, eff_kwargs = _effective_instructions(doc)
    # Defensive vacuous-pass branch — reachable only if the dataset is
    # loaded without process_docs.
    if not eff_ids:
        return {
            "prompt_level_strict_acc": True,
            "inst_level_strict_acc": [],
            "prompt_level_loose_acc": True,
            "inst_level_loose_acc": [],
        }

    inp = ifeval_utils.InputExample(
        key=doc["key"],
        instruction_id_list=eff_ids,
        # The verifier's `build_description` consults `inp.prompt` only for
        # instructions that declare 'prompt' as an arg (notably
        # combination:repeat_prompt, which is in the skip list and thus
        # filtered out upstream). Pass the original concatenated IFEval
        # prompt so any future un-skipped prompt-referencing verifier
        # still behaves correctly.
        prompt=doc["prompt"],
        kwargs=eff_kwargs,
    )
    out_strict = ifeval_utils.test_instruction_following_strict(inp, response)
    out_loose = ifeval_utils.test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


# Thin shim so the YAML can reference a single namespace for all scoring
# functions; lets us swap the IFEval verifier impl without touching the YAML.
def agg_inst_level_acc(items):
    return ifeval_utils.agg_inst_level_acc(items)
