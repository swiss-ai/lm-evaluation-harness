"""S-RuLES task glue: per-doc message building + scoring against the
vendored scenario evaluators from ``normster/llm_rules``.

The system prompt for each row is the scenario instance's ``prompt``
attribute (rendered at instantiation time from ``doc["params"]``), with
the three precedence-reminder phrases stripped — matching upstream's
``--remove_precedence_reminders`` flag.

The scoring entry point is :func:`process_results`, which:
  1. Reconstructs the full conversation ``[system, prefilled..., model]``
     as ``Message`` objects.
  2. Instantiates the scenario with the row's params.
  3. Calls ``scenario.evaluate(messages, system_instructions=True)``
     (the paper setting — rules live in the system role, not in a user
     turn that the model must echo back).
  4. Returns ``{"passed": 0 or 1}``. A ``ValueError`` raised by
     ``is_valid_user_message`` (against the rules format pinned to
     upstream SHA ``f627e56``) is counted as a failed row so the
     denominator still includes it.
"""

from __future__ import annotations

import json
from typing import Any

# Absolute imports — see the comment in task.py. The `!function`
# loader in lm_eval/utils.py:import_function bypasses package init,
# so relative imports raise ImportError.
from lm_eval.tasks.realguardrails.s_rules.message import Message
from lm_eval.tasks.realguardrails.s_rules.scenarios import SCENARIOS
from lm_eval.tasks.realguardrails.s_rules.scenarios.utils import (
    remove_precedence_reminders,
)


def _doc_params(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the row's scenario params as a ``dict``.

    The bucket JSONLs store ``params`` as a JSON-encoded string
    (``params_json``) so HF Datasets sees a homogeneous Value("string")
    column across scenarios with heterogeneous param schemas (Encryption:
    ``{secret}``, Millionaires: ``{balance1, balance2, user1, user2}``,
    ForbiddenWord: ``{word}``, etc.). Test helpers may construct docs
    in-memory with a plain ``params`` dict — accept both shapes.
    """
    if "params_json" in doc:
        raw = doc["params_json"]
        return json.loads(raw) if isinstance(raw, str) else raw
    return doc.get("params", {})


def _instantiate_scenario(doc: dict[str, Any]) -> Any:
    """Build a scenario instance from the row's params.

    Centralizes the SCENARIOS lookup + ``__init__(param_dict=...)`` call
    so callers don't accidentally instantiate twice.
    """
    return SCENARIOS[doc["scenario"]](param_dict=_doc_params(doc))


def _system_prompt_for(scenario_instance: Any) -> str:
    """Render the scenario's system prompt with precedence reminders stripped.

    ``scenario_instance.prompt`` is rendered at ``__init__`` time from
    ``template`` + ``params``; stripping happens on the rendered string
    so identical phrasings inside the template body are not also
    removed (the reminder strings are distinctive enough that this is
    not a concern in practice).
    """
    return remove_precedence_reminders(scenario_instance.prompt)


def build_messages_for_doc(doc: dict[str, Any]) -> list[dict[str, str]]:
    """Build ``[system, ...prefilled_turns]`` dicts for an S-RuLES row.

    The prefilled turns come from ``doc["messages"]`` verbatim. The
    system content is the scenario's rendered ``prompt`` with precedence
    reminders stripped. The final element of ``doc["messages"]`` is the
    user turn the model must respond to; trailing the list with the
    model's response is the job of :func:`process_results`.
    """
    scenario_instance = _instantiate_scenario(doc)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt_for(scenario_instance)}
    ]
    for msg in doc["messages"]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    return messages


def _trailing_user_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the suffix of ``messages`` starting after the last assistant turn."""
    last_asst_idx = -1
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            last_asst_idx = i
    return list(messages[last_asst_idx + 1 :])


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, int]:
    """Score the rollout outcome against the scenario evaluator.

    ``results`` is the list of model-generated assistant turns, in turn
    order. Under :class:`SRulesTask`'s ``output_type: multi_turn_generate``
    the evaluator runs the iterative rollout, calls ``should_stop`` after
    each turn (which early-exits on scenario failure), and finally
    invokes this method with whatever turns were generated. A row that
    early-exited at turn ``k`` has ``len(results) == k+1`` and the final
    assistant turn is the one that triggered the failure.

    Conversation rebuilt for the final ``scenario.evaluate`` call::

        [system, ...prefilled..., user_0, results[0], user_1, results[1], ..., user_n, results[n]]

    where ``user_i`` is the i-th *trailing* user turn from
    ``doc["messages"]``. ``scenario.evaluate`` then scans the full
    conversation as it does upstream; passing implies every turn obeyed
    the rule, failing implies one of them did not. Test helpers can
    still pass a single string (legacy single-shot shape); that path is
    treated as ``[that_one_string]``.
    """
    if isinstance(results, str):
        # Legacy single-string input from older test fixtures.
        results = [results]

    scenario_instance = _instantiate_scenario(doc)
    sys_msg = {"role": "system", "content": _system_prompt_for(scenario_instance)}

    str_results = [r if isinstance(r, str) else str(r) for r in results]

    trailing = _trailing_user_messages(doc["messages"])
    prefilled_dicts = [
        {"role": m["role"], "content": m["content"]}
        for m in doc["messages"][: len(doc["messages"]) - len(trailing)]
    ]

    interleaved: list[dict[str, Any]] = []
    for i, user_msg in enumerate(trailing):
        if i >= len(str_results):
            # The driver early-exited before this turn ran; nothing more
            # to score. Stop appending.
            break
        interleaved.append({"role": "user", "content": user_msg["content"]})
        interleaved.append({"role": "assistant", "content": str_results[i]})

    chat_dicts = [sys_msg] + prefilled_dicts + interleaved
    chat_messages = Message.unserialize(chat_dicts)

    try:
        result = scenario_instance.evaluate(chat_messages, system_instructions=True)
        passed = int(result.passed)
    except ValueError:
        passed = 0

    return {"passed": passed}
