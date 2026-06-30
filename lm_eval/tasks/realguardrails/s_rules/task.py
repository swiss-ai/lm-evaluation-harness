"""Custom ConfigurableTask for S-RuLES (iterative multi-turn rollout).

S-RuLES rows carry prefilled multi-turn dialogue context (0-6 prefilled
assistant turns) followed by 1-7 trailing user turns the model must
respond to. Upstream ``scripts/evaluate_batched.py:323`` of
``normster/llm_rules`` generates ONE assistant turn at a time and
re-evaluates after each, early-exiting on first scenario failure.

This task uses the harness's ``output_type: multi_turn_generate`` to
get exactly that behavior. The evaluator's per-turn driver
(``run_multi_turn_rollout`` in ``lm_eval/evaluator.py``) drives the core
``multiturn_*`` hooks; ``ConfigurableTask``'s default implementations of
those hooks delegate to the three scripted hooks we expose here:

* :meth:`build_initial_messages` — system + any prefilled context up to
  but not including the first trailing user turn.
* :meth:`next_user_turn` — yields the next trailing user turn from
  ``doc["messages"]``, or None when exhausted.
* :meth:`should_stop` — re-runs ``scenario.evaluate(...)`` after each
  assistant turn and returns True on scenario failure (matches
  upstream's early-exit).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.utils import Message

# Absolute imports — see utils.py for the rationale. The `!function`
# loader in lm_eval/utils.py:import_function bypasses package init.
from lm_eval.tasks.realguardrails.s_rules.message import Message as RGMessage
from lm_eval.tasks.realguardrails.s_rules.utils import (
    _instantiate_scenario,
    _system_prompt_for,
    _trailing_user_messages,
    build_messages_for_doc,
)


class SRulesTask(ConfigurableTask):
    """ConfigurableTask wiring for S-RuLES.

    The class is referenced from leaf YAMLs as
    ``class: !function task.SRulesTask``. All other task-config knobs
    (``dataset_path``, ``generation_kwargs``, ``metric_list``,
    ``process_results``, etc.) flow through unchanged.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # TaskManager.load passes the full YAML config (including the
        # `class` directive itself) into the python_task constructor.
        # `TaskConfig.__init__` rejects the unknown `class` keyword, so
        # strip it before delegating. Avoid mutating the caller's dict.
        if config is not None and "class" in config:
            config = {k: v for k, v in config.items() if k != "class"}
        super().__init__(config=config)

    def download(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Resolve relative ``data_files`` paths against this module's directory.

        The leaf YAMLs ship ``data_files`` as
        ``{split: data/<bucket>.jsonl}`` — a path relative to the task
        directory, not CWD. ``datasets.load_dataset`` interprets relative
        paths against ``os.getcwd()``, which would break the task when
        ``lm-eval`` is invoked from anywhere other than the repo root.
        Anchor to ``Path(__file__).parent`` here and let the parent
        download() do everything else unchanged.

        Falls back to the configured ``dataset_kwargs`` on a bare
        ``download()`` (e.g. the CI task-scan test) so the anchoring still
        runs — otherwise the parent would reload the unanchored relative path.
        """
        if dataset_kwargs is None and self.config.dataset_kwargs:
            dataset_kwargs = self.config.dataset_kwargs
        if dataset_kwargs and "data_files" in dataset_kwargs:
            data_files = dataset_kwargs["data_files"]
            base = Path(__file__).parent
            anchored: dict[str, Any] = {}
            for split, path in data_files.items():
                if isinstance(path, str) and not os.path.isabs(path):
                    anchored[split] = str(base / path)
                else:
                    anchored[split] = path
            dataset_kwargs = {**dataset_kwargs, "data_files": anchored}
        return super().download(dataset_kwargs, **kwargs)

    # =========================================================================
    # multi_turn_generate hooks
    # =========================================================================

    def build_initial_messages(self, doc: dict[str, Any]) -> list[Message]:
        """Return ``[system, ...prefilled_user_asst_pairs]``.

        The trailing user turns the model is asked to respond to are
        NOT included here; :meth:`next_user_turn` yields them one at a
        time. This matches upstream ``build_initial_messages`` at
        ``scripts/evaluate_batched.py``.
        """
        full = build_messages_for_doc(doc)  # [system, ...all from doc["messages"]]
        trailing = _trailing_user_messages(doc["messages"])
        prefix = full[: len(full) - len(trailing)]
        return [Message(m["role"], m["content"]) for m in prefix]

    def next_user_turn(
        self,
        doc: dict[str, Any],
        history: list[Message],
        turn_idx: int,
    ) -> str | None:
        """Yield trailing user turn ``turn_idx`` from ``doc["messages"]``."""
        trailing = _trailing_user_messages(doc["messages"])
        if turn_idx < len(trailing):
            return trailing[turn_idx]["content"]
        return None

    def should_stop(
        self,
        doc: dict[str, Any],
        history: list[Message],
        turn_idx: int,
    ) -> bool:
        """Re-run ``scenario.evaluate`` after each assistant turn.

        Returns True (early-exit) on scenario failure or on a
        ``ValueError`` from the evaluator (which signals a malformed user
        message — upstream's contract). Matches the
        ``response.append(); evaluate(); break on fail`` loop in
        ``evaluate_batched.py``.
        """
        scenario = _instantiate_scenario(doc)
        sys_msg = {"role": "system", "content": _system_prompt_for(scenario)}
        chat_dicts = [sys_msg] + [m.to_dict() for m in history]
        try:
            return not scenario.evaluate(
                RGMessage.unserialize(chat_dicts),
                system_instructions=True,
            ).passed
        except ValueError as err:
            # Upstream `evaluate_batched.py` lets ValueError propagate
            # (a malformed user message crashes the worker). We treat it
            # as an early-exit failure for graceful evaluation, but log
            # so the divergence is visible.
            import logging

            logging.getLogger(__name__).warning(
                "scenario.evaluate raised ValueError on %r (turn %d): %s — "
                "treating as early-exit failure.",
                doc.get("scenario"),
                turn_idx,
                err,
            )
            return True


__all__ = ["SRulesTask"]
