"""Per-document state for the scripted multi-turn rollout convenience layer.

``ConfigurableTask``'s default ``init_multiturn_state`` returns one of these
for scripted multi-turn tasks (those that override ``build_initial_messages`` /
``next_user_turn`` / ``should_stop`` rather than the core ``multiturn_*`` hooks).
The evaluator's ``run_multi_turn_rollout`` driver treats episode state opaquely,
so agentic tasks (e.g. BFCL) may use their own state object instead.

Standalone module to avoid the ``api.task`` ↔ ``evaluator`` import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from lm_eval.api.utils import Message


@dataclass
class MultiTurnState:
    """Per-document conversation state during a scripted iterative rollout.

    Attributes:
        doc: The dataset row this state belongs to.
        history: Conversation messages, mutated in place. Starts with
            the task's initial messages (from ``build_initial_messages``);
            each turn the convenience layer appends one ``user`` then one
            ``assistant`` ``Message``.
        done: ``next_user_turn`` returned ``None`` or ``should_stop``
            returned ``True``. Driver skips this doc once set.
        turn_idx: Index of the next assistant turn to generate (0-based).
        responses: Model-generated assistant turns, in turn order — the
            value returned to ``process_results``.
        gen_kwargs: Generation kwargs reused for every turn of this episode.
        chat_template: Callable rendering ``history`` (as message dicts) to a
            prompt string; supplied by the driver from ``lm.apply_chat_template``.
    """

    doc: dict
    history: list[Message] = field(default_factory=list)
    done: bool = False
    turn_idx: int = 0
    responses: list[str] = field(default_factory=list)
    gen_kwargs: dict = field(default_factory=dict)
    chat_template: Callable | None = None
