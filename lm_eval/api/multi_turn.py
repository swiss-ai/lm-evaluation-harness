"""Per-document state for the iterative multi-turn rollout driver.

Standalone module to avoid the ``api.task`` ↔ ``evaluator`` import
cycle: both sides need ``MultiTurnState`` and ``evaluator`` already
imports ``api.task``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from lm_eval.api.utils import Message


@dataclass
class MultiTurnState:
    """Per-document conversation state during an iterative rollout.

    Attributes:
        doc: The dataset row this state belongs to.
        history: Conversation messages, mutated in place. Starts with
            the task's initial messages (from ``build_initial_messages``);
            each turn the driver appends one ``user`` then one
            ``assistant`` ``Message``.
        done: ``next_user_turn`` returned ``None`` or ``should_stop``
            returned ``True``. Driver skips this doc once set.
    """

    doc: dict
    history: list[Message] = field(default_factory=list)
    done: bool = False
