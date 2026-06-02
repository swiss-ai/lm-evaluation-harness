"""Per-document state for the iterative multi-turn rollout driver.

Lives outside ``api/task.py`` because both ``api/task.py`` (which
constructs instances inside ``ConfigurableTask._build_initial_multi_turn_states``)
and ``lm_eval.evaluator`` (which mutates them inside
``run_multi_turn_rollout``) need to type-annotate ``dict[int, MultiTurnState]``.
``evaluator`` already imports ``api.task``; the reverse import would
introduce a cycle, so the dataclass lives in its own module that both
sides import lazily.
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
        history: Mutable list of ``Message`` objects. Starts with the
            scenario / task's initial messages (system + any prefilled
            user/assistant turns up to but not including the first user
            turn the model must respond to). Mutated in place by the
            driver: each turn appends one ``user`` Message followed by one
            ``assistant`` Message containing the model response.
        done: True once the doc has either exhausted its user turns (the
            task's ``next_user_turn`` returned None) or hit an early-exit
            via ``should_stop``. The driver stops issuing requests for
            this doc once ``done`` is True.
    """

    doc: dict
    history: list[Message] = field(default_factory=list)
    done: bool = False
