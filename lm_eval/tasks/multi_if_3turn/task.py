"""Scripted three-turn rollout for Multi-IF (see utils.py)."""

from __future__ import annotations

from typing import Any

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.utils import Message


class MultiIF3TurnTask(ConfigurableTask):
    """One user turn per model call; the replies are fed back as assistant turns.

    No system prompt, as upstream (multi_turn_instruct_following_eval_*.py)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        # TaskConfig rejects the `class` directive TaskManager passes through.
        if config is not None and "class" in config:
            config = {k: v for k, v in config.items() if k != "class"}
        super().__init__(config=config)

    def build_initial_messages(self, doc: dict[str, Any]) -> list[Message]:
        return []

    def next_user_turn(
        self, doc: dict[str, Any], history: list[Message], turn_idx: int
    ) -> str | None:
        if turn_idx < len(doc["turns"]):
            return doc["turns"][turn_idx]["prompt"]
        return None


__all__ = ["MultiIF3TurnTask"]
