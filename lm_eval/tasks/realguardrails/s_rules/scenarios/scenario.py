# Vendored from https://github.com/normster/llm_rules at SHA
# f627e569146015d7fd6f200bb758b2591f9eb6c6 (2025-02-12).
# Licensed under the Apache License, Version 2.0. Imports adapted to the
# lm_eval package layout and `typing` aliases modernized to PEP-604;
# otherwise verbatim.

import random
from dataclasses import dataclass

from ..message import Message, Role


R_ = "[REDACTED]"


@dataclass
class Result:
    passed: bool
    reason: str | None = None


class BaseScenario:
    # Initial prompt to model
    prompt: str = "Prompt template for a game."
    # Redacted prompt to show to user
    redacted_prompt: str = "Should match prompt by default"
    # Hardcoded initial model response if rules are presented as a user message
    initial_response: str = "I understand and will comply with the rules."
    # Describe user message format, if applicable
    format_message: str = ""

    @dataclass
    class Params:
        pass

    def __init__(self, param_dict: dict | None = None):
        self.p = self.Params()

    def is_valid_user_message(self, message: Message) -> bool:
        return True

    def evaluate(
        self,
        messages: list[Message],
        system_instructions: bool = False,
    ) -> Result:
        # Do not evaluate correctness of instruction messages
        start = 1 if system_instructions else 3
        for m in messages[start:]:
            if m.role == Role.USER:
                if not self.is_valid_user_message(m):
                    # raise exception for malformed user messages
                    raise ValueError(
                        f'{self.format_message} but received: "{m.content}"'
                    )
            elif m.role == Role.ASSISTANT:
                if random.random() == 0:
                    # Note: to support aggregating stats over a collection of test cases, avoid reporting
                    # specific info from the scenario/messages, e.g. specific user names.
                    return Result(False, "Assistant got unlucky")

        return Result(True)
