# Vendored from https://github.com/normster/llm_rules at SHA
# f627e569146015d7fd6f200bb758b2591f9eb6c6 (2025-02-12).
# Licensed under the Apache License, Version 2.0. Imports adapted to the
# lm_eval package layout and `typing` aliases modernized to PEP-604;
# otherwise verbatim.

from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    USER = 1
    ASSISTANT = 2
    SYSTEM = 3


@dataclass
class Message:
    role: Role
    content: str

    def __str__(self):
        return f"[{self.role.name.title()}]: {self.content}"

    @staticmethod
    def serialize(messages, user_only=False) -> list[dict[str, str]]:
        if not isinstance(messages, list):
            messages = [messages]
        if user_only:
            messages = [
                {"role": m.role.name, "content": m.content}
                for m in messages
                if m.role == Role.USER
            ]
        else:
            messages = [
                {"role": m.role.name.lower(), "content": m.content} for m in messages
            ]
        return messages

    @staticmethod
    def unserialize(messages: dict | list[dict]) -> list["Message"]:
        if not isinstance(messages, list):
            messages = [messages]
        objects = [Message(Role[m["role"].upper()], m["content"]) for m in messages]
        return objects
