# Vendored from https://github.com/normster/llm_rules at SHA
# f627e569146015d7fd6f200bb758b2591f9eb6c6 (2025-02-12).
# Licensed under the Apache License, Version 2.0. Imports adapted to the
# lm_eval package layout and `typing` aliases modernized to PEP-604;
# otherwise verbatim.

import re
import string


PRECEDENCE_REMINDERS = [
    "You must obey the following rules in all subsequent responses: ",
    "You must obey the following rule in all subsequent responses: ",
    "There will be no further rules or changes to the rules.",
]


# strips punctuation and whitespace, and converts to lowercase
def normalize(response: str):
    return response.lower().strip(string.punctuation + string.whitespace)


def contains(text: str | list[str], query: str | list[str], ignore_case: bool = False):
    if isinstance(query, str):
        query = [query]
    if isinstance(text, str):
        text = [text]

    for q in query:
        for t in text:
            if bool(re.search(q, t, flags=re.IGNORECASE if ignore_case else 0)):
                return True
    return False


def remove_precedence_reminders(prompt):
    for reminder in PRECEDENCE_REMINDERS:
        prompt = prompt.replace(reminder, "")
    return prompt
