from __future__ import annotations

import hashlib
import random
import re
from collections import Counter
from typing import TYPE_CHECKING

from lm_eval.api.metrics import is_degenerating_text


if TYPE_CHECKING:
    import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Same docs as the other GPQA variants, but the choice order is seeded per question.

    The other variants call an unseeded ``random.shuffle``, so the score can differ from
    run to run. Here the order depends only on the question text, so repeated runs (and
    the k samples of one run) see the same prompt.
    """

    def _process_doc(doc):
        correct = preprocess(doc["Correct Answer"])
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            correct,
        ]
        seed = int(hashlib.sha256(doc["Question"].encode("utf-8")).hexdigest(), 16)
        random.Random(seed).shuffle(choices)
        correct_answer_index = choices.index(correct)

        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": choices,
            "answer": f"({chr(65 + correct_answer_index)})",
        }

    return dataset.map(_process_doc)


# --- Thinking runs ---------------------------------------------------------------
# The answer is read from the final reply only. The model-level strip
# (`think_end_token` / `autodetect_think_tokens`) usually removed the reasoning already;
# stripping here is idempotent with it and covers runs without it.

THINK_START = "<think>"
THINK_END = "</think>"

# "The answer is (B)", "Answer: B", "**Answer:** (B)", "answer is: **B**", ...
_ANSWER_RE = re.compile(
    r"answer(?:\s+is)?\s*[:\-]?\s*[\*\s]*[\(\[]?\s*([A-D])\s*[\)\]]?(?![A-Za-z])",
    re.IGNORECASE,
)


def strip_thinking(response: str) -> str:
    """Final reply of a thinking response.

    The text after the last ``</think>``. A reply that opens ``<think>`` but never
    closes it (truncated mid-reasoning) has no final reply, so it is empty and is
    scored wrong. A response with no think tags at all is returned unchanged.
    """
    if THINK_END in response:
        return response.rpartition(THINK_END)[2].strip()
    if response.lstrip().startswith(THINK_START):
        return ""
    return response


def think_answer(response: str) -> str:
    """Chosen letter as ``"(X)"`` from the final reply, ``""`` if there is none.

    Only an explicit "answer is X" / "Answer: X" marker counts (the last one wins). There
    is deliberately no catch-all fallback to any stray A-D.
    """
    matches = _ANSWER_RE.findall(strip_thinking(response))
    return f"({matches[-1].upper()})" if matches else ""


def _score(response: str, doc: dict) -> int:
    return int(think_answer(response) == doc["answer"])


class ThinkMajorityVoteFilter:
    """maj@k over ``think_answer`` (empty answers ignored unless all are empty).
    Chain a ``take_first`` after this filter."""

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list):
            answers = [think_answer(r) for r in resp_list]
            votable = [a for a in answers if a] or answers
            winner = Counter(votable).most_common(1)[0][0]
            for resp, ans in zip(resp_list, answers):
                if ans == winner:
                    return resp
            return resp_list[0]

        return [[select(r)] for r in resps]


class ThinkPassAtKFilter:
    """pass@k: returns a correct response if any of the k exists, else the first.
    Chain a ``take_first`` after this filter."""

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list, doc):
            for resp in resp_list:
                if _score(resp, doc):
                    return resp
            return resp_list[0]

        return [[select(r, d)] for r, d in zip(resps, docs)]


def think_process_results(doc: dict, results: list) -> dict[str, float]:
    """Accuracy and degeneration rate for thinking runs: one response, or all k for
    mean@k. ``degeneration`` is computed on the raw response, since looping happens in
    the reasoning."""
    response = results[0]
    responses = list(response) if isinstance(response, list) else [response]

    scores = [_score(r, doc) for r in responses]
    degens = [int(is_degenerating_text(r.lower())) for r in responses]

    return {
        "acc": sum(scores) / len(scores),
        "degeneration": sum(degens) / len(degens),
    }
