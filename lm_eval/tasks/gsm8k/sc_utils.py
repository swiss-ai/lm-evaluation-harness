"""Self-consistency metrics for zero-shot GSM8K (score-first / mean@k / maj@k / pass@k).

GSM8K answers are integers, so we extract the final numeric answer from each
sampled response and score each attempt individually. A custom ``process_results``
is required because the built-in ``exact_match`` metric only scores a single
response per doc (``results[0]``) and cannot average over the k samples needed for
mean@k — see ``lm_eval/api/task.py``. The maj@k / pass@k filters mirror the AIME
self-consistency setup, differing only in how the answer is extracted.
"""

import re
from collections import Counter

from lm_eval.api.metrics import is_degenerating_text


# Same numeric-answer pattern GSM8K's zero-shot/CoT filters use: the last number
# (optionally $-prefixed, comma-grouped, or decimal) not followed by another digit.
_NUMBER_RE = re.compile(r"(\$?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\$?)(?![0-9])")


def _clean_number(text: str) -> str:
    return text.replace(",", "").replace(" ", "").replace("$", "")


def extract_answer(response: str) -> str:
    """Extract the final numeric answer from a model response (``''`` if none)."""
    matches = _NUMBER_RE.findall(response)
    return _clean_number(matches[-1]) if matches else ""


def extract_gold(doc: dict) -> str:
    """The gold numeric answer (after the ``####`` marker) from a GSM8K doc."""
    gold = str(doc["answer"])
    if "####" in gold:
        gold = gold.split("####")[-1]
    return _clean_number(gold)


def _score_response(response: str, gold: str) -> int:
    """1 if the response's extracted answer matches the gold answer, else 0."""
    pred = extract_answer(response)
    return int(pred != "" and pred == gold)


class MajorityVoteFilter:
    """Self-consistency (maj@k) filter.

    Majority vote over the extracted answers, returning a representative *full*
    response whose answer matches the winner so the downstream ``process_results``
    extraction and the ``degeneration`` metric keep working unchanged. Empty
    extractions are ignored while voting unless every sample is empty.
    Chain a ``take_first`` after this filter to unwrap the single voted response.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list):
            answers = [extract_answer(r) for r in resp_list]
            votable = [a for a in answers if a != ""] or answers
            winner = Counter(votable).most_common(1)[0][0]
            for resp, ans in zip(resp_list, answers):
                if ans == winner:
                    return resp
            return resp_list[0]

        return [[select(r)] for r in resps]


class PassAtKFilter:
    """pass@k filter.

    A problem counts as solved if *any* of the k sampled responses is correct.
    Using the gold answer (available via ``docs``), this returns a correct sample
    when one exists, else the first sample, so the single-response scoring in
    ``process_results`` yields 1 iff at least one attempt was correct. With n == k
    samples this equals the standard unbiased pass@k estimator.
    Chain a ``take_first`` after this filter to unwrap the selected response.
    """

    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, resps, docs):
        def select(resp_list, doc):
            gold = extract_gold(doc)
            for resp in resp_list:
                if _score_response(resp, gold):
                    return resp
            return resp_list[0]

        return [[select(r, d)] for r, d in zip(resps, docs)]


def process_results(doc: dict, results: list) -> dict[str, float]:
    # A filter may collapse the repeats to a single response (take_first, maj@k,
    # pass@k) or pass them all through (take_first_k for mean@k). Normalize to a
    # list so a single metric path handles both.
    response = results[0]
    responses = list(response) if isinstance(response, list) else [response]

    gold = extract_gold(doc)

    # For a single response this is 0/1 (per-sample accuracy); for the full set of
    # k samples it is the mean accuracy over the k attempts (mean@k / avg@k).
    scores = [_score_response(r, gold) for r in responses]
    degens = [int(is_degenerating_text(r.lower())) for r in responses]

    return {
        "exact_match": sum(scores) / len(scores),
        "degeneration": sum(degens) / len(degens),
    }
