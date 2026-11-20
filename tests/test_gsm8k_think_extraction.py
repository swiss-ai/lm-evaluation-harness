"""Unit tests for the answer extraction and scoring of gsm8k_think."""

from __future__ import annotations

import pytest

from tests._sysp_utils import load_task_utils


sc = load_task_utils("lm_eval/tasks/gsm8k/sc_utils.py", module_name="gsm8k_sc_utils")

# (response, expected extraction, id)
EXTRACT = [
    (r"so \boxed{42}.", "42", "boxed"),
    (r"\boxed{1} then \boxed{204}", "204", "boxed-last-wins"),
    (r"\boxed{\$18}", "18", "boxed-currency"),
    ("Answer: 16", "16", "marker"),
    ("Final answer is 5. Hope that helps, 100 users", "5", "marker-beats-later-number"),
    ("Answer: 18\n\nI checked 3 times", "18", "marker-beats-trailing-number"),
    ("Answer: $1,260", "1260", "marker-currency-and-comma"),
    ("Answer: -5", "-5", "marker-negative"),
    ("no marker, ends 12 then 34", "34", "last-number"),
    ("the answer is unclear", "", "no-answer"),
    # Leading zeros are dropped, whole-valued decimals collapse to an integer, and a
    # non-integer is kept as written.
    ("Answer: 007", "7", "marker-leading-zeros"),
    (r"\boxed{007}", "7", "boxed-leading-zeros"),
    ("no marker, ends 007", "7", "last-number-leading-zeros"),
    ("Answer: 18.00", "18", "marker-whole-valued-decimal"),
    ("Answer: 18.5", "18.5", "marker-decimal"),
    ("Answer: " + "9" * 500, "", "digit-cap"),
]


@pytest.mark.parametrize(
    "text,expected", [(c[0], c[1]) for c in EXTRACT], ids=[c[2] for c in EXTRACT]
)
def test_extract_answer_marked(text: str, expected: str) -> None:
    assert sc.extract_answer_marked(text) == expected


# (response, gold document answer, expected score, id)
SCORE = [
    ("Answer: 18", "so #### 18", 1, "exact"),
    ("Answer: 007", "so #### 7", 1, "leading-zeros"),
    ("Answer: 1,260", "so #### 1,260", 1, "comma-gold"),
    ("Answer: 18.00", "so #### 18", 1, "whole-valued-decimal"),
    ("Answer: 18.5", "so #### 18", 0, "decimal-is-wrong"),
    ("Answer: 19", "so #### 18", 0, "wrong"),
    ("no answer here", "so #### 18", 0, "no-answer"),
]


@pytest.mark.parametrize(
    "text,answer,expected",
    [(c[0], c[1], c[2]) for c in SCORE],
    ids=[c[3] for c in SCORE],
)
def test_think_scoring(text: str, answer: str, expected: int) -> None:
    result = sc.think_process_results({"answer": answer}, [text])
    assert result["exact_match"] == expected


def test_vote_and_pass_filters_use_the_marked_extractor() -> None:
    docs = [{"answer": "so #### 18"}]
    # The old last-number extraction would read "3" from the first response.
    resps = [
        ["Answer: 18\n\nchecked 3 times", "Answer: 17", "Answer: 018", "Answer: 5"]
    ]
    assert sc.ThinkMajorityVoteFilter().apply(resps, docs) == [
        ["Answer: 18\n\nchecked 3 times"]
    ]
    assert sc.ThinkPassAtKFilter().apply([["Answer: 1", "Answer: 018"]], docs) == [
        ["Answer: 018"]
    ]


def test_existing_task_scoring_is_unchanged() -> None:
    # gsm8k_cot_zeroshot_self_consistency keeps the last-number extraction.
    assert sc.extract_answer("Answer: 18\n\nchecked 3 times") == "3"
    assert sc.extract_answer("Answer: 007") == "007"
