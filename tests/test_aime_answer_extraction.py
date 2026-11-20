"""Unit tests for AIME answer extraction and scoring."""

from __future__ import annotations

import pytest

from tests._sysp_utils import load_task_utils


aime = load_task_utils("lm_eval/tasks/aime/utils.py", module_name="aime_utils")

# (response, expected extraction, id)
EXTRACT = [
    # Unchanged behaviour: boxed, then the last answer marker, then a trailing bare
    # number in math, then the last number anywhere.
    (r"so \boxed{42}.", "42", "boxed"),
    (r"\boxed{1} then \boxed{204}", "204", "boxed-last-wins"),
    ("Answer: 16", "16", "marker"),
    ("Final answer is 1,260.", "1260", "marker-comma-and-sentence-period"),
    ("Answer: 016", "16", "marker-zero-padded"),
    ("Answer: 007", "7", "marker-leading-zeros"),
    (r"\boxed{007}", "7", "boxed-leading-zeros"),
    ("no marker, ends 007", "7", "last-number-leading-zeros"),
    ("Answer: 18\n\nI checked 3 times", "18", "marker-beats-trailing-number"),
    ("so $$2016 \\equiv 16 \\pmod{1000}$$ hence $$ 16 $$", "16", "math-span"),
    ("no marker, ends 12 then 34", "34", "last-number"),
    ("no number at all", "", "no-answer"),
    ("Answer: " + "9" * 500, "", "digit-cap"),
    # Decimals are read whole instead of truncated to the integer part.
    ("Answer: 18.5", "18.5", "marker-decimal"),
    ("Answer: $3.50", "3.50", "marker-decimal-currency"),
    ("Answer: 18.00", "18", "marker-whole-valued-decimal"),
    ("Answer: 18.", "18", "marker-trailing-period-is-not-a-decimal"),
    (r"\boxed{18.0}", "18", "boxed-whole-valued-decimal"),
    ("the total is 144.5", "144.5", "last-number-decimal"),
    ("so $$ 7.25 $$", "7.25", "math-span-decimal"),
]


@pytest.mark.parametrize(
    "text,expected", [(c[0], c[1]) for c in EXTRACT], ids=[c[2] for c in EXTRACT]
)
def test_extract_answer(text: str, expected: str) -> None:
    assert aime.extract_answer(text) == expected


# (response, gold, expected score, id)
SCORE = [
    ("Answer: 18", "18", 1, "exact"),
    ("Answer: 18", "018", 1, "zero-padded-gold"),
    ("Answer: 007", "7", 1, "zero-padded-answer"),
    ("Answer: 18.00", "18", 1, "whole-valued-decimal"),
    # A non-integer answer is wrong, not an integer answer with a fraction cut off.
    ("Answer: 18.5", "18", 0, "decimal-not-truncated-to-gold"),
    ("Answer: 3.5", "3", 0, "decimal-not-truncated-to-gold-2"),
    ("Answer: 18.5", "185", 0, "decimal-point-is-not-dropped"),
    ("Answer: 19", "18", 0, "wrong"),
    ("no answer here", "18", 0, "no-answer"),
]


@pytest.mark.parametrize(
    "text,gold,expected",
    [(c[0], c[1], c[2]) for c in SCORE],
    ids=[c[3] for c in SCORE],
)
def test_score_response(text: str, gold: str, expected: int) -> None:
    assert aime._score_response(text, gold) == expected
