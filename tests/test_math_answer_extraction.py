"""Unit tests for MATH answer extraction (hendrycks_math / minerva_math)."""

from __future__ import annotations

import pytest


# These task utils raise at import time unless the `[math]` extra is installed
# (sympy / antlr4-python3-runtime==4.11 / math_verify), so skip rather than error
# in environments that do not have it.
pytest.importorskip("sympy")
pytest.importorskip("antlr4")
pytest.importorskip("math_verify")

from tests._sysp_utils import load_task_utils  # noqa: E402


UTILS = {
    "hendrycks_math": load_task_utils(
        "lm_eval/tasks/hendrycks_math/utils.py", module_name="hendrycks_math_utils"
    ),
    "minerva_math": load_task_utils(
        "lm_eval/tasks/minerva_math/utils.py", module_name="minerva_math_utils"
    ),
}

INVALID = "[invalidanswer]"

# (completion, expected extraction, id)
CASES = [
    # The phrase is present and complete: the original path, which must not change.
    (
        "Final Answer: The final answer is $[2,5)$. I hope it is correct.",
        "$[2,5)$",
        "phrase-complete",
    ),
    # The phrase is present but the trailing "I hope it is correct." is missing. The
    # function appends it; that only matches if appended with a separating space.
    ("Final Answer: The final answer is 7.", "7", "phrase-without-suffix"),
    # No phrase at all -- the few-shot examples end at "...is $X$." so a model
    # mirroring them never emits it. Fall back to the completion's own \boxed{}.
    (r"The answer is $\boxed{2}$.", "2", "boxed-only"),
    (r"So we get $\boxed{10}$.", "10", "boxed-only-2"),
    (r"first $\boxed{3}$ then later $\boxed{42}$", "42", "boxed-last-wins"),
    # Neither phrase nor box: must still be reported invalid.
    ("I could not solve this problem.", INVALID, "no-answer"),
]


@pytest.mark.parametrize("task", sorted(UTILS))
@pytest.mark.parametrize(
    "text,expected", [(c[0], c[1]) for c in CASES], ids=[c[2] for c in CASES]
)
def test_get_unnormalized_answer(task: str, text: str, expected: str) -> None:
    assert UTILS[task].get_unnormalized_answer(text) == expected
