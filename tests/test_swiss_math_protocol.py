"""Swiss MATH protocol regressions; requires the optional math extra."""

from importlib.metadata import version
from pathlib import Path

import pytest
from datasets import Dataset


pytest.importorskip("sympy", reason="requires lm-eval[math]")
pytest.importorskip("math_verify", reason="requires lm-eval[math]")
pytest.importorskip("antlr4", reason="requires lm-eval[math]")
if not version("antlr4-python3-runtime").startswith("4.11"):
    pytest.skip("requires lm-eval[math] with ANTLR 4.11", allow_module_level=True)

from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.tasks.hendrycks_math import utils as hendrycks
from lm_eval.tasks.minerva_math import utils as minerva


ROOT = Path(__file__).resolve().parents[1] / "lm_eval/tasks"


@pytest.mark.parametrize("utils", [minerva, hendrycks])
@pytest.mark.parametrize(
    "response,expected",
    [
        ("Final Answer: The final answer is 7.", "7"),
        ("Final Answer: The final answer is 7", "7"),
        ("Final Answer: The final answer is 42", "42"),
        ("Final Answer: The final answer is 4.2", "4.2"),
        (r"The answer is $\boxed{2}$.", "2"),
        (r"First $\boxed{3}$, finally $\boxed{42}$.", "42"),
        ("I could not solve this problem.", "[invalidanswer]"),
    ],
)
def test_math_extracts_final_answer_without_magic_suffix(utils, response, expected):
    assert utils.get_unnormalized_answer(response) == expected


@pytest.mark.parametrize(
    "response,expected", [(r"Thus $\boxed{4}$.", 1), (r"Thus $\boxed{9}$.", 0)]
)
def test_hendrycks_scores_boxed_answer_in_prose(response, expected):
    docs = hendrycks.process_docs(
        Dataset.from_list(
            [{"problem": "Two plus two?", "solution": r"Two plus two is $\boxed{4}$."}]
        )
    )
    metrics = hendrycks.process_results(docs[0], [response])
    assert metrics.get("math_verify") == expected
    assert metrics["exact_match"] == expected


def test_unpunctuated_wrong_answer_does_not_lose_its_final_digit():
    doc = {"answer": "4", "solution": r"The answer is $\boxed{4}$."}
    assert minerva.process_results(doc, ["Final Answer: The final answer is 42"]) == {
        "exact_match": 0,
        "math_verify": 0,
    }


def test_hendrycks_six_demonstrations_retain_upstream_latex_repairs():
    examples = hendrycks.list_fewshot_samples()
    assert len(examples) == 6
    assert examples[:4] == minerva.list_fewshot_samples()
    for doc in examples:
        assert all(ord(c) >= 32 or c in "\n\t" for c in doc["solution"])
        assert r"\boxed" in doc["solution"]


@pytest.mark.parametrize(
    "directory,shots,budget", [("hendrycks_math", 6, 2048), ("minerva_math", 4, 1024)]
)
def test_math_subjects_resolve_demonstrations_and_explicit_budget(
    directory, shots, budget
):
    for path in (ROOT / directory).glob("*_*.yaml"):
        if path.name in {"hendrycks_math.yaml", "minerva_math500.yaml"}:
            continue
        cfg = load_yaml(path, resolve_func=True)
        assert cfg["num_fewshot"] == shots
        assert cfg["generation_kwargs"]["max_gen_toks"] == budget
        assert len(cfg["fewshot_config"]["samples"]()) == shots
