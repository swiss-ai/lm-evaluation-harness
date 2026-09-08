"""MBPP response extraction without loading or executing the code-eval metric."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import evaluate
import pytest


@pytest.fixture
def mbpp(monkeypatch):
    # utils.py probes code_eval at import. Replace only that external scorer;
    # the production extraction and response-batch functions run unchanged.
    monkeypatch.setattr(
        evaluate,
        "load",
        lambda *args, **kwargs: SimpleNamespace(compute=lambda **kw: None),
    )
    path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/mbpp/utils.py"
    spec = importlib.util.spec_from_file_location("mbpp_extraction_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            "def sort_matrix(matrix):\n    return sorted(matrix)\n```",
            "def sort_matrix(matrix):\n    return sorted(matrix)",
        ),
        ("import math\nvalue = math.pi\n```", "import math\nvalue = math.pi"),
        ("from math import pi\n```", "from math import pi"),
        (
            "async def answer():\n    return 42\n```",
            "async def answer():\n    return 42",
        ),
        ("class Answer:\n    value = 42\n```", "class Answer:\n    value = 42"),
        ("value = 42", "value = 42"),
        ("def answer():\n    return 42", "def answer():\n    return 42"),
        (
            "```python\ndef answer():\n    return 42\n```",
            "def answer():\n    return 42",
        ),
        ("```\nimport math\n```", "import math"),
        ("\n```py\nvalue = 42\n```", "value = 42"),
        ("```python\nvalue = 42", "value = 42"),
        ("```python\r\nvalue = 42\r\n```", "value = 42"),
        (
            "def answer():\n    return 42\n```\nExplanation",
            "def answer():\n    return 42",
        ),
        ("```python\nvalue = 42\n```\n```python\nvalue = 0\n```", "value = 42"),
        ("", ""),
        ("```", ""),
    ],
)
def test_extract_code_preserves_prefilled_completion_tokens(mbpp, response, expected):
    assert mbpp.extract_code_blocks(response) == expected


def test_build_predictions_preserves_document_and_candidate_nesting(mbpp):
    responses = [
        ["def answer():\n    return 42\n```", "```python\nvalue = 7\n```"],
        ["import math\n```"],
    ]
    assert mbpp.build_predictions(responses, [{}, {}]) == [
        ["def answer():\n    return 42", "value = 7"],
        ["import math"],
    ]
