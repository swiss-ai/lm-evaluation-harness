"""Offline AlpacaEval adapter checks; only external scoring is replaced."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def metric():
    return importlib.import_module("lm_eval.tasks.alpaca_eval.metric")


@pytest.fixture
def doc():
    return {
        "instruction": "Name a Swiss city.",
        "output": "Bern.",
        "generator": "gpt4_1106_preview",
        "dataset": "helpful_base",
    }


def test_process_preserves_reference_and_raw_completion(metric, doc):
    result = metric.alpaca_eval_process(
        doc, ["<think>Consider cities.</think>\nZürich is one. "]
    )
    assert result == {
        "length_controlled_winrate": {
            "instruction": "Name a Swiss city.",
            "completion": "Zürich is one.",
            "raw_completion": "<think>Consider cities.</think>\nZürich is one. ",
            "reference_output": "Bern.",
            "reference_generator": "gpt4_1106_preview",
            "dataset": "helpful_base",
        },
        "avg_word_count": 3,
    }


@pytest.mark.parametrize(
    ("completion", "expected", "words"),
    [
        ("", "", 0),
        ("\n  ", "", 0),
        ("<think>Only reasoning.</think>", "", 0),
        ("Opened in prompt.</think> Final answer.", "Final answer.", 2),
        ("<think>One.</think>A <think>Two.</think>B", "A B", 2),
    ],
)
def test_process_handles_empty_and_thinking_outputs(
    metric, doc, completion, expected, words
):
    result = metric.alpaca_eval_process(doc, [completion])
    assert result["length_controlled_winrate"]["completion"] == expected
    assert result["length_controlled_winrate"]["raw_completion"] == completion
    assert result["avg_word_count"] == words


def test_process_rejects_missing_completion(metric, doc):
    with pytest.raises(ValueError, match="completion"):
        metric.alpaca_eval_process(doc, [])


def test_module_load_does_not_require_optional_scorer(metric):
    result = subprocess.run(  # noqa: S603 - fixed Python executable and local task path
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])",
            metric.__file__,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_aggregation_rejects_empty_evaluation(metric):
    with pytest.raises(ValueError, match="empty"):
        metric.alpaca_eval_agg([])


def test_aggregation_requires_explicit_judge(metric, doc, monkeypatch):
    monkeypatch.delenv("ALPACA_EVAL_ANNOTATORS_CONFIG", raising=False)
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    with pytest.raises(ValueError, match="ALPACA_EVAL_ANNOTATORS_CONFIG"):
        metric.alpaca_eval_agg([item])


@pytest.fixture
def scorer(monkeypatch, tmp_path):
    # The real package resolves the annotator configuration. No judge API runs.
    if importlib.util.find_spec("alpaca_eval") is None:
        pytest.skip("Install lm_eval[alpaca_eval] for the optional scorer checks.")
    import alpaca_eval
    import pandas as pd

    calls = []
    response = {
        "leaderboard": pd.DataFrame(
            {"length_controlled_winrate": [37.5], "win_rate": [40.0]}, index=["model"]
        )
    }

    def evaluate(**kwargs):
        calls.append(kwargs)
        return response["leaderboard"], []

    monkeypatch.setattr(alpaca_eval, "evaluate", evaluate)
    monkeypatch.setenv(
        "ALPACA_EVAL_ANNOTATORS_CONFIG", "weighted_alpaca_eval_gpt4_turbo"
    )
    monkeypatch.setenv("ALPACA_EVAL_OUTPUT_DIR", str(tmp_path / "results"))
    return calls, response, tmp_path / "results"


def test_aggregation_prepares_records_and_persists_provenance(
    metric, doc, scorer, monkeypatch
):
    calls, _, output_root = scorer
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-untouched")
    monkeypatch.setenv("OPENAI_API_BASE", "https://example.invalid/v1")
    item = metric.alpaca_eval_process(doc, ["<think>A thought.</think>Zürich."])[
        "length_controlled_winrate"
    ]
    assert metric.alpaca_eval_agg([item]) == 0.375

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["model_outputs"] == [
        {
            "instruction": "Name a Swiss city.",
            "output": "Zürich.",
            "generator": "model",
            "dataset": "helpful_base",
        }
    ]
    assert kwargs["reference_outputs"] == [
        {
            "instruction": "Name a Swiss city.",
            "output": "Bern.",
            "generator": "gpt4_1106_preview",
            "dataset": "helpful_base",
        }
    ]
    assert kwargs["name"] == "model"
    assert kwargs["fn_metric"] == "get_length_controlled_winrate"
    assert kwargs["sort_by"] == "length_controlled_winrate"
    assert kwargs["precomputed_leaderboard"] is None
    assert kwargs["is_cache_leaderboard"] is False
    assert kwargs["is_return_instead_of_print"] is True
    assert Path(kwargs["annotators_config"]).is_file()
    run_dir = Path(kwargs["output_path"])
    assert run_dir.parent == output_root
    assert kwargs["metric_kwargs"] == {"save_weights_dir": str(run_dir / "glm_weights")}
    assert kwargs["caching_path"] == str(run_dir / "annotation_cache.json")

    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["status"] == "completed"
    assert provenance["annotators_config"] == "weighted_alpaca_eval_gpt4_turbo"
    assert provenance["annotators"] == [
        {
            "name": "weighted_alpaca_eval_gpt4_turbo",
            "model_name": "gpt-4-1106-preview",
            "fn_completions": "openai_completions",
        }
    ]
    assert len(provenance["annotators_config_sha256"]) == 64
    assert provenance["alpaca_eval_version"]
    assert provenance["scorer_score_percent"] == 37.5
    assert provenance["harness_score_fraction"] == 0.375
    assert provenance["reference_generators"] == ["gpt4_1106_preview"]
    assert (
        json.loads((run_dir / "collected_outputs.json").read_text())[0][
            "raw_completion"
        ]
        == "<think>A thought.</think>Zürich."
    )
    assert (
        json.loads((run_dir / "model_outputs.json").read_text())
        == kwargs["model_outputs"]
    )
    assert (
        json.loads((run_dir / "reference_outputs.json").read_text())
        == kwargs["reference_outputs"]
    )
    assert "test-key-untouched" not in (run_dir / "provenance.json").read_text()
    import os

    assert os.environ["OPENAI_API_KEY"] == "test-key-untouched"
    assert os.environ["OPENAI_API_BASE"] == "https://example.invalid/v1"


def test_custom_judge_config_is_used_without_copying_credentials(
    metric, doc, scorer, monkeypatch, tmp_path
):
    calls, _, _ = scorer
    config = tmp_path / "custom_judge.yaml"
    config.write_text("""custom_judge:
  prompt_template: 'Instruction: {instruction} A: {output_1} B: {output_2}'
  fn_completions: openai_completions
  completions_kwargs:
    model_name: local-judge
    client_kwargs:
      api_key: secret-fixture-key
""")
    monkeypatch.setenv("ALPACA_EVAL_ANNOTATORS_CONFIG", str(config))
    item = metric.alpaca_eval_process(doc, [""])["length_controlled_winrate"]
    assert metric.alpaca_eval_agg([item]) == 0.375
    assert calls[0]["annotators_config"] == str(config)
    assert calls[0]["model_outputs"][0]["output"] == ""
    provenance = (Path(calls[0]["output_path"]) / "provenance.json").read_text()
    assert "local-judge" in provenance
    assert "secret-fixture-key" not in provenance


@pytest.mark.parametrize("config_text", ["{}", "[]", "judge: invalid", "judge: {}"])
def test_invalid_judge_config_fails_before_scoring(
    metric, doc, scorer, monkeypatch, tmp_path, config_text
):
    calls, _, _ = scorer
    config = tmp_path / "invalid.yaml"
    config.write_text(config_text)
    monkeypatch.setenv("ALPACA_EVAL_ANNOTATORS_CONFIG", str(config))
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    with pytest.raises(ValueError, match="annotator"):
        metric.alpaca_eval_agg([item])
    assert calls == []


def test_missing_judge_config_fails_before_scoring(metric, doc, scorer, monkeypatch):
    calls, _, _ = scorer
    monkeypatch.setenv("ALPACA_EVAL_ANNOTATORS_CONFIG", "nonexistent-annotator-fixture")
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    with pytest.raises(FileNotFoundError, match="annotator"):
        metric.alpaca_eval_agg([item])
    assert calls == []


@pytest.mark.parametrize(
    "bad_score", [None, float("nan"), float("inf"), -1.0, 101.0, "37.5", True]
)
def test_invalid_score_does_not_become_a_success(metric, doc, scorer, bad_score):
    import pandas as pd

    calls, response, _ = scorer
    response["leaderboard"] = pd.DataFrame(
        {"length_controlled_winrate": [bad_score]}, index=["model"]
    )
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    with pytest.raises(ValueError, match="length_controlled_winrate"):
        metric.alpaca_eval_agg([item])
    provenance = json.loads(
        (Path(calls[0]["output_path"]) / "provenance.json").read_text()
    )
    assert provenance["status"] == "failed"


@pytest.mark.parametrize(
    "leaderboard_kind", ["none", "empty", "missing_metric", "missing_model"]
)
def test_missing_score_is_an_explicit_failure(metric, doc, scorer, leaderboard_kind):
    import pandas as pd

    _, response, _ = scorer
    response["leaderboard"] = {
        "none": None,
        "empty": pd.DataFrame(columns=["length_controlled_winrate"]),
        "missing_metric": pd.DataFrame({"win_rate": [40.0]}, index=["model"]),
        "missing_model": pd.DataFrame(
            {"length_controlled_winrate": [90.0]}, index=["baseline"]
        ),
    }[leaderboard_kind]
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    with pytest.raises(ValueError, match="length_controlled_winrate"):
        metric.alpaca_eval_agg([item])


def test_model_score_is_selected_by_name(metric, doc, scorer):
    import pandas as pd

    _, response, _ = scorer
    response["leaderboard"] = pd.DataFrame(
        {"length_controlled_winrate": [90.0, 37.5]}, index=["baseline", "model"]
    )
    item = metric.alpaca_eval_process(doc, ["Zürich."])["length_controlled_winrate"]
    assert metric.alpaca_eval_agg([item]) == 0.375
