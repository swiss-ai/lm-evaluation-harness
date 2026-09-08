"""Portable, optional AlpacaEval scoring with explicit annotator configuration."""

import hashlib
import json
import logging
import math
import os
import re
import tempfile
from importlib.metadata import version
from numbers import Real
from pathlib import Path


logger = logging.getLogger(__name__)

REFERENCE_DATASET = "hf://datasets/tatsu-lab/alpaca_eval/alpaca_eval_gpt4_baseline.json"
MODEL_NAME = "model"


def _strip_thinking_traces(text):
    """Preserve the Swiss task's full-block and prompt-opened thinking cleanup."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL).strip()


def alpaca_eval_process(doc, results, **kwargs):
    """Collect generated and reference outputs without importing a judge client."""
    if not results:
        raise ValueError(
            "AlpacaEval requires a generated completion; an empty string is allowed."
        )
    raw_output = results[0]
    if not isinstance(raw_output, str):
        raise TypeError("AlpacaEval completion must be a string.")
    clean_output = _strip_thinking_traces(raw_output)
    return {
        "length_controlled_winrate": {
            "instruction": doc["instruction"],
            "completion": clean_output,
            "raw_completion": raw_output,
            "reference_output": doc["output"],
            "reference_generator": doc.get("generator", "gpt4"),
            "dataset": doc.get("dataset", "alpaca_eval"),
        },
        "avg_word_count": len(clean_output.split()),
    }


def _annotator_config(config_name):
    """Resolve a local or bundled config using the official scorer's loader."""
    from alpaca_eval import constants, utils

    config_path = Path(config_name).expanduser()
    if not config_path.exists():
        config_path = constants.EVALUATORS_CONFIG_DIR / config_name
    if config_path.is_dir():
        config_path /= "configs.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"AlpacaEval annotator configuration not found: {config_name}"
        )
    config_path = config_path.resolve()
    try:
        config = utils.load_configs(config_path)
    except (AssertionError, TypeError) as exc:
        raise ValueError(
            "AlpacaEval annotator configuration must be a nonempty mapping."
        ) from exc
    if not isinstance(config, dict) or not config:
        raise ValueError(
            "AlpacaEval annotator configuration must be a nonempty mapping."
        )
    for name, settings in config.items():
        if (
            not isinstance(name, str)
            or not isinstance(settings, dict)
            or not settings.get("prompt_template")
            or not settings.get("fn_completions")
        ):
            raise ValueError(
                "Each AlpacaEval annotator requires prompt_template and fn_completions."
            )
    return config_path, config


def _write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _score_from_leaderboard(leaderboard):
    """Extract this model's finite percentage, rejecting missing or invalid scores."""
    import pandas as pd

    metric = "length_controlled_winrate"
    if (
        not isinstance(leaderboard, pd.DataFrame)
        or metric not in leaderboard.columns
        or MODEL_NAME not in leaderboard.index
    ):
        raise ValueError(f"AlpacaEval did not return {metric} for {MODEL_NAME}.")
    score = leaderboard.loc[MODEL_NAME, metric]
    if (
        not isinstance(score, Real)
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 100
    ):
        raise ValueError(
            f"AlpacaEval returned invalid {metric}; expected a finite number in [0, 100]."
        )
    return float(score)


def alpaca_eval_agg(items):
    """Run the official scorer and return length-controlled win rate as a fraction.

    ``ALPACA_EVAL_ANNOTATORS_CONFIG`` must name an official bundled annotator or
    a custom config file/directory. Credentials and endpoints use that scorer's
    own configuration. Run artifacts persist under ``ALPACA_EVAL_OUTPUT_DIR``
    (default: ``./alpaca_eval_results``), including judge/scorer provenance.
    """
    if not items:
        raise ValueError("Cannot score an empty AlpacaEval evaluation.")
    config_name = os.environ.get("ALPACA_EVAL_ANNOTATORS_CONFIG", "").strip()
    if not config_name:
        raise ValueError(
            "Set ALPACA_EVAL_ANNOTATORS_CONFIG to an explicit AlpacaEval annotator "
            "name or config path. The judge determines the evaluation protocol."
        )
    try:
        from alpaca_eval import evaluate
    except ImportError as exc:
        raise ImportError(
            "AlpacaEval scoring requires the optional dependency: pip install 'lm_eval[alpaca_eval]'."
        ) from exc

    config_path, config = _annotator_config(config_name)
    model_outputs = [
        {
            "instruction": item["instruction"],
            "output": item["completion"],
            "generator": MODEL_NAME,
            "dataset": item["dataset"],
        }
        for item in items
    ]
    reference_outputs = [
        {
            "instruction": item["instruction"],
            "output": item["reference_output"],
            "generator": item["reference_generator"],
            "dataset": item["dataset"],
        }
        for item in items
    ]

    output_root = (
        Path(os.environ.get("ALPACA_EVAL_OUTPUT_DIR", "alpaca_eval_results"))
        .expanduser()
        .resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    provenance = {
        "status": "started",
        "alpaca_eval_version": version("alpaca-eval"),
        "annotators_config": config_name,
        "annotators_config_path": str(config_path),
        "annotators_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        # Deliberately omit credentials and decoder/client kwargs from provenance.
        "annotators": [
            {
                "name": name,
                "model_name": settings.get("completions_kwargs", {}).get("model_name"),
                "fn_completions": settings["fn_completions"],
            }
            for name, settings in config.items()
        ],
        "reference_dataset": REFERENCE_DATASET,
        "reference_generators": sorted({item["reference_generator"] for item in items}),
        "sample_count": len(items),
        "output_preprocessing": "strip_think_blocks_and_prompt_opened_thinking_then_whitespace",
        "metric": "get_length_controlled_winrate",
        "scorer_score_unit": "percent (0-100)",
        "harness_score_unit": "fraction (0-1)",
    }
    _write_json(run_dir / "collected_outputs.json", items)
    _write_json(run_dir / "model_outputs.json", model_outputs)
    _write_json(run_dir / "reference_outputs.json", reference_outputs)
    _write_json(run_dir / "provenance.json", provenance)
    logger.info(
        "AlpacaEval %s; annotator config %s; %s examples; artifacts: %s",
        provenance["alpaca_eval_version"],
        config_name,
        len(items),
        run_dir,
    )

    try:
        leaderboard, _annotations = evaluate(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=str(config_path),
            name=MODEL_NAME,
            is_return_instead_of_print=True,
            fn_metric="get_length_controlled_winrate",
            sort_by="length_controlled_winrate",
            # Otherwise the scorer writes GLM weights into its installed package.
            metric_kwargs={"save_weights_dir": str(run_dir / "glm_weights")},
            output_path=str(run_dir),
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            caching_path=str(run_dir / "annotation_cache.json"),
        )
        score = _score_from_leaderboard(leaderboard)
    except Exception as exc:
        provenance.update(status="failed", error_type=type(exc).__name__)
        _write_json(run_dir / "provenance.json", provenance)
        raise
    provenance.update(
        status="completed",
        scorer_score_percent=score,
        harness_score_fraction=score / 100.0,
    )
    _write_json(run_dir / "provenance.json", provenance)
    logger.info(
        "AlpacaEval length-controlled win rate: %.4f%% (harness: %.6f)",
        score,
        score / 100.0,
    )
    return score / 100.0
