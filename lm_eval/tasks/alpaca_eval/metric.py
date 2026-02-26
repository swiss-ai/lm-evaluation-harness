"""AlpacaEval metric for lm-evaluation-harness.

Implements length-controlled winrate evaluation using the alpaca_eval library
with Llama-3.3-70B-Instruct as pairwise judge, served via the CSCS SwissAI API.

The evaluation follows the AlpacaEval 2 protocol:
  1. Model outputs are compared pairwise against GPT-4 baseline responses.
  2. A judge LLM produces a single token (m or M) indicating which output is
     better. Logprob-based scoring yields a continuous preference signal:
     P(m) / (P(m) + P(M)), where m = reference wins, M = model wins.
  3. Length-controlled winrate applies a GLM to debias for output length
     differences (Dubois et al., 2024).

Prerequisites:
    alpaca-eval
    CSCS_SERVING_API (with access to Llama-3.3-70B-Instruct)
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# CSCS SwissAI serving endpoint
API_URL = "https://api.swissai.cscs.ch/v1"

# Judge model served at the CSCS endpoint
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Name for the annotator config (written into alpaca_eval's evaluators_configs/)
ANNOTATOR_NAME = "cscs_llama3_70b"

# Annotator configuration matching the weighted AlpacaEval protocol:
# logprob-based classification with a single-token output (m or M).
ANNOTATOR_CONFIG = {
    ANNOTATOR_NAME: {
        "prompt_template": "alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt",
        "fn_completions": "openai_completions",
        "completions_kwargs": {
            "model_name": JUDGE_MODEL,
            "max_tokens": 1,
            "temperature": 1,
            "logprobs": True,
            "top_logprobs": 5,
            "requires_chatml": True,
        },
        "fn_completion_parser": "logprob_parser",
        "completion_parser_kwargs": {
            "numerator_token": "m",
            "denominator_tokens": ["m", "M"],
            "is_binarize": False,
        },
        "completion_key": "completions_all",
        "batch_size": 1,
    }
}


def _ensure_annotator_config():
    """Write the annotator config into alpaca_eval's evaluators_configs directory.

    The config is placed inside alpaca_eval's own evaluators_configs/ so that
    the prompt_template relative path (alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt)
    resolves correctly against the bundled prompt templates.

    This follows the same pattern used by run_alpaca_eval.sh.

    Returns:
        str: The annotator config name to pass to alpaca_eval.evaluate().
    """
    from alpaca_eval import constants as ae_constants

    config_dir = ae_constants.EVALUATORS_CONFIG_DIR / ANNOTATOR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "configs.yaml"

    with open(config_file, "w") as f:
        yaml.dump(ANNOTATOR_CONFIG, f, default_flow_style=False)

    logger.info(f"Annotator config written to: {config_file}")
    return ANNOTATOR_NAME


def _strip_thinking_traces(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output.

    Handles two cases:
      1. Both <think> and </think> present in the output.
      2. Only </think> present (because <think> was in the prompt,
         e.g. for OLMo-3-32B-Think or Qwen3 thinking models).
    """
    # Case 1: full <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Case 2: leading content up to </think> (tag was opened in the prompt)
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def alpaca_eval_process(doc, predictions, **kwargs):
    """Per-document result collector. Defers judge calls to aggregation.

    Called once per document by lm-evaluation-harness during the results
    processing phase. Packages the instruction, model completion, and GPT-4
    reference output for later batch annotation by alpaca_eval.

    Thinking traces (<think>...</think>) are stripped from model completions
    before they are passed to the judge, matching the behaviour of
    generate_alpaca_completions.py.

    Args:
        doc: Dataset row with keys 'instruction', 'output' (GPT-4 reference),
            'generator', and 'dataset'.
        predictions: List of model-generated strings; predictions[0] is used.

    Returns:
        Dict mapping the metric name to collected data for aggregation.
    """
    raw_output = predictions[0]
    clean_output = _strip_thinking_traces(raw_output)

    if clean_output != raw_output:
        logger.debug(
            "Stripped thinking traces from output "
            f"(raw={len(raw_output)} chars → clean={len(clean_output)} chars)"
        )

    return {
        "length_controlled_winrate": {
            "instruction": doc["instruction"],
            "completion": clean_output,
            "reference_output": doc["output"],
            "dataset": doc.get("dataset", "alpaca_eval"),
        }
    }


def alpaca_eval_agg(items):
    """Aggregate all items by running alpaca_eval annotation and returning winrate.

    Called once after all documents have been processed. Configures the CSCS
    serving endpoint as the OpenAI-compatible backend, writes the annotator
    config, builds model and reference output lists, then calls
    alpaca_eval.evaluate() to obtain the length-controlled winrate.

    The annotator uses Llama-3.3-70B-Instruct with logprob-based classification:
    for each (instruction, reference_output, model_output) triple, the judge
    produces logprobs for tokens 'm' (reference wins) and 'M' (model wins).
    The continuous preference score P(m) / (P(m) + P(M)) is then debiased for
    output length via a generalized linear model to produce the length-controlled
    winrate.

    Args:
        items: List of dicts collected from alpaca_eval_process(), each with
            keys 'instruction', 'completion', 'reference_output', 'dataset'.

    Returns:
        float: Length-controlled winrate (0-100 scale).

    Raises:
        EnvironmentError: If CSCS_SERVING_API environment variable is not set.
        ImportError: If the alpaca_eval package is not installed.
    """
    api_key = os.getenv("CSCS_SERVING_API")
    if not api_key:
        raise EnvironmentError(
            "CSCS_SERVING_API environment variable must be set for AlpacaEval "
            "annotation. Get your key from the CSCS SwissAI platform."
        )

    # Route alpaca_eval's OpenAI client to the CSCS serving endpoint
    os.environ["OPENAI_API_BASE"] = API_URL
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["IS_ALPACA_EVAL_2"] = "True"

    # Write annotator config into alpaca_eval's config directory
    annotator_name = _ensure_annotator_config()

    # Build model outputs in alpaca_eval's expected format
    model_outputs = [
        {
            "instruction": item["instruction"],
            "output": item["completion"],
            "generator": "model",
            "dataset": item["dataset"],
        }
        for item in items
    ]

    # Build reference outputs (GPT-4 baseline)
    reference_outputs = [
        {
            "instruction": item["instruction"],
            "output": item["reference_output"],
            "generator": "gpt4",
            "dataset": item["dataset"],
        }
        for item in items
    ]

    # ── Output length statistics ──────────────────────────────────────
    output_lengths = [len(m["output"]) for m in model_outputs]
    logger.info(
        f"Output length stats (chars): "
        f"min={min(output_lengths)}, max={max(output_lengths)}, "
        f"mean={sum(output_lengths)/len(output_lengths):.0f}, "
        f"median={sorted(output_lengths)[len(output_lengths)//2]}"
    )

    logger.info(
        f"Running AlpacaEval annotation on {len(model_outputs)} examples "
        f"using {JUDGE_MODEL} as judge via CSCS serving API..."
    )

    from alpaca_eval import evaluate

    with tempfile.TemporaryDirectory() as tmpdir:
        df_leaderboard, annotations = evaluate(
            model_outputs=model_outputs,
            reference_outputs=reference_outputs,
            annotators_config=annotator_name,
            is_return_instead_of_print=True,
            fn_metric="get_length_controlled_winrate",
            output_path=tmpdir,
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            caching_path=os.path.join(tmpdir, "cache.json"),
        )

    # ── Log the full leaderboard (includes SE computed by alpaca_eval) ─
    logger.info(f"AlpacaEval leaderboard columns: {list(df_leaderboard.columns)}")
    logger.info(f"AlpacaEval full leaderboard:\n{df_leaderboard.to_string()}")

    lc_winrate = df_leaderboard["length_controlled_winrate"].iloc[0]
    logger.info(f"AlpacaEval length-controlled winrate: {lc_winrate:.2f}")

    # Log standard error if available in the leaderboard
    se_candidates = [
        c for c in df_leaderboard.columns
        if "standard_error" in c.lower() or "_se" in c.lower() or c == "se"
    ]
    for col in se_candidates:
        val = df_leaderboard[col].iloc[0]
        logger.info(f"  {col}: {val}")

    return float(lc_winrate)
