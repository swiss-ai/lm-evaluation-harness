"""Arena-Hard v2.0 metric for lm-evaluation-harness.

Implements pairwise judge evaluation using Qwen3.5—27B as judge,
served via the CSCS SwissAI API.

The evaluation follows the Arena-Hard protocol:
  1. Model outputs are compared pairwise against o3-mini-2025-01-31 baseline responses.
  2. A judge LLM evaluates which response is better and outputs a verdict
     in the format [[A>>B]], [[A>B]], [[A=B]], [[B>A]], or [[B>>A]].
  3. Two rounds are performed per question with positions swapped to mitigate
     position bias.
  4. Strong preferences (>> or <<) receive 3x weight in scoring.
  5. Win rate is computed via bootstrap resampling (100 rounds) with 90% CI.

Reference: https://github.com/lm-sys/arena-hard-auto

Prerequisites:
    openai
    huggingface_hub
    numpy
    CSCS_SERVING_API environment variable (with access to Qwen3.5—27B judge endpoint. 
                                Make sure this model is served on the CSCS platform.)
"""

import concurrent.futures
import json
import logging
import os
import re
import time

import numpy as np


logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# CSCS SwissAI serving endpoint
API_URL = "https://api.swissai.svc.cscs.ch/v1"
JUDGE_MODEL = "Qwen/Qwen3.5-27B"

# Restrict evaluation to a single Arena-Hard v2.0 category. The released
# dataset bundles 500 "hard_prompt" + 250 "creative_writing" items; we only
# evaluate the hard_prompt subset (questions and baseline answers alike).
CATEGORY_FILTER = "hard_prompt"

# HuggingFace coordinates for the Arena-Hard v2.0 question file. Kept in
# sync with arena_hard_v2.yaml so the baseline loader can re-derive the
# uid -> category map without depending on lm-eval's filtered dataset.
QUESTION_REPO_ID = "lmarena-ai/arena-hard-auto"
QUESTION_FILE = "data/arena-hard-v2.0/question.jsonl"

# Number of concurrent threads hitting the judge API.
# vLLM with continuous batching on GH200s can handle high concurrency.
_JUDGE_MAX_WORKERS = 32

# Number of retries for the OpenAI client (covers transient HTTP errors).
_CLIENT_MAX_RETRIES = 8
_TIMEOUT_SECONDS = 800
# Bootstrap resampling parameters
NUM_BOOTSTRAP = 100

# Environment variables for optional logging
JUDGE_LOG_PATH_ENV = "ARENA_HARD_JUDGE_LOG_PATH"
MAX_GEN_TOKS_ENV = "ARENA_HARD_MAX_GEN_TOKS"

# ── Scoring method configuration ─────────────────────────────────
# "weighted"      = original Arena-Hard weighted average + bootstrap
# "style_control" = Bradley-Terry model controlling for style features
SCORING_METHOD = os.environ.get("ARENA_HARD_SCORING_METHOD", "weighted")

# Which style features to control for (only used when SCORING_METHOD = "style_control").
# Supported: ["length", "markdown"] (both), ["length"], or ["markdown"].
CONTROL_FEATURES = ["length", "markdown"]

# ── Judge system prompt (from arena-hard-auto/utils/judge_utils.py) ──
SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the "
    "responses provided by two AI assistants to the user prompt displayed "
    "below. You will be given assistant A's answer and assistant B's answer. "
    "Your job is to evaluate which assistant's answer is better.\n\n"
    "Begin your evaluation by generating your own answer to the prompt. You "
    "must provide your answers before judging any answers.\n\n"
    "When evaluating the assistants' answers, compare both assistants' "
    "answers with your answer. You must identify and correct any mistakes "
    "or inaccurate information.\n\n"
    "Then consider if the assistant's answers are helpful, relevant, and "
    "concise. Helpful means the answer correctly responds to the prompt or "
    "follows the instructions. Note when user prompt has any ambiguity or "
    "more than one interpretation, it is more helpful and appropriate to ask "
    "for clarifications or more information from the user than providing an "
    "answer based on assumptions. Relevant means all parts of the response "
    "closely connect or are appropriate to what is being asked. Concise "
    "means the response is clear and not verbose or excessive.\n\n"
    "Then consider the creativity and novelty of the assistant's answers "
    "when needed. Finally, identify any missing important information in "
    "the assistants' answers that would be beneficial to include when "
    "responding to the user prompt.\n\n"
    "After providing your explanation, you must output only one of the "
    "following choices as your final verdict with a label:\n\n"
    "1. Assistant A is significantly better: [[A>>B]]\n"
    "2. Assistant A is slightly better: [[A>B]]\n"
    "3. Tie, relatively the same: [[A=B]]\n"
    "4. Assistant B is slightly better: [[B>A]]\n"
    "5. Assistant B is significantly better: [[B>>A]]\n\n"
    'Example output: "My final verdict is tie: [[A=B]]".'
)

# ── Pairwise prompt template (from arena-hard-v2.0) ──────────────────
PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)

# ── Regex patterns for verdict extraction (from arena-hard-v2.0) ─────
REGEX_PATTERNS = [
    r"\[\[([AB<>=]+)\]\]",
    r"\[([AB<>=]+)\]",
]

# ── Score mapping with weight=3 for strong preferences ────────────────
# From arena-hard-auto/show_result.py load_judgments().
# Scores represent win probability from A's perspective:
#   A wins = 1, B wins = 0, tie = 0.5.
WEIGHT = 3
LABEL_TO_SCORE = {
    "A>>B": [1] * WEIGHT,
    "A>B": [1],
    "A=B": [0.5],
    "A<B": [0],
    "A<<B": [0] * WEIGHT,
    "B>>A": [0] * WEIGHT,
    "B>A": [0],
    "B=A": [0.5],
    "B<A": [1],
    "B<<A": [1] * WEIGHT,
}


# ── Dataset filter ───────────────────────────────────────────────────


def filter_hard_prompt(dataset):
    """Restrict the Arena-Hard v2.0 dataset to the hard_prompt category.

    Wired into arena_hard_v2.yaml as `process_docs`. Drops the 250
    creative_writing items so only the 500 hard_prompt items are evaluated.
    """
    return dataset.filter(lambda doc: doc.get("category") == CATEGORY_FILTER)


# ── Baseline answer cache ────────────────────────────────────────────
_BASELINE_CACHE = None
_CATEGORY_UIDS_CACHE = None


def _load_category_uids():
    """Return the set of uids whose question category == CATEGORY_FILTER."""
    global _CATEGORY_UIDS_CACHE
    if _CATEGORY_UIDS_CACHE is not None:
        return _CATEGORY_UIDS_CACHE

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=QUESTION_REPO_ID,
        filename=QUESTION_FILE,
        repo_type="dataset",
    )
    uids = set()
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("category") == CATEGORY_FILTER:
                uids.add(entry["uid"])
    _CATEGORY_UIDS_CACHE = uids
    return uids


def _load_baseline_answers():
    """Load o3-mini-2025-01-31 baseline answers for the hard_prompt subset.

    Checks for a local o3-mini-2025-01-31_baseline.jsonl in the task directory first.
    Falls back to downloading from HuggingFace if not found. The released baseline
    file covers all 750 v2.0 items; entries whose uid is not in the hard_prompt
    category are dropped so the cache only holds answers we will actually use.

    Returns:
        dict: Mapping from question uid to baseline answer text.
    """
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    local_path = os.path.join(os.path.dirname(__file__), "o3-mini-2025-01-31_baseline.jsonl")
    if os.path.exists(local_path):
        path = local_path
        logger.info(f"Using local baseline file: {path}")
    else:
        from huggingface_hub import hf_hub_download

        logger.info("Local baseline not found, downloading from HuggingFace...")
        path = hf_hub_download(
            repo_id="lmarena-ai/arena-hard-auto",
            filename="data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl",
            repo_type="dataset",
        )

    category_uids = _load_category_uids()
    baseline = {}
    skipped = 0
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            uid = entry["uid"]
            if uid not in category_uids:
                skipped += 1
                continue
            answer = entry["messages"][-1]["content"]["answer"]
            baseline[uid] = answer

    logger.info(
        f"Loaded {len(baseline)} o3-mini-2025-01-31 baseline answers "
        f"for category={CATEGORY_FILTER} (skipped {skipped} out-of-category entries)"
    )
    _BASELINE_CACHE = baseline
    return baseline


# ── Text processing utilities ─────────────────────────────────────────


def _get_score(judgment_text, patterns=REGEX_PATTERNS):
    """Extract verdict label from judge output text.

    Ported from arena-hard-auto/gen_judgment.py::get_score().

    Args:
        judgment_text: Full text response from the judge model.
        patterns: List of regex patterns to try in order.

    Returns:
        str or None: Verdict label (e.g. "A>>B", "B>A") or None if not found.
    """
    for pattern in patterns:
        compiled = re.compile(pattern)
        matches = compiled.findall(judgment_text.upper())
        matches = [m for m in matches if m != ""]
        if len(set(matches)) > 0:
            return matches[-1].strip("\n")
    return None


def _strip_thinking_traces(text):
    """Remove <think>...</think> reasoning traces from model output.

    Handles two cases:
      1. Both <think> and </think> present in the output.
      2. Only </think> present (because <think> was in the prompt,
         e.g. for OLMo-3-32B-Think or Qwen3 thinking models).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ── Per-document processing ──────────────────────────────────────────


def arena_hard_process(doc, predictions, **kwargs):
    """Per-document result collector.  Defers judge calls to aggregation.

    Called once per document by lm-evaluation-harness during the results
    processing phase.  Packages the question, model completion, and o3-mini-2025-01-31
    baseline output for later batch annotation.

    Thinking traces (<think>...</think>) are stripped from model completions
    before they are passed to the judge.

    Args:
        doc: Dataset row with keys 'uid', 'prompt', 'category', 'cluster'.
        predictions: List of model-generated strings; predictions[0] is used.

    Returns:
        Dict mapping the metric name to collected data for aggregation.
    """
    baseline_answers = _load_baseline_answers()

    raw_output = predictions[0]
    clean_output = _strip_thinking_traces(raw_output)

    if clean_output != raw_output:
        logger.debug(
            "Stripped thinking traces from output "
            f"(raw={len(raw_output)} chars -> clean={len(clean_output)} chars)"
        )

    uid = doc["uid"]
    baseline_output = baseline_answers.get(uid)
    if baseline_output is None:
        logger.warning(f"No baseline answer found for uid={uid}")

    word_count = len(clean_output.split())

    return {
        "arena_hard_score": {
            "uid": uid,
            "prompt": doc["prompt"],
            "model_output": clean_output,
            "raw_output": raw_output,
            "baseline_output": baseline_output,
            "category": doc.get("category", "arena-hard-v2.0"),
        },
        "avg_word_count": word_count,
    }


# ── Judge infrastructure ─────────────────────────────────────────────


def _judge_call(client, uid, round_num, user_prompt):
    """Execute a single judge API call. Designed to run inside a thread pool.

    Transient HTTP errors (429, 5xx, connection resets) are retried
    automatically by the OpenAI client (configured with max_retries).
    The try/except here catches truly unrecoverable errors so that one
    failed request does not crash the entire batch.

    Args:
        client: An openai.OpenAI client instance (thread-safe for I/O calls).
        uid: Unique identifier for the question being judged.
        round_num: 1 or 2, indicating the position-swap round.
        user_prompt: The fully formatted pairwise comparison prompt string.

    Returns:
        Tuple of (uid, round_num, judgment_text_or_None).
    """
    prompt_chars = len(user_prompt)
    logger.info(
        f"Judge call START uid={uid} round={round_num} "
        f"prompt_chars={prompt_chars}"
    )
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=6000,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.time() - t0
        content = resp.choices[0].message.content
        logger.info(
            f"Judge call OK uid={uid} round={round_num} "
            f"elapsed={elapsed:.1f}s response_chars={len(content or '')}"
        )
        return (uid, round_num, content)
    except Exception as e:
        elapsed = time.time() - t0
        # Classify the error for easier log analysis
        err_str = str(e)
        if "504" in err_str:
            err_type = "504_GATEWAY_TIMEOUT"
        elif "502" in err_str:
            err_type = "502_BAD_GATEWAY"
        elif "429" in err_str:
            err_type = "429_RATE_LIMITED"
        elif "timeout" in err_str.lower() or "timed out" in err_str.lower():
            err_type = "CLIENT_TIMEOUT"
        else:
            err_type = type(e).__name__
        logger.warning(
            f"Judge call FAILED uid={uid} round={round_num} "
            f"error_type={err_type} elapsed={elapsed:.1f}s "
            f"prompt_chars={prompt_chars} "
            f"(retries were exhausted after {_CLIENT_MAX_RETRIES} attempts): "
            f"{err_str[:300]}"
        )
        return (uid, round_num, None)


def _build_judge_tasks(valid_items):
    """Build the flat list of all 2*N judge tasks.

    Each task is a tuple: (uid, round_number, formatted_user_prompt).
    Round 1: baseline is position A, model is position B.
    Round 2: model is position A, baseline is position B (swap).
    Both rounds for the same item are independent.

    Args:
        valid_items: List of item dicts with 'uid', 'prompt',
            'model_output', 'baseline_output'.

    Returns:
        List of (uid, round_num, prompt_str) tuples.
    """
    tasks = []
    for item in valid_items:
        uid = item["uid"]

        # Round 1: baseline=A, model=B
        prompt_r1 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["baseline_output"],
            ANSWER_B=item["model_output"],
        )
        tasks.append((uid, 1, prompt_r1))

        # Round 2: model=A, baseline=B (position swap)
        prompt_r2 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["model_output"],
            ANSWER_B=item["baseline_output"],
        )
        tasks.append((uid, 2, prompt_r2))

    return tasks


def _run_judge_calls(client, tasks):
    """Fire all judge calls concurrently and collect results.

    Uses ThreadPoolExecutor because the workload is I/O-bound (waiting on
    HTTP responses from the vLLM server). Threads release the GIL during
    I/O, so Python threads are ideal here.

    Args:
        client: An openai.OpenAI client instance.
        tasks: List of (uid, round_num, prompt_str) tuples from
            _build_judge_tasks().

    Returns:
        dict: Mapping uid -> {round_num: judgment_text_or_None}.
    """
    total_calls = len(tasks)
    results_map = {}
    completed = 0
    succeeded = 0
    failed = 0
    judging_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_JUDGE_MAX_WORKERS
    ) as executor:
        future_to_task = {
            executor.submit(_judge_call, client, uid, rnd, prompt): (uid, rnd)
            for uid, rnd, prompt in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            uid, rnd, judgment = future.result()
            results_map.setdefault(uid, {})[rnd] = judgment
            completed += 1
            if judgment is not None:
                succeeded += 1
            else:
                failed += 1

            if completed % 50 == 0 or completed == total_calls:
                elapsed = time.time() - judging_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Judge progress: {completed}/{total_calls} calls done "
                    f"({succeeded} ok, {failed} failed, "
                    f"{elapsed:.0f}s elapsed, {rate:.1f} calls/s)"
                )

    total_judging_time = time.time() - judging_start
    logger.info(
        f"All {total_calls} judge calls completed in {total_judging_time:.0f}s "
        f"({total_calls / total_judging_time:.1f} calls/s). "
        f"Succeeded: {succeeded}, Failed: {failed}"
    )

    if failed > 0:
        failed_uids = [
            f"uid={uid} round={rnd}"
            for uid, rounds in results_map.items()
            for rnd, judgment in rounds.items()
            if judgment is None
        ]
        logger.warning(
            f"{failed} judge calls failed permanently. "
            f"Failed calls: {', '.join(failed_uids)}"
        )

    return results_map


# ── Scoring ──────────────────────────────────────────────────────────
#
# The scoring step converts raw judge verdicts into numeric scores.
# This is the extension point for alternative scoring methods (e.g.,
# style-controlled scoring with a Bradley-Terry model).
#
# Any scoring function must conform to:
#   (valid_items, results_map) -> (scores: list[float], null_count: int)


def _compute_scores_weighted(valid_items, results_map):
    """Convert judge verdicts to numeric scores using weighted preferences.

    This is the default Arena-Hard scoring method from
    arena-hard-auto/show_result.py:
      - Round 2 (model=A): label_to_score used directly.
      - Round 1 (baseline=A): label_to_score flipped (1 - s) so that
        higher score always = better for the model under test.
      - Strong preferences (>> or <<) receive 3x weight.

    Args:
        valid_items: List of item dicts with 'uid' key.
        results_map: Dict mapping uid -> {round_num: judgment_text}.

    Returns:
        Tuple of (scores, null_count) where scores is a list of floats
        and null_count is the number of items with failed/invalid judgments.
    """
    all_scores = []
    null_judgments = 0

    for item in valid_items:
        uid = item["uid"]
        judgments = results_map.get(uid, {})

        judgment1 = judgments.get(1)
        judgment2 = judgments.get(2)

        score1 = _get_score(judgment1) if judgment1 else None
        score2 = _get_score(judgment2) if judgment2 else None

        logger.debug(f"  uid={uid}: round1={score1}, round2={score2}")

        if score1 is None or score2 is None:
            null_judgments += 1
            logger.warning(
                f"Null judgment for uid={uid}: round1={score1}, round2={score2}"
            )
            continue

        r1_scores = LABEL_TO_SCORE.get(score1)
        r2_scores = LABEL_TO_SCORE.get(score2)

        if r1_scores is None or r2_scores is None:
            null_judgments += 1
            logger.warning(
                f"Unknown score label for uid={uid}: "
                f"round1={score1}, round2={score2}"
            )
            continue

        # Round 2 (model=A): use directly.
        # Round 1 (baseline=A): flip so higher = better for model.
        item_scores = r2_scores + [1 - s for s in r1_scores]
        all_scores.extend(item_scores)

    return all_scores, null_judgments


def _compute_scores_style_controlled(valid_items, results_map):
    """Compute style-controlled win rate using a Bradley-Terry model.

    Instead of averaging raw scores, fits a logistic regression that
    separates style effects (length, markdown formatting) from quality.
    Uses bootstrap resampling for confidence intervals.

    Args:
        valid_items: List of item dicts with 'uid', 'model_output',
            'baseline_output' keys.
        results_map: Dict mapping uid -> {round_num: judgment_text}.

    Returns:
        Tuple of (win_rate, ci_lower, ci_upper, null_count) where
        win_rate/ci values are on 0-1 scale, or (None, None, None, N)
        if no valid judgments.
    """
    from lm_eval.tasks.arena_hard_v2.style_utils import (
        bootstrap_style_controlled_winrate,
        compute_relative_features,
        extract_style_features,
        zscore_normalize,
    )

    all_outcomes = []
    all_model_feats = []
    all_baseline_feats = []
    null_judgments = 0

    for item in valid_items:
        uid = item["uid"]
        judgments = results_map.get(uid, {})

        judgment1 = judgments.get(1)
        judgment2 = judgments.get(2)

        score1 = _get_score(judgment1) if judgment1 else None
        score2 = _get_score(judgment2) if judgment2 else None

        if score1 is None or score2 is None:
            null_judgments += 1
            continue

        r1_scores = LABEL_TO_SCORE.get(score1)
        r2_scores = LABEL_TO_SCORE.get(score2)

        if r1_scores is None or r2_scores is None:
            null_judgments += 1
            continue

        # Same scoring as weighted: round2 direct + round1 flipped
        item_scores = r2_scores + [1 - s for s in r1_scores]

        # Extract style features once per item, replicate for each score entry
        m_feats = extract_style_features(item["model_output"])
        b_feats = extract_style_features(item["baseline_output"])

        for s in item_scores:
            all_outcomes.append(s)
            all_model_feats.append(m_feats)
            all_baseline_feats.append(b_feats)

    if not all_outcomes:
        return None, None, None, null_judgments

    import torch

    # Convert to torch tensors (style_utils uses PyTorch internally,
    # matching the reference arena-hard-auto implementation)
    outcomes = torch.tensor(all_outcomes, dtype=torch.float32)
    model_features = torch.tensor(
        np.vstack(all_model_feats), dtype=torch.float32
    )
    baseline_features = torch.tensor(
        np.vstack(all_baseline_feats), dtype=torch.float32
    )

    # Normalize: relative features → z-score
    relative = compute_relative_features(model_features, baseline_features)
    normalized = zscore_normalize(relative)

    # Select feature columns based on CONTROL_FEATURES
    # (show_result.py lines 190-199)
    if "length" in CONTROL_FEATURES and "markdown" in CONTROL_FEATURES:
        style_matrix = normalized  # all 4 columns
        num_style = 4
    elif "length" in CONTROL_FEATURES:
        style_matrix = normalized[:, :1]  # just relative length
        num_style = 1
    elif "markdown" in CONTROL_FEATURES:
        style_matrix = normalized[:, 1:]  # 3 markdown columns
        num_style = 3
    else:
        raise ValueError(f"Invalid CONTROL_FEATURES: {CONTROL_FEATURES}")

    # Build design matrix: [intercept | style_features]
    # In our single-model case the intercept replaces the one-hot model
    # encoding from show_result.py line 182-186
    intercept = torch.ones((len(outcomes), 1))
    features = torch.cat([intercept, style_matrix], dim=1)

    if len(valid_items) - null_judgments < 5:
        logger.warning(
            f"Only {len(valid_items) - null_judgments} valid items for "
            "style-controlled scoring; confidence intervals may be unreliable"
        )

    win_rate, ci_lower, ci_upper = bootstrap_style_controlled_winrate(
        features, outcomes, num_style_features=num_style, num_bootstrap=NUM_BOOTSTRAP
    )

    return win_rate, ci_lower, ci_upper, null_judgments


# ── Statistics ───────────────────────────────────────────────────────


def _bootstrap_winrate(scores, num_bootstrap=NUM_BOOTSTRAP, seed=42):
    """Compute bootstrap win rate with confidence interval.

    Resamples the score array to estimate mean win rate and a 90%
    confidence interval, following the arena-hard-auto methodology.

    Args:
        scores: List of numeric scores (0 or 1, with 0.5 for ties).
        num_bootstrap: Number of bootstrap rounds.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (win_rate, ci_lower, ci_upper), all on 0-100 scale.
    """
    scores_arr = np.array(scores)
    rng = np.random.default_rng(seed=seed)

    bootstrap_means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)
    mean_score = bootstrap_means.mean()
    ci_lower = np.quantile(bootstrap_means, 0.05)
    ci_upper = np.quantile(bootstrap_means, 0.95)

    return mean_score, ci_lower, ci_upper


# ── Optional logging ─────────────────────────────────────────────────


def _log_judgments(valid_items, results_map, win_rate):
    """Optionally log per-item judge verdicts to a JSONL file.

    Controlled by the ARENA_HARD_JUDGE_LOG_PATH environment variable.
    Each line is a JSON object with the judge model, overall win rate,
    per-round verdicts, full judgment text, and item-level score.
    """
    judge_log_path = os.getenv(JUDGE_LOG_PATH_ENV)
    if not judge_log_path:
        return

    per_item_logs = []
    for item in valid_items:
        uid = item["uid"]
        judgments = results_map.get(uid, {})
        j1, j2 = judgments.get(1), judgments.get(2)
        v1 = _get_score(j1) if j1 else None
        v2 = _get_score(j2) if j2 else None

        item_score = None
        if v1 and v2:
            r1 = LABEL_TO_SCORE.get(v1)
            r2 = LABEL_TO_SCORE.get(v2)
            if r1 is not None and r2 is not None:
                scores = r2 + [1 - s for s in r1]
                item_score = sum(scores) / len(scores)

        per_item_logs.append({
            "uid": uid,
            "category": item.get("category"),
            "round1_verdict": v1,
            "round2_verdict": v2,
            "round1_judgment": j1,
            "round2_judgment": j2,
            "item_score": item_score,
        })

    try:
        log_dir = os.path.dirname(judge_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(judge_log_path, "a", encoding="utf-8") as f:
            for rec in per_item_logs:
                out = {
                    "metric": "arena_hard_v2",
                    "judge_model": JUDGE_MODEL,
                    "win_rate": win_rate,
                    **rec,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(
            f"Wrote {len(per_item_logs)} Arena-Hard judge logs to {judge_log_path}"
        )
    except Exception as e:
        logger.warning(f"Failed to write judge log {judge_log_path}: {e}")


def _log_truncation(items):
    """Optionally log per-item truncation info to a JSONL file.

    Controlled by the ARENA_HARD_JUDGE_LOG_PATH environment variable.
    The truncation log is saved alongside the judge log with a
    ``_truncation`` suffix (e.g. ``judge.jsonl`` -> ``judge_truncation.jsonl``).

    If ARENA_HARD_MAX_GEN_TOKS is set, a rough heuristic (~4 chars/token)
    flags outputs that likely hit the generation limit.  Truncated outputs
    may end mid-sentence, so the truncation rate is a lower bound on the
    fraction of outputs degraded by the token limit.
    """
    judge_log_path = os.getenv(JUDGE_LOG_PATH_ENV)
    if not judge_log_path:
        return

    base, sep, ext = judge_log_path.rpartition(".")
    if sep:
        truncation_log_path = f"{base}_truncation.{ext}"
    else:
        truncation_log_path = f"{judge_log_path}_truncation"

    max_gen_toks_str = os.getenv(MAX_GEN_TOKS_ENV)
    max_toks = int(max_gen_toks_str) if max_gen_toks_str else None

    truncation_records = []
    truncated_count = 0

    for item in items:
        raw = item.get("raw_output", "")
        clean = item.get("model_output", "")
        char_count = len(raw)
        clean_char_count = len(clean)
        word_count = len(raw.split())
        est_tokens = char_count // 4  # rough heuristic for English text

        if max_toks is not None:
            is_truncated = est_tokens >= int(max_toks * 0.95)
        else:
            is_truncated = None

        if is_truncated:
            truncated_count += 1

        truncation_records.append({
            "uid": item["uid"],
            "raw_output_chars": char_count,
            "clean_output_chars": clean_char_count,
            "raw_output_words": word_count,
            "estimated_tokens": est_tokens,
            "max_gen_toks": max_toks,
            "potentially_truncated": is_truncated,
        })

    try:
        log_dir = os.path.dirname(truncation_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(truncation_log_path, "a", encoding="utf-8") as f:
            summary = {
                "type": "summary",
                "metric": "arena_hard_v2",
                "total_items": len(items),
                "truncated_count": truncated_count,
                "truncation_rate": (
                    truncated_count / len(items) if items else 0
                ),
                "max_gen_toks": max_toks,
            }
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            for rec in truncation_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            f"Wrote truncation log ({truncated_count}/{len(items)} "
            f"potentially truncated) to {truncation_log_path}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to write truncation log {truncation_log_path}: {e}"
        )


# ── Aggregation (orchestrator) ───────────────────────────────────────


def arena_hard_agg(items):
    """Run Arena-Hard pairwise judging and compute bootstrap win rate.

    Orchestrates the full pipeline: filter items -> build prompts ->
    run concurrent judge calls -> score verdicts -> bootstrap statistics.

    Args:
        items: List of dicts from arena_hard_process(), each with keys
            'uid', 'prompt', 'model_output', 'baseline_output', 'category'.

    Returns:
        float: Win rate score (0-100 scale).
    """
    api_key = os.getenv("CSCS_SERVING_API")
    if not api_key:
        raise OSError(
            "CSCS_SERVING_API environment variable must be set for Arena-Hard "
            "judging. Get your key from the CSCS SwissAI platform."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=API_URL,
        api_key=api_key,
        max_retries=_CLIENT_MAX_RETRIES,
        timeout=_TIMEOUT_SECONDS,
    )

    # Filter items without baseline answers
    valid_items = [item for item in items if item["baseline_output"] is not None]
    if len(valid_items) < len(items):
        logger.warning(
            f"Skipping {len(items) - len(valid_items)} items without baseline answers"
        )

    total_calls = 2 * len(valid_items)
    logger.info(
        f"Running Arena-Hard v2.0 judging on {len(valid_items)} items "
        f"using {JUDGE_MODEL} as judge via CSCS API "
        f"({total_calls} judge calls with {_JUDGE_MAX_WORKERS} concurrent workers)..."
    )
    logger.info(f"Scoring method: {SCORING_METHOD} (env ARENA_HARD_SCORING_METHOD)")

    # Build prompts and run judge calls
    tasks = _build_judge_tasks(valid_items)
    results_map = _run_judge_calls(client, tasks)

    # Convert verdicts to scores and compute win rate
    if SCORING_METHOD == "style_control":
        logger.info(
            f"Using style-controlled scoring (features: {CONTROL_FEATURES})"
        )
        win_rate, ci_lower, ci_upper, null_judgments = (
            _compute_scores_style_controlled(valid_items, results_map)
        )
        if null_judgments > 0:
            logger.warning(f"Number of null/invalid judgments: {null_judgments}")
        if win_rate is None:
            logger.error("No valid judgments produced. Returning 0.")
            return 0.0
        score_entries_msg = ""
    else:
        all_scores, null_judgments = _compute_scores_weighted(
            valid_items, results_map
        )
        if null_judgments > 0:
            logger.warning(f"Number of null/invalid judgments: {null_judgments}")
        if not all_scores:
            logger.error("No valid judgments produced. Returning 0.")
            return 0.0
        win_rate, ci_lower, ci_upper = _bootstrap_winrate(all_scores)
        score_entries_msg = f", total score entries: {len(all_scores)}"

    logger.info(
        f"Arena-Hard v2.0 results ({SCORING_METHOD}): {win_rate:.4f} "
        f"(90% CI: {ci_lower:.4f} - {ci_upper:.4f})"
    )
    logger.info(
        f"  Valid judgments: {len(valid_items) - null_judgments}/{len(valid_items)}"
        f"{score_entries_msg}"
    )

    _log_judgments(valid_items, results_map, win_rate)
    _log_truncation(items)

    return float(win_rate)
