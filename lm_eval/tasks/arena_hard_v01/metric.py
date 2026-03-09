"""Arena-Hard v0.1 metric for lm-evaluation-harness.

Implements pairwise judge evaluation using Llama-3.3-70B-Instruct as judge,
served via the CSCS SwissAI API.

The evaluation follows the Arena-Hard protocol:
  1. Model outputs are compared pairwise against GPT-4-0314 baseline responses.
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
    CSCS_SERVING_API environment variable (with access to Llama-3.3-70B-Instruct)
"""

import json
import logging
import os
import re
import time

import numpy as np


logger = logging.getLogger(__name__)

# ── CSCS SwissAI serving endpoint ────────────────────────────────────
API_URL = "https://api.swissai.cscs.ch/v1"
JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

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

# ── Pairwise prompt template (from arena-hard-v0.1.yaml) ─────────────
PROMPT_TEMPLATE = (
    "<|User Prompt|>\n{QUESTION}\n\n"
    "<|The Start of Assistant A's Answer|>\n{ANSWER_A}\n"
    "<|The End of Assistant A's Answer|>\n\n"
    "<|The Start of Assistant B's Answer|>\n{ANSWER_B}\n"
    "<|The End of Assistant B's Answer|>"
)

# ── Regex patterns for verdict extraction (from arena-hard-v0.1.yaml) ─
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

NUM_BOOTSTRAP = 100


# ── Baseline answer cache ────────────────────────────────────────────
_BASELINE_CACHE = None


def _load_baseline_answers():
    """Download and cache GPT-4-0314 baseline answers from HuggingFace.

    Returns:
        dict: Mapping from question uid to baseline answer text.
    """
    global _BASELINE_CACHE
    if _BASELINE_CACHE is not None:
        return _BASELINE_CACHE

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="lmarena-ai/arena-hard-auto",
        filename="data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl",
        repo_type="dataset",
    )

    baseline = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            uid = entry["uid"]
            answer = entry["messages"][-1]["content"]["answer"]
            baseline[uid] = answer

    logger.info(f"Loaded {len(baseline)} GPT-4-0314 baseline answers from HuggingFace")
    _BASELINE_CACHE = baseline
    return baseline


# ── Score extraction ─────────────────────────────────────────────────
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


# ── Thinking trace removal ───────────────────────────────────────────
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
    processing phase.  Packages the question, model completion, and GPT-4-0314
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

    return {
        "arena_hard_score": {
            "uid": uid,
            "prompt": doc["prompt"],
            "model_output": clean_output,
            "baseline_output": baseline_output,
            "category": doc.get("category", "arena-hard-v0.1"),
        }
    }


# ── Aggregation (judge calls + scoring) ──────────────────────────────
def arena_hard_agg(items):
    """Run Arena-Hard pairwise judging and compute bootstrap win rate.

    For each item:
      1. Round 1: baseline=A, model=B -> judge produces verdict
      2. Round 2: model=A, baseline=B -> judge produces verdict (position swap)
    Verdicts are mapped to numeric scores with 3x weight for strong preferences.
    Bootstrap resampling (100 rounds) produces mean win rate + 90% CI.

    Scoring logic from arena-hard-auto/show_result.py:
      - games[1] (round 2, model=A): label_to_score used directly
      - games[0] (round 1, baseline=A): label_to_score flipped (1 - s)
      This ensures higher score = better for model in both rounds.

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

    client = OpenAI(base_url=API_URL, api_key=api_key)

    # Filter items without baseline
    valid_items = [item for item in items if item["baseline_output"] is not None]
    if len(valid_items) < len(items):
        logger.warning(
            f"Skipping {len(items) - len(valid_items)} items without baseline answers"
        )

    # Output length statistics and truncation detection.
    # We do not have the tokenizer here, so we use a character-based heuristic:
    # average English token ≈ 4 chars, so max_new_tokens=4096 ≈ 16384 chars.
    # Outputs near this threshold were likely truncated by the generation limit.
    MAX_NEW_TOKENS = 4096  # must match generation_kwargs in arena_hard_v01.yaml
    APPROX_CHARS_PER_TOKEN = 4
    truncation_char_threshold = MAX_NEW_TOKENS * APPROX_CHARS_PER_TOKEN

    output_lengths = [len(item["model_output"]) for item in valid_items]
    likely_truncated = sum(
        1 for length in output_lengths if length >= truncation_char_threshold * 0.95
    )
    logger.info(
        f"Output length stats (chars): "
        f"min={min(output_lengths)}, max={max(output_lengths)}, "
        f"mean={sum(output_lengths) / len(output_lengths):.0f}, "
        f"median={sorted(output_lengths)[len(output_lengths) // 2]}"
    )
    if likely_truncated > 0:
        logger.info(
            f"Likely truncated outputs: {likely_truncated}/{len(valid_items)} "
            f"(>= {truncation_char_threshold * 0.95:.0f} chars, "
            f"approx {MAX_NEW_TOKENS} tokens). Consider increasing "
            f"max_new_tokens in arena_hard_v01.yaml."
        )

    logger.info(
        f"Running Arena-Hard v0.1 judging on {len(valid_items)} items "
        f"using {JUDGE_MODEL} as judge via CSCS API "
        f"(2 rounds per item = {2 * len(valid_items)} judge calls)..."
    )

    all_scores = []
    null_judgments = 0
    judging_start = time.time()

    for i, item in enumerate(valid_items):
        item_start = time.time()
        elapsed = item_start - judging_start
        if i > 0:
            avg_per_item = elapsed / i
            remaining = avg_per_item * (len(valid_items) - i)
            logger.info(
                f"  Judging item {i + 1}/{len(valid_items)} "
                f"(elapsed: {elapsed:.0f}s, ~{remaining:.0f}s remaining)..."
            )
        else:
            logger.info(f"  Judging item {i + 1}/{len(valid_items)}...")

        # ── Round 1: baseline=A, model=B ─────────────────────────
        user_prompt_r1 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["baseline_output"],
            ANSWER_B=item["model_output"],
        )
        try:
            t0 = time.time()
            resp1 = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_r1},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            judgment1 = resp1.choices[0].message.content
            logger.info(f"    Round 1 done ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"Judge API error on item {item['uid']} round 1: {e}")
            judgment1 = None

        # ── Round 2: model=A, baseline=B (position swap) ─────────
        user_prompt_r2 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["model_output"],
            ANSWER_B=item["baseline_output"],
        )
        try:
            t0 = time.time()
            resp2 = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt_r2},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            judgment2 = resp2.choices[0].message.content
            logger.info(f"    Round 2 done ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"Judge API error on item {item['uid']} round 2: {e}")
            judgment2 = None

        # ── Extract scores ────────────────────────────────────────
        score1 = _get_score(judgment1) if judgment1 else None
        score2 = _get_score(judgment2) if judgment2 else None

        logger.info(
            f"    Verdicts: round1={score1}, round2={score2} "
            f"(item took {time.time() - item_start:.1f}s)"
        )

        if score1 is None or score2 is None:
            null_judgments += 1
            logger.warning(
                f"Null judgment for uid={item['uid']}: "
                f"round1={score1}, round2={score2}"
            )
            continue

        r1_scores = LABEL_TO_SCORE.get(score1)
        r2_scores = LABEL_TO_SCORE.get(score2)

        if r1_scores is None or r2_scores is None:
            null_judgments += 1
            logger.warning(
                f"Unknown score label for uid={item['uid']}: "
                f"round1={score1}, round2={score2}"
            )
            continue

        # games[1] (round 2, model=A): scores used directly
        # games[0] (round 1, baseline=A): scores flipped (1 - s)
        item_scores = r2_scores + [1 - s for s in r1_scores]
        all_scores.extend(item_scores)

    total_judging_time = time.time() - judging_start
    logger.info(f"Judging completed in {total_judging_time:.0f}s")

    if null_judgments > 0:
        logger.warning(f"Number of null/invalid judgments: {null_judgments}")

    if not all_scores:
        logger.error("No valid judgments produced. Returning 0.")
        return 0.0

    # ── Bootstrap win rate ────────────────────────────────────────
    scores_arr = np.array(all_scores)
    rng = np.random.default_rng(seed=42)

    bootstrap_means = []
    for _ in range(NUM_BOOTSTRAP):
        sample = rng.choice(scores_arr, size=len(scores_arr), replace=True)
        bootstrap_means.append(sample.mean())

    bootstrap_means = np.array(bootstrap_means)
    mean_score = bootstrap_means.mean()
    ci_lower = np.quantile(bootstrap_means, 0.05)
    ci_upper = np.quantile(bootstrap_means, 0.95)

    win_rate = mean_score * 100
    ci_lower_pct = ci_lower * 100
    ci_upper_pct = ci_upper * 100

    logger.info(
        f"Arena-Hard v0.1 results: {win_rate:.1f}% "
        f"(90% CI: {ci_lower_pct:.1f}% - {ci_upper_pct:.1f}%)"
    )
    logger.info(
        f"  Valid judgments: {len(valid_items) - null_judgments}/{len(valid_items)}, "
        f"total score entries: {len(all_scores)}"
    )

    return float(win_rate)
