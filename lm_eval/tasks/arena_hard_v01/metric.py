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

import concurrent.futures
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


# ── Single judge API call (used by the thread pool) ──────────────────
def _judge_call(client, uid, round_num, user_prompt):
    """Execute a single judge API call. Designed to run inside a thread pool.

    Each call sends the pairwise comparison prompt to the judge model and
    returns the raw judgment text.  Exceptions are caught and logged so that
    one failed request does not crash the entire batch.

    Args:
        client: An openai.OpenAI client instance (thread-safe for I/O calls).
        uid: Unique identifier for the question being judged.
        round_num: 1 or 2, indicating the position-swap round.
        user_prompt: The fully formatted pairwise comparison prompt string.

    Returns:
        Tuple of (uid, round_num, judgment_text_or_None).
    """
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        return (uid, round_num, resp.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Judge API error uid={uid} round={round_num}: {e}")
        return (uid, round_num, None)


# ── Aggregation (judge calls + scoring) ──────────────────────────────

# Number of concurrent threads hitting the judge API.
# vLLM with continuous batching on GH200s can handle high concurrency;
_JUDGE_MAX_WORKERS = 128


def arena_hard_agg(items):
    """Run Arena-Hard pairwise judging and compute bootstrap win rate.

    **Concurrency strategy (Fully Flat ThreadPoolExecutor)**:
    All 2*N judge API calls (round 1 + round 2 for every item) are
    completely independent of each other — no call needs the result of
    any other call.  We therefore flatten them into a single list and
    fire them all at once through a ThreadPoolExecutor.  The vLLM server
    uses continuous batching, so it can serve many requests in parallel
    efficiently.

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

    # ── Filter items that have no baseline answer ─────────────────
    valid_items = [item for item in items if item["baseline_output"] is not None]
    if len(valid_items) < len(items):
        logger.warning(
            f"Skipping {len(items) - len(valid_items)} items without baseline answers"
        )

    total_calls = 2 * len(valid_items)
    logger.info(
        f"Running Arena-Hard v0.1 judging on {len(valid_items)} items "
        f"using {JUDGE_MODEL} as judge via CSCS API "
        f"({total_calls} judge calls with {_JUDGE_MAX_WORKERS} concurrent workers)..."
    )

    # ── Build the flat list of all 2*N judge tasks ────────────────
    # Each task is a tuple: (uid, round_number, formatted_user_prompt).
    # Round 1: baseline is position A, model is position B.
    # Round 2: model is position A, baseline is position B (swap).
    # Both rounds for the same item are independent — neither needs
    # the other's result — so we can fire them all at once.
    tasks = []
    for item in valid_items:
        uid = item["uid"]

        # Round 1 prompt: baseline=A, model=B
        prompt_r1 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["baseline_output"],
            ANSWER_B=item["model_output"],
        )
        tasks.append((uid, 1, prompt_r1))

        # Round 2 prompt: model=A, baseline=B (position swap)
        prompt_r2 = PROMPT_TEMPLATE.format(
            QUESTION=item["prompt"],
            ANSWER_A=item["model_output"],
            ANSWER_B=item["baseline_output"],
        )
        tasks.append((uid, 2, prompt_r2))

    # ── Fire all judge calls concurrently ─────────────────────────
    # We use concurrent.futures.ThreadPoolExecutor because the workload
    # is I/O-bound (waiting on HTTP responses from the vLLM server).
    # Threads release the GIL during I/O, so Python threads are ideal
    # here — no need for multiprocessing or async.
    #
    # results_map collects judgments keyed by (uid, round_num) so we
    # can pair them back up after all calls finish.
    results_map = {}  # uid -> {round_num: judgment_text_or_None}
    completed = 0  # counter for progress logging
    judging_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=_JUDGE_MAX_WORKERS
    ) as executor:
        # Submit all tasks to the pool.  executor.submit() returns a Future.
        # We store a mapping from Future -> (uid, round) so we can identify
        # which task completed when we iterate over as_completed().
        future_to_task = {
            executor.submit(_judge_call, client, uid, rnd, prompt): (uid, rnd)
            for uid, rnd, prompt in tasks
        }

        # as_completed() yields futures in the order they finish (not
        # submission order).  This lets us log progress as results arrive
        # rather than blocking on the slowest call first.
        for future in concurrent.futures.as_completed(future_to_task):
            uid, rnd, judgment = future.result()
            # Store the judgment text in our results map
            results_map.setdefault(uid, {})[rnd] = judgment
            completed += 1

            # Log progress every 50 completed calls so we can monitor
            # throughput without flooding the logs.
            if completed % 50 == 0 or completed == total_calls:
                elapsed = time.time() - judging_start
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Judge progress: {completed}/{total_calls} calls done "
                    f"({elapsed:.0f}s elapsed, {rate:.1f} calls/s)"
                )

    total_judging_time = time.time() - judging_start
    logger.info(
        f"All {total_calls} judge calls completed in {total_judging_time:.0f}s "
        f"({total_calls / total_judging_time:.1f} calls/s)"
    )

    # ── Pair up round 1 + round 2 results and compute scores ─────
    all_scores = []
    null_judgments = 0

    for item in valid_items:
        uid = item["uid"]
        judgments = results_map.get(uid, {})

        # Retrieve the raw judgment text for each round
        judgment1 = judgments.get(1)
        judgment2 = judgments.get(2)

        # Extract the verdict label (e.g. "A>>B", "B>A") from the text
        score1 = _get_score(judgment1) if judgment1 else None
        score2 = _get_score(judgment2) if judgment2 else None

        logger.debug(f"  uid={uid}: round1={score1}, round2={score2}")

        # Both rounds must succeed for this item to contribute to the score
        if score1 is None or score2 is None:
            null_judgments += 1
            logger.warning(
                f"Null judgment for uid={uid}: round1={score1}, round2={score2}"
            )
            continue

        # Map verdict labels to numeric score lists
        r1_scores = LABEL_TO_SCORE.get(score1)
        r2_scores = LABEL_TO_SCORE.get(score2)

        if r1_scores is None or r2_scores is None:
            null_judgments += 1
            logger.warning(
                f"Unknown score label for uid={uid}: "
                f"round1={score1}, round2={score2}"
            )
            continue

        # Scoring convention from arena-hard-auto/show_result.py:
        #   - Round 2 (model=A): use label_to_score directly.
        #   - Round 1 (baseline=A): flip scores (1 - s) so that
        #     higher score always = better for the model under test.
        item_scores = r2_scores + [1 - s for s in r1_scores]
        all_scores.extend(item_scores)

    if null_judgments > 0:
        logger.warning(f"Number of null/invalid judgments: {null_judgments}")

    if not all_scores:
        logger.error("No valid judgments produced. Returning 0.")
        return 0.0

    # ── Bootstrap win rate ────────────────────────────────────────
    # NOTE: Currently this is not used/passed on. But also not a real
    # bottleneck compared with the judge calls.
    # So it is left for now and can be obtained from the logs if needed.
    # Resample the score array 100 times to estimate mean win rate
    # and a 90% confidence interval, following the arena-hard-auto
    # methodology.
    # In future we might also want to implement this with style control and
    # BT model,
    # but as we always compare against same baseline BT is not really needed...

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
