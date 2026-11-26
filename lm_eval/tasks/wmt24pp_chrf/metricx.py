"""MetricX-24 for the wmt24pp_metricx tasks: reference-based, as in the WMT24++ paper.

Follows google-research/metricx (metricx24/predict.py and models.py):

- the input is ``"source: {src} candidate: {hyp} reference: {ref}"``, tokenized with the
  mT5 tokenizer, truncated to 1536 tokens, without its trailing EOS;
- the model is an mT5 encoder-decoder run for one decoder step from token 0; the score is
  the logit of token 250089 (``<extra_id_10>``), clamped to [0, 25].

``MT5ForRegression`` holds the same weights as transformers' ``MT5ForConditionalGeneration``
(shared embedding, encoder, decoder, untied lm_head), so the checkpoint loads into it.

Scores are errors: 0 is a perfect translation, 25 the worst, so lower is better.

The model is MetricX-24-Hybrid-XXL (bfloat16) by default, the size MetricX recommends;
``METRICX_MODEL`` and ``METRICX_TOKENIZER`` choose another. Scoring runs once, in the
aggregation step, after every translation is generated: on a GPU when torch has one,
otherwise on every CPU core the process may use.
"""

import logging
import os
import time


DEFAULT_MODEL = "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
DEFAULT_TOKENIZER = "google/mt5-xxl"
MAX_INPUT_LENGTH = 1536
SCORE_TOKEN_ID = 250089  # <extra_id_10>
MAX_SCORE = 25.0
# Inputs per forward pass; sorted by length, so a batch pads little.
BATCH_SIZE = int(os.environ.get("METRICX_BATCH_SIZE", "8"))

eval_logger = logging.getLogger(__name__)


def metricx_input(source, hypothesis, reference):
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


def process_results(doc, results):
    """chrF and BLEU items as the built-in metrics make them, and MetricX's inputs."""
    pred = results[0]
    ref = doc["target"]
    return {
        "chrf": (ref, pred),
        "bleu": (ref, pred),
        "metricx24": {"source": doc["source"], "hypothesis": pred, "reference": ref},
    }


def _use_all_cpus(torch):
    """A SLURM step may be pinned to fewer cores than the node has; use what it may."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        cpus = os.cpu_count() or 1
    torch.set_num_threads(cpus)
    return cpus


def _load(model_name, tokenizer_name):
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "bfloat16" in model_name else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
    model.to(device).eval()
    return tokenizer, model, device


def score(items, model=None, tokenizer=None, device=None, batch_size=BATCH_SIZE):
    """One MetricX-24 score per item ({source, hypothesis, reference}), in item order."""
    import torch

    if model is None:
        model_name = os.environ.get("METRICX_MODEL", DEFAULT_MODEL)
        tokenizer_name = os.environ.get("METRICX_TOKENIZER", DEFAULT_TOKENIZER)
        cpus = _use_all_cpus(torch)
        tokenizer, model, device = _load(model_name, tokenizer_name)
        eval_logger.info(
            f"MetricX-24: {model_name} on {device} ({cpus} CPU threads), "
            f"{len(items)} segments, batch {batch_size}"
        )

    encoded = [
        tokenizer(
            metricx_input(it["source"], it["hypothesis"], it["reference"]),
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )["input_ids"][:-1]  # drop the EOS, as MetricX does
        for it in items
    ]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    scores = [0.0] * len(encoded)
    pad = tokenizer.pad_token_id
    start = time.monotonic()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        width = max(len(encoded[i]) for i in idx)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            ids[row, : len(encoded[i])] = torch.tensor(encoded[i])
            mask[row, : len(encoded[i])] = 1
        decoder_ids = torch.zeros((len(idx), 1), dtype=torch.long)
        with torch.inference_mode():
            logits = model(
                input_ids=ids.to(device),
                attention_mask=mask.to(device),
                decoder_input_ids=decoder_ids.to(device),
            ).logits
        batch = logits[:, 0, SCORE_TOKEN_ID].float().clamp(0, MAX_SCORE).tolist()
        for i, s in zip(idx, batch):
            scores[i] = s
        done = b + len(idx)
        if done == len(order) or (b // batch_size) % 10 == 0:
            elapsed = time.monotonic() - start
            eval_logger.info(
                f"MetricX-24: {done}/{len(order)} segments scored ({elapsed:.0f}s)"
            )
    return scores


def metricx24_agg(items):
    """Mean MetricX-24 error over the task's segments (lower is better)."""
    if not items:
        raise RuntimeError("MetricX-24: no segments to score")
    scores = score(items)
    mean = sum(scores) / len(scores)
    eval_logger.info(f"MetricX-24 mean error: {mean:.4f} over {len(scores)} segments")
    return mean
