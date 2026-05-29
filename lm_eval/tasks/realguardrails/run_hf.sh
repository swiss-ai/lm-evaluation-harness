#!/usr/bin/env bash
# Launch the RealGuardrails / SystemCheck single-turn benchmarks locally
# via the HF (transformers) backend.
#
# What this runs
# --------------
# By default: `realguardrails_s_ifeval` + the `realguardrails_tensortrust`
# group (3 leaf tasks aggregated as macro-avg). Override with --tasks.
#
# Critical flags baked in
# -----------------------
#   --apply_chat_template   The per-doc system prompt only reaches the model
#                           via the chat template's system role. Without
#                           this flag the constraint gets concatenated as a
#                           plain-text prefix to the user message and the
#                           task silently turns into IFEval-Separated.
#   --log_samples           Per-doc generations + verdicts are written so
#                           you can spot-check disagreements with the paper.
#   --seed 42               Greedy decoding is deterministic at T=0, but
#                           accelerate / dataloader shuffling still touch RNG.
#
# Backend caveats
# ---------------
# The paper (Mu et al., 2025, arXiv:2502.12197) runs evaluation on **vLLM**
# with `max_model_len=4096`. This script uses the **HF transformers**
# backend instead, which means:
#
#   * Numbers may drift from the paper by ~0.5–1.0 pp even at T=0 — bf16
#     attention kernel differences (FlashAttention vs SDPA), rotary
#     precision, and batched-vs-single-batch accumulation produce
#     occasional argmax disagreements at the model's confidence margin.
#     This is intrinsic to the backend swap; it is not a config bug.
#
#   * `max_length=4096` IS set in MODEL_ARGS — required to keep the HF
#     auto-batching probe from allocating ~48 GiB of bf16 embeddings on
#     Llama-3.x (whose `max_position_embeddings` is 128K). It does NOT
#     truncate real prompts: empirically the longest S-IFEval / TT
#     prompt is 1,448 tokens. As a side effect this also matches the
#     paper's vLLM `max_model_len=4096`. The cap and the paper's
#     `max_model_len` are conceptually different (one bounds the HF
#     probe + harness truncation logic; the other is a vLLM KV-cache
#     setting) but at this value both happen to be no-ops for the
#     actual benchmark data.
#
# Chat-template caveat
# --------------------
# Tested on Llama-3.x and Qwen-2.5 tokenizers, whose chat templates
# natively emit a system-role slot. Templates that **fold system → user**
# (e.g., Mistral v0.1 / v0.2's `[INST]` wrapping) silently degrade the
# benchmark to IFEval-Separated / TensorTrust-without-defender semantics.
# Verify your model's chat_template before reading paper-parity into the
# numbers.
#
# Prereqs
# -------
#   pip install -e ".[hf,sysp_eval]"
#   # For gated models (the meta-llama/* family below):
#   huggingface-cli login    # then accept each model's license on hf.co
#
# Models worth running for paper-faithfulness checks (Tables 7 + 8).
# Targets are paper-published `prompt_strict` for the wide panel; the
# 4-tuple for Llama-3.1-8B-Instruct and Llama-3.3-70B-Instruct is the
# (prompt_strict / inst_strict / prompt_loose / inst_loose) row from
# Table 8.
#
#   meta-llama/Llama-3.2-3B-Instruct                  → prompt_strict 58.7
#   meta-llama/Llama-3.1-8B-Instruct                  → 66.6 / 76.6 / 69.6 / 78.5
#   normster/RealGuardrails-Llama3.1-8B-Instruct-SFT-DPO → prompt_strict 83.8
#   meta-llama/Llama-3.3-70B-Instruct                 → 87.9 / 92.1 / 90.2 / 93.6
#       (70B; needs tensor-parallel via `--extra-model-args parallelize=True`)
#
# Apple Silicon note
# ------------------
# The HF backend in lm-eval does NOT auto-detect MPS — its
# auto-detection only checks CUDA and silently falls back to CPU. This
# script preempts that: on Darwin without an explicit --device, it
# probes PyTorch for MPS availability and torch >= 2.1, then sets
# DEVICE=mps. Linux/Windows are left to the harness's default.
# 3B fits comfortably in 16-24 GB unified memory; 8B is feasible but
# slow (expect tens of minutes for the smoke test). For 70B you want a
# multi-GPU box.
#
# Usage
# -----
#   ./run_hf.sh <model_id>
#   ./run_hf.sh <model_id> --smoke
#   ./run_hf.sh <model_id> --tasks realguardrails_s_ifeval --batch_size 4
#   ./run_hf.sh <model_id> --extra-model-args dtype=bfloat16,parallelize=True
#
# Outputs land in:
#   ./results/realguardrails/<model_slug>/<UTC-timestamp>/

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
TASKS="realguardrails_s_ifeval,realguardrails_tensortrust"
BATCH_SIZE="auto"
DEVICE=""
LIMIT=""
EXTRA_MODEL_ARGS=""
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/realguardrails}"
SEED="42"

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed '$d'
    exit "${1:-0}"
}

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke)
            # 20 docs per task — enough to validate the pipeline + chat
            # template wiring without spending GPU hours.
            LIMIT="--limit 20"
            shift
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --extra-model-args)
            EXTRA_MODEL_ARGS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown flag: $1" >&2
            usage 2
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            else
                echo "Unexpected positional argument: $1" >&2
                usage 2
            fi
            shift
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "error: <model_id> is required, e.g.:" >&2
    echo "  $0 meta-llama/Llama-3.2-3B-Instruct --smoke" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Preflight: lm-eval on PATH, torch importable, version reportable
# ---------------------------------------------------------------------------
if ! command -v lm-eval >/dev/null 2>&1; then
    echo "error: 'lm-eval' not found on PATH." >&2
    echo "Activate the env that has lm_eval installed:" >&2
    echo "  pip install -e \".[hf,sysp_eval]\"" >&2
    exit 127
fi

# Resolve the Python interpreter that owns the lm-eval install via the
# shebang on the lm-eval entry point. Falls back to whatever python3 /
# python is on PATH if the shebang is unparseable (e.g. a wrapper).
LM_EVAL_BIN="$(command -v lm-eval)"
PY="$(head -n 1 "$LM_EVAL_BIN" 2>/dev/null | sed -e 's|^#!||' -e 's| .*||')"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || command -v python || true)"
fi

if [[ -z "$PY" ]] || ! "$PY" -c "import torch" >/dev/null 2>&1; then
    echo "error: PyTorch is not importable from the lm-eval env." >&2
    echo "Install the HF backend deps:" >&2
    echo "  pip install -e \".[hf,sysp_eval]\"" >&2
    exit 1
fi

TORCH_VERSION="$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null)"
echo "PyTorch: $TORCH_VERSION"

# ---------------------------------------------------------------------------
# Device auto-detection
# ---------------------------------------------------------------------------
# On macOS the lm-eval HF backend does NOT auto-pick MPS — its
# auto-detect branch only checks CUDA and silently falls back to CPU
# (`lm_eval/models/huggingface.py:169-175`). Pre-empt that: when running
# on Darwin without an explicit --device, probe for MPS and use it if
# available. Linux + Windows are left to the harness's CUDA/CPU default
# logic.
if [[ -z "$DEVICE" && "$(uname)" == "Darwin" ]]; then
    # Two-line probe: (a) MPS built+available, (b) torch >= 2.1 (the
    # harness raises otherwise at `huggingface.py:162-167`).
    PROBE="$("$PY" -c "
import torch
v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
mps_ok = torch.backends.mps.is_available()
torch_ok = v >= (2, 1)
print('mps' if (mps_ok and torch_ok) else ('low_torch' if mps_ok else 'no_mps'))
" 2>/dev/null)"
    case "$PROBE" in
        mps)
            DEVICE="mps"
            echo "macOS detected with PyTorch MPS available; defaulting to --device mps."
            ;;
        low_torch)
            echo "warning: macOS + MPS detected, but PyTorch $TORCH_VERSION < 2.1." >&2
            echo "warning: lm-eval refuses MPS on torch < 2.1; will fall back to CPU." >&2
            echo "warning: upgrade torch or pass --device explicitly to silence this." >&2
            ;;
        no_mps|*)
            echo "warning: macOS detected but PyTorch MPS is not available; falling back to CPU." >&2
            ;;
    esac
fi

# ---------------------------------------------------------------------------
# Build the lm-eval command
# ---------------------------------------------------------------------------
MODEL_SLUG="${MODEL//\//__}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# `max_length=4096` is required for the HF backend on Llama-3.x. It does
# NOT truncate actual prompts (the longest S-IFEval / TensorTrust prompt
# is ~1.4K tokens) — but it bounds the test tensor that
# `_detect_batch_size` allocates during the auto-batching probe
# (`lm_eval/models/huggingface.py:825-827`). Llama-3.1/3.2/3.3 ship with
# `max_position_embeddings = 131072` (128K), so the default probe tries
# to allocate ~48 GiB of bf16 embeddings on the very first attempt and
# OOMs. Bound by `max_length` instead. Matches the paper's vLLM setup
# (`max_model_len=4096`) as a side benefit.
MODEL_ARGS="pretrained=${MODEL},dtype=auto,max_length=4096"
if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
    MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

DEVICE_ARG=()
if [[ -n "$DEVICE" ]]; then
    DEVICE_ARG=(--device "$DEVICE")
fi

cat <<EOF
=== RealGuardrails / SystemCheck — local HF run ===
Model:       $MODEL
Tasks:       $TASKS
Limit:       ${LIMIT:+${LIMIT##--limit }}${LIMIT:-(none — full eval)}
Batch size:  $BATCH_SIZE
Device:      ${DEVICE:-(auto)}
Model args:  $MODEL_ARGS
Output:      $OUTPUT_DIR
Seed:        $SEED
EOF

if [[ -n "$LIMIT" ]]; then
    echo
    echo "WARNING: --smoke / --limit truncates each task to the first N docs."
    echo "Resulting numbers are NOT representative and are NOT paper-comparable."
    echo "Drop --smoke for a full run before comparing to Tables 7 / 8."
fi

# Tee a copy of stdout/stderr so the per-run log is alongside the metrics.
LOG_FILE="${OUTPUT_DIR}/run.log"

# shellcheck disable=SC2086  # we intentionally word-split $LIMIT
lm-eval run \
    --model hf \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --apply_chat_template \
    --batch_size "$BATCH_SIZE" \
    --output_path "$OUTPUT_DIR" \
    --log_samples \
    --seed "$SEED" \
    "${DEVICE_ARG[@]}" \
    $LIMIT 2>&1 | tee "$LOG_FILE"

echo
echo "Done. Results: $OUTPUT_DIR"
echo "Per-doc samples: $OUTPUT_DIR/samples_*.jsonl"
