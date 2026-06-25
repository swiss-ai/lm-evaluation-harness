#!/usr/bin/env bash
# Launch the RealGuardrails / SystemCheck single-turn benchmarks via vLLM.
#
# This is the vLLM counterpart to run_hf.sh. It runs in one of two modes:
#
#   1. LOCAL ENGINE (default)
#      `--model vllm` — vLLM loads the model in-process. This is the
#      backend the paper (Mu et al., 2025, arXiv:2502.12197) used
#      (max_model_len=4096), so it is the parity-faithful path. Engine
#      knobs are exposed: --tensor-parallel-size, --data-parallel-size,
#      --gpu-memory-utilization, --max-model-len.
#
#   2. REMOTE ENDPOINT (--endpoint URL)
#      `--model local-chat-completions` — talks to an already-running
#      OpenAI-compatible vLLM server (`vllm serve ...`) over HTTP. Useful
#      when the server lives on another node / is shared. Knobs:
#      --num-concurrent, --served-model-name, --api-key, --tokenizer.
#
#      !! IMPORTANT — boilerplate stripping does NOT apply in this mode !!
#      strip_system_boilerplate is a client-side rewrite of the tokenizer's
#      chat template, but in endpoint mode the *server* owns the template
#      and applies it to the messages we send. So Llama 3.x "Cutting
#      Knowledge Date" / "Today Date" boilerplate will leak into the system
#      header and TensorTrust scores will diverge from the paper by
#      ~19-24 pp. To get clean numbers from a server, start it with a
#      pre-stripped chat template, e.g.:
#        vllm serve <model> --chat-template /path/to/stripped_template.jinja
#      The script prints a loud warning for Llama models in endpoint mode.
#
# What this runs
# --------------
# By default: `realguardrails_s_ifeval` + the `realguardrails_tensortrust`
# group (3 leaf tasks aggregated as macro-avg). Override with --tasks.
#
# Critical flags baked in (both modes)
# ------------------------------------
#   --apply_chat_template   The per-doc system prompt only reaches the model
#                           via the chat template's system role. Without it
#                           the constraint is concatenated as a plain-text
#                           prefix and the task silently degrades to
#                           IFEval-Separated.
#   --log_samples           Per-doc generations + verdicts written for
#                           spot-checking against the paper.
#   --seed 42               Greedy decoding is deterministic at T=0.
#
# Boilerplate handling (local engine only)
# ----------------------------------------
# strip_system_boilerplate=true is baked into MODEL_ARGS in local mode and
# auto-strips Llama 3.x date boilerplate from the tokenizer's chat template
# at the Python level. The authority check is OFF by default; the
# RealGuardrails tasks warn if run without engaging any of these model_args:
#   strip_system_boilerplate=true   — regex-strip known Llama 3.x boilerplate
#   check_system_prompt_authority=true        — run the authority probe at init (raises
#                                     RuntimeError on detected boilerplate)
#   allow_system_boilerplate=true   — acknowledge / silence the warning, use
#                                     the template as-is (and waive the probe)
# Baking strip=true handles Llama and silences the warning without running the
# probe (so system-folding templates like gemma-3/4 don't false-positive).
#
# Stop tokens / EOS (local engine)
# --------------------------------
# The S-IFEval task uses `until: []` and relies on the model's EOS token to
# stop generation. lm-eval's vLLM backend derives its stop sequence from the
# *tokenizer's* single `eos_token_id` only (vllm_causallms.py). Some chat
# fine-tunes are mis-configured for this: e.g.
# normster/RealGuardrails-Qwen2.5-7B-SFT declares eos_token=<|endoftext|>
# (and ships NO generation_config.json) while its chat template ends every
# assistant turn with <|im_end|>. vLLM then never sees a stop, runs on to
# max_gen_toks, and emits a wall of <|im_end|> that fails the S-IFEval format
# verifiers. Symptom: S-IFEval collapses ~15-20 pp vs the HF run for that
# model ONLY (TensorTrust, being substring/regex, is unaffected; the HF
# backend is unaffected because it stops/strips at the chat-end token).
#
# Fix: pass the model's real turn-end token as an extra stop via --gen-kwargs:
#   ./run_vllm.sh normster/RealGuardrails-Qwen2.5-7B-SFT \
#       --gen-kwargs 'until=<|im_end|>'
# Well-formed models whose generation_config.json lists the chat EOS (e.g.
# stock Qwen2.5-Instruct's [151645, 151643], or Llama whose tokenizer eos IS
# <|eot_id|>) do not need this.
#
# Prereqs
# -------
#   # Local engine:
#   pip install -e ".[vllm,sysp_eval]"
#   # Remote endpoint:
#   pip install -e ".[api,sysp_eval]"   # only the API client deps are needed
#   # Gated models (meta-llama/*): huggingface-cli login + accept license.
#
# Usage
# -----
#   # Local engine
#   ./run_vllm.sh <model_id>
#   ./run_vllm.sh <model_id> --smoke
#   ./run_vllm.sh <model_id> --tensor-parallel-size 4          # 70B etc.
#   ./run_vllm.sh <model_id> --data-parallel-size 2            # data parallel
#   ./run_vllm.sh <model_id> --gpu-memory-utilization 0.85 --max-model-len 8192
#   ./run_vllm.sh <model_id> --extra-model-args quantization=awq
#   ./run_vllm.sh <model_id> --gen-kwargs 'until=<|im_end|>'   # extra stop token
#
#   # Remote endpoint (vLLM OpenAI server)
#   ./run_vllm.sh <model_id> --endpoint http://localhost:8000
#   ./run_vllm.sh <model_id> --endpoint http://node42:8000/v1 \
#       --served-model-name my-model --num-concurrent 16 --api-key EMPTY
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
LIMIT_ARG=()
EXTRA_MODEL_ARGS=""
EXTRA_GEN_KWARGS=""
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/realguardrails}"
SEED="42"

# Local-engine knobs
TENSOR_PARALLEL_SIZE="1"
DATA_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.9"
MAX_MODEL_LEN="4096"

# Remote-endpoint knobs
ENDPOINT=""
SERVED_MODEL_NAME=""
NUM_CONCURRENT="8"
API_KEY=""
TOKENIZER=""

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
            # 20 docs per task — validates pipeline + chat-template wiring
            # without spending GPU hours.
            LIMIT_ARG=(--limit 20)
            shift
            ;;
        --tasks)
            TASKS="$2"; shift 2 ;;
        --batch_size|--batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --extra-model-args)
            EXTRA_MODEL_ARGS="$2"; shift 2 ;;
        --gen-kwargs|--gen_kwargs)
            # Passed through to lm-eval --gen_kwargs (merged into the task's
            # generation_kwargs). Main use: add a stop token for models whose
            # eos_token is not their chat turn-end, e.g. 'until=<|im_end|>'.
            EXTRA_GEN_KWARGS="$2"; shift 2 ;;
        --output)
            OUTPUT_ROOT="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        # --- local engine ---
        --tensor-parallel-size|--tensor_parallel_size|--tp)
            TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --data-parallel-size|--data_parallel_size|--dp)
            DATA_PARALLEL_SIZE="$2"; shift 2 ;;
        --gpu-memory-utilization|--gpu_memory_utilization|--gmu)
            GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --max-model-len|--max_model_len)
            MAX_MODEL_LEN="$2"; shift 2 ;;
        # --- remote endpoint ---
        --endpoint|--base-url|--base_url)
            ENDPOINT="$2"; shift 2 ;;
        --served-model-name|--served_model_name)
            SERVED_MODEL_NAME="$2"; shift 2 ;;
        --num-concurrent|--num_concurrent)
            NUM_CONCURRENT="$2"; shift 2 ;;
        --api-key|--api_key)
            API_KEY="$2"; shift 2 ;;
        --tokenizer)
            TOKENIZER="$2"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        --)
            shift; break ;;
        -*)
            echo "Unknown flag: $1" >&2; usage 2 ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            else
                echo "Unexpected positional argument: $1" >&2; usage 2
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

# Mode-specific flag validation
if [[ -n "$ENDPOINT" ]]; then
    MODE="endpoint"
    for pair in \
        "--tensor-parallel-size:$TENSOR_PARALLEL_SIZE:1" \
        "--data-parallel-size:$DATA_PARALLEL_SIZE:1" \
        "--gpu-memory-utilization:$GPU_MEMORY_UTILIZATION:0.9" \
        "--max-model-len:$MAX_MODEL_LEN:4096"; do
        flag="${pair%%:*}"; rest="${pair#*:}"; val="${rest%%:*}"; def="${rest##*:}"
        if [[ "$val" != "$def" ]]; then
            echo "error: $flag is a local-engine knob and has no effect with --endpoint." >&2
            echo "Configure the server itself (vllm serve ...) instead." >&2
            exit 2
        fi
    done
else
    MODE="local"
    if [[ -n "$SERVED_MODEL_NAME$API_KEY$TOKENIZER" ]]; then
        echo "error: --served-model-name/--api-key/--tokenizer require --endpoint." >&2
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# Preflight: lm-eval on PATH, resolve owning Python interpreter
# ---------------------------------------------------------------------------
if ! command -v lm-eval >/dev/null 2>&1; then
    echo "error: 'lm-eval' not found on PATH." >&2
    echo "Activate the env that has lm_eval installed:" >&2
    if [[ "$MODE" == "local" ]]; then
        echo "  pip install -e \".[vllm,sysp_eval]\"" >&2
    else
        echo "  pip install -e \".[api,sysp_eval]\"" >&2
    fi
    exit 127
fi

LM_EVAL_BIN="$(command -v lm-eval)"
PY="$(head -n 1 "$LM_EVAL_BIN" 2>/dev/null | sed -e 's|^#!||' -e 's| .*||')"
if [[ ! -x "$PY" ]]; then
    PY="$(command -v python3 || command -v python || true)"
fi

if [[ "$MODE" == "local" ]]; then
    if [[ -z "$PY" ]] || ! "$PY" -c "import vllm" >/dev/null 2>&1; then
        echo "error: vLLM is not importable from the lm-eval env." >&2
        echo "Install the vLLM backend:  pip install -e \".[vllm,sysp_eval]\"" >&2
        exit 1
    fi
    VLLM_VERSION="$("$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo '?')"
    echo "vLLM: $VLLM_VERSION"
fi

# ---------------------------------------------------------------------------
# Build MODEL_ARGS per mode
# ---------------------------------------------------------------------------
if [[ "$MODE" == "local" ]]; then
    LM_EVAL_MODEL="vllm"
    # max_model_len=4096 matches the paper; the longest S-IFEval / TT prompt
    # is ~1.4K tokens so it is a no-op on real data. strip_system_boilerplate
    # auto-cleans Llama 3.x date boilerplate at the Python level.
    MODEL_ARGS="pretrained=${MODEL},dtype=auto,max_model_len=${MAX_MODEL_LEN}"
    MODEL_ARGS="${MODEL_ARGS},tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
    MODEL_ARGS="${MODEL_ARGS},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    if [[ "$DATA_PARALLEL_SIZE" != "1" ]]; then
        MODEL_ARGS="${MODEL_ARGS},data_parallel_size=${DATA_PARALLEL_SIZE}"
    fi
    MODEL_ARGS="${MODEL_ARGS},strip_system_boilerplate=true"
else
    LM_EVAL_MODEL="local-chat-completions"

    # Normalize the endpoint to the chat-completions route.
    BASE_URL="$ENDPOINT"
    case "$BASE_URL" in
        */chat/completions) : ;;                                   # already full
        */v1|*/v1/) BASE_URL="${BASE_URL%/}/chat/completions" ;;   # .../v1 -> +route
        *) BASE_URL="${BASE_URL%/}/v1/chat/completions" ;;         # host:port -> +/v1/...
    esac

    # vLLM servers require *some* API key; default to a placeholder. The
    # API backend reads it from OPENAI_API_KEY.
    if [[ -n "$API_KEY" ]]; then
        export OPENAI_API_KEY="$API_KEY"
    elif [[ -z "${OPENAI_API_KEY:-}" ]]; then
        export OPENAI_API_KEY="EMPTY"
    fi

    SERVED_MODEL="${SERVED_MODEL_NAME:-$MODEL}"
    MODEL_ARGS="model=${SERVED_MODEL},base_url=${BASE_URL}"
    MODEL_ARGS="${MODEL_ARGS},num_concurrent=${NUM_CONCURRENT},max_retries=3"
    if [[ -n "$TOKENIZER" ]]; then
        MODEL_ARGS="${MODEL_ARGS},tokenizer_backend=huggingface,tokenizer=${TOKENIZER}"
    fi

    # Loud warning: server-side template means no client-side stripping.
    if [[ "$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')" == *llama* ]]; then
        echo
        echo "############################################################"
        echo "# WARNING: endpoint mode + Llama model.                     "
        echo "# strip_system_boilerplate does NOT apply over HTTP — the   "
        echo "# server owns the chat template. Llama 3.x date boilerplate "
        echo "# will leak into the system header and TensorTrust will     "
        echo "# diverge from the paper by ~19-24 pp.                       "
        echo "# Start the server with a pre-stripped template:            "
        echo "#   vllm serve $MODEL --chat-template <stripped.jinja>      "
        echo "############################################################"
        echo
    fi
fi

if [[ -n "$EXTRA_MODEL_ARGS" ]]; then
    MODEL_ARGS="${MODEL_ARGS},${EXTRA_MODEL_ARGS}"
fi

# ---------------------------------------------------------------------------
# Build the lm-eval command
# ---------------------------------------------------------------------------
MODEL_SLUG="${MODEL//\//__}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_SLUG}/${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

LIMIT_DISPLAY="(none — full eval)"
if [[ ${#LIMIT_ARG[@]} -gt 0 ]]; then
    LIMIT_DISPLAY="${LIMIT_ARG[1]}"
fi

if [[ "$MODE" == "local" ]]; then
    BACKEND_DISPLAY="vllm (local engine), TP=${TENSOR_PARALLEL_SIZE} DP=${DATA_PARALLEL_SIZE} gmu=${GPU_MEMORY_UTILIZATION} max_model_len=${MAX_MODEL_LEN}"
    BOILERPLATE_DISPLAY="strip"
else
    BACKEND_DISPLAY="local-chat-completions @ ${BASE_URL} (concurrency=${NUM_CONCURRENT})"
    BOILERPLATE_DISPLAY="server-side (not stripped)"
fi

cat <<EOF
=== RealGuardrails / SystemCheck — vLLM run ===
Model:       $MODEL
Mode:        $MODE
Backend:     $BACKEND_DISPLAY
Tasks:       $TASKS
Limit:       $LIMIT_DISPLAY
Batch size:  $BATCH_SIZE
Boilerplate: $BOILERPLATE_DISPLAY
Gen kwargs:  ${EXTRA_GEN_KWARGS:-(task defaults)}
Output:      $OUTPUT_DIR
Seed:        $SEED
EOF

if [[ ${#LIMIT_ARG[@]} -gt 0 ]]; then
    echo
    echo "WARNING: --smoke / --limit truncates each task to the first N docs."
    echo "Resulting numbers are NOT representative and are NOT paper-comparable."
    echo "Drop --smoke for a full run before comparing to Tables 7 / 8."
fi

# Tee a copy of stdout/stderr so the per-run log is alongside the metrics.
LOG_FILE="${OUTPUT_DIR}/run.log"

GEN_KWARGS_ARG=()
if [[ -n "$EXTRA_GEN_KWARGS" ]]; then
    GEN_KWARGS_ARG=(--gen_kwargs "$EXTRA_GEN_KWARGS")
fi

EVAL_ARGS=(
    --model "$LM_EVAL_MODEL"
    --model_args "$MODEL_ARGS"
    --tasks "$TASKS"
    --apply_chat_template
    --batch_size "$BATCH_SIZE"
    --output_path "$OUTPUT_DIR"
    --log_samples
    --seed "$SEED"
    ${GEN_KWARGS_ARG[@]+"${GEN_KWARGS_ARG[@]}"}
    ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"}
)

lm-eval run "${EVAL_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "Done. Results: $OUTPUT_DIR"
echo "Per-doc samples: $OUTPUT_DIR/samples_*.jsonl"
