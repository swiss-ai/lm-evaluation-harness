#!/usr/bin/env bash
# Run inside an allocated GPU job, with this branch installed in its environment.
set -euo pipefail
model=${1:?Usage: bash scripts/run_polar_vllm.sh MODEL [additional lm-eval arguments]}
shift
: "${POLAR_ATTACKER_BASE_URL:?Set the fixed B endpoint for protocol 5}"
export POLAR_MODEL_A_NAME="$model"
export POLAR_THINKING_MODE=required
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# Explicit token overrides are useful for templates without discoverable markers.
model_args="pretrained=$model,enable_thinking=true,autodetect_think_tokens=true,track_thinking_metrics=true,check_system_prompt_authority=true"
if [[ -n "${POLAR_THINK_END_TOKEN:-}" ]]; then
    model_args+=",think_end_token=$POLAR_THINK_END_TOKEN"
fi
if [[ -n "${POLAR_THINK_START_TOKEN:-}" ]]; then
    model_args+=",think_start_token=$POLAR_THINK_START_TOKEN"
fi
if [[ -n "${POLAR_VLLM_ARGS:-}" ]]; then
    model_args+=",$POLAR_VLLM_ARGS"
fi
exec python -m lm_eval run \
    --model vllm --model_args "$model_args" \
    --tasks polar_bench --include_path "$root/lm_eval/tasks/polar_bench" \
    --apply_chat_template --num_fewshot 0 --log_samples --log_length_metrics \
    --output_path "${POLAR_OUTPUT_PATH:-results/polar_bench}" "$@"
