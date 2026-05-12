# Swiss AI Charter Alignment

This task evaluates generated responses to Swiss AI Charter prompt subsets with a hosted OpenAI-compatible judge.

The task bundles:

- `data/prompts/prism_prompts_filtered_220_hf.parquet`
- `data/prompts/wildchat_prompts_filtered_220_hf.parquet`
- `data/prompts/synthetic_prompts_filtered_220_hf.parquet`
- `data/charter.json`

Available tasks:

- `swiss_ai_charter_alignment_prism`: 220 PRISM prompts.
- `swiss_ai_charter_alignment_wildchat`: 220 WildChat prompts.
- `swiss_ai_charter_alignment_synthetic`: 220 synthetic prompts.
- `swiss_ai_charter_alignment_full`: joined 660-prompt dataset.
- `swiss_ai_charter_alignment`: group over the three 220-prompt source subsets.

Useful environment variables:

- `SWISS_AI_CHARTER_JUDGE_MODEL`: hosted judge model. Defaults to `Qwen/Qwen3.5-27B`.
- `SWISS_AI_CHARTER_JUDGE_API_KEY`, `CSCS_SERVING_API`, or `OPENAI_API_KEY`: judge API key.
- `SWISS_AI_CHARTER_JUDGE_API_BASE`: judge API base URL. Defaults to the SwissAI CSCS endpoint when `CSCS_SERVING_API` is set.
- `SWISS_AI_CHARTER_PROMPTS_PATH`: optional override for the prompts parquet.
- `SWISS_AI_CHARTER_PRISM_PROMPTS_PATH`, `SWISS_AI_CHARTER_WILDCHAT_PROMPTS_PATH`, `SWISS_AI_CHARTER_SYNTHETIC_PROMPTS_PATH`: optional subset-specific parquet overrides.
- `SWISS_AI_CHARTER_PATH`: optional override for the charter JSON.
- `SWISS_AI_CHARTER_JUDGE_LOG_PATH`: optional JSONL path for per-sample judge records.
- `SWISS_AI_CHARTER_JUDGE_MAX_WORKERS`: concurrent judge calls. Defaults to `8`.
- `SWISS_AI_CHARTER_JUDGE_LOGPROBS`: set to `0` to disable logprob scoring and parse the sampled digit instead.

Example:

```bash
CSCS_SERVING_API=... lm_eval --model ... --tasks swiss_ai_charter_alignment
CSCS_SERVING_API=... lm_eval --model ... --tasks swiss_ai_charter_alignment_full
```
