# MT-Bench

This task integrates the FastChat MT-Bench question set in single-model scoring
mode. It uses the harness `generate_until_multiturn` output type to collect two
assistant turns per question, then defers LLM-as-judge scoring to aggregation.

The reported scores mirror the reference single-answer judgment flow:

- `mt_bench_turn_1_score`: single-turn score for the first answer.
- `mt_bench_turn_2_score`: multi-turn score focused on the second answer.
- `mt_bench_score`: average across valid turn-1 and turn-2 judge scores.
- `mt_bench_judge_success_rate`: fraction of judge calls that returned a valid
  score.

Math, reasoning, and coding categories use the reference-aware judge prompts
from the MT-Bench reference implementation. Other categories use the generic
judge prompts.

Configure the judge with an OpenAI-compatible endpoint:

```bash
export MT_BENCH_JUDGE_MODEL='Qwen/Qwen3.5-27B'
export MT_BENCH_JUDGE_API_KEY='...'
# Optional:
export MT_BENCH_JUDGE_API_BASE='https://api.example.com/v1'
export MT_BENCH_JUDGE_MAX_WORKERS=4
export MT_BENCH_JUDGE_LOG_PATH='mt_bench_judge.jsonl'

lm_eval --tasks mt_bench ...
```

If `MT_BENCH_JUDGE_API_KEY` is unset, the task also accepts `CSCS_SERVING_API`
or `OPENAI_API_KEY`. When `CSCS_SERVING_API` is used without an explicit base
URL, the CSCS SwissAI serving endpoint is selected.

The default judge model is `Qwen/Qwen3.5-27B`, matching the hosted judge used by
HalluLens, Arena-Hard, and Swiss AI Charter in this repository.
