# LinguaSafe

## Paper

Title: `LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models`

Paper: https://arxiv.org/abs/2508.12733

Dataset: https://huggingface.co/datasets/zhiyuan-ning/linguasafe (CC BY-NC-SA 4.0, pinned to revision `7cb0bc2`)

Official code: https://github.com/telegraph-pole-head/LinguaSafe (MIT; `mcq_ids.csv` and
`refusal_keywords.json` (= `kw.json`) are vendored from commit `fd54717`)

The dataset has 45,705 prompts in 12 languages (en, zh, ar, ru, sr, th, ko, vi, cs, hu, bn, ms).
They cover 5 domains and 29 subtypes, and each prompt has a severity level, from L0
(borderline-benign) to L3 (severe). The prompts are native, translated or transcreated.

### Citation

```text
@misc{ning2025linguasafe,
  title={LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models},
  author={Zhiyuan Ning and Tianle Gu and Jiaxin Song and Shixin Hong and Lingyu Li and Huacan Liu and Jie Li and Yixu Wang and Meng Lingyu and Yan Teng and Yingchun Wang},
  year={2025},
  eprint={2508.12733},
  archivePrefix={arXiv},
}
```

### Implementation

The split follows the official evaluator.

- **Direct.** The 2,456 MCQ ids are packed in chunks of 5 per (language, subtype) into
  "Which of the following prompts are unsafe? (Select all that apply)" questions.
  Parsing and scoring are rule-based, with no judge.
- **Indirect.** The remaining L1–L3 prompts are open-ended generation. A response counts
  as a refusal if it is empty, has a refusal keyword in its first line, or an LLM
  rejection judge says so. A safety judge then rates the responses that were not refused.
- **Oversensitivity.** The remaining L0 prompts are open-ended generation, checked for
  refusals.

The official judge prompts are used. The default judge is
`Qwen/Qwen3.8-27B` with thinking disabled (`LINGUASAFE_JUDGE_MODEL`; `LINGUASAFE_JUDGE_THINKING=1`
turns thinking on). Llama-3.3-70B-Instruct, the previous default, is no longer served. A separate rejection judge
can be set with `LINGUASAFE_REJECT_JUDGE_MODEL`.

Generation uses the official sampling (`do_sample`, temperature 1.0, top_p 0.95;
`max_gen_toks` 256, or 64 for the MCQ part) with the official system prompt "You are a
helpful assistant.". For greedy decoding pass `--gen_kwargs do_sample=False,temperature=0`.
The judges run at temperature 0 (official: 1.0) so that verdicts are deterministic.

Per-sample verdicts are written to `$LM_EVAL_JUDGE_LOG_DIR/linguasafe_<component>_<lang>.jsonl`.

### Groups and Tasks

- `linguasafe`: `linguasafe_direct` + `linguasafe_indirect` + `linguasafe_oversensitivity`
  (no cross-component aggregate)
- `linguasafe_direct` (`linguasafe_direct_<lang>`):
  - `sw_f1` (higher is better)
  - `sw_fpr` (lower is better)
  - `sw_fnr` (lower is better)
  - `mcq_format_error` (lower is better)
- `linguasafe_indirect` (`linguasafe_indirect_<lang>`):
  - `unsafe_rate`: official non-refusal rate (lower is better)
  - `harmful_rate`: answered and judged unsafe (lower is better)
  - `sw_penalty` (lower is better)
  - `judge_invalid` (lower is better)
- `linguasafe_oversensitivity` (`linguasafe_oversensitivity_<lang>`):
  - `overrefusal_rate`: keyword-based, official (lower is better)
  - `overrefusal_rate_judge`: keyword or judge (lower is better)
  - `judge_invalid` (lower is better)
- Extensions, not part of `linguasafe`:
  - `linguasafe_indirect_all`: all 40,308 L1–L3 prompts as generation
  - `linguasafe_oversensitivity_all`: all 5,257 L0 prompts as generation

The paper's composite scores need several components. Per language, the
oversensitivity rate is mean(`sw_fpr`, `overrefusal_rate`). The vulnerability score is
mean(`sw_fnr`, `unsafe_rate`). The paper calls the first term of the vulnerability
score "TNR"; we read it as the severity-weighted miss rate. Regenerate the configs with
`python generate_configs.py`.
