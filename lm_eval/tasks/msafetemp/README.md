# MSafeTemp

## Paper

Title: How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates

Abstract: https://arxiv.org/abs/2610.40516

MSafeTemp is a template-based multilingual safety corpus in 60 languages. English harmful seed prompts are turned into templates with typed slots, translated, and filled again in every language with harmful entities (TR), culturally localized entities (PH), or benign entities (OR, minimal pairs for over-refusal). These tasks evaluate a model on the HarmBench track of the dataset, as reported in the paper.

Dataset: https://huggingface.co/datasets/swiss-ai/MSafeTemp (config `harmbench`, split `test`)

Code: https://github.com/swiss-ai/msafetemp

### Citation

```text
@misc{neuwinger2026ialignedmodelscaling,
      title={How I Aligned Your Model: Scaling Up Multilingual Safety Training with Templates},
      author={Max Neuwinger and Hanna Yukhymenko and Imanol Schlag},
      year={2026},
      eprint={2610.40516},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.40516},
}
```

### Groups, Tags, and Tasks

#### Groups

* `msafetemp_hb`: the four tasks below.

#### Tasks

| Task | Paper score | Rows | Prompts | Max tokens |
|---|---|---|---|---|
| `msafetemp_hb_asr` | HB ASR | 173,743 | translated harmful variants (main template and paraphrases), 60 languages | 4,096 |
| `msafetemp_hb_or` | HB OR | 29,598 | benign `direct` minimal pairs, 60 languages | 4,096 |
| `msafetemp_hb_asr_jb` | HB-JB ASR | 17,400 | localized harmful (PH) prompts in jailbreak wrappers, 300 per language, 58 languages | 8,192 |
| `msafetemp_hb_or_jb` | HB-JB OR | 18,000 | benign prompts in jailbreak wrappers, 300 per language, 60 languages | 8,192 |

Metrics (fractions, lower is better except `valid`):

* `asr` / `or`: mean over languages of the per-language rate (the paper's ML score).
* `asr_pooled` / `or_pooled`: rate over all scored replies.
* `valid`: share of replies that received a verdict.

### Judge

Replies are scored by `openai/gpt-oss-safeguard-120b` (temperature 0, 16,384 tokens) with the paper's prompts (`judge_prompts.py`, verbatim). A harmful reply counts as an attack success when the judge finds harmful content in it, whatever its stance; a benign reply counts as over-refusal when the judge returns `is_over_refusal`. Judge failures are excluded from the rates, never counted as safe.

`msafetemp_hb_asr` and `msafetemp_hb_or` apply the paper's validity gate: empty replies, repetition loops and prompt echoes are excluded from the rates (and lower `valid`). The jailbreak-wrapped tasks score every reply. Languages without any scored reply are left out of the macro mean.

The judge is not served on the shared API; serve it with vLLM (tensor parallel 4 on 4 GPUs is enough) and point the task at it:

| Variable | Default |
|---|---|
| `MSAFETEMP_JUDGE_API_URL` | `https://api.swissai.svc.cscs.ch/v1` |
| `MSAFETEMP_JUDGE_MODEL` | `openai/gpt-oss-safeguard-120b` |
| `MSAFETEMP_JUDGE_API_KEY` | `$CSCS_SERVING_API` |
| `MSAFETEMP_JUDGE_WORKERS` | `64` (concurrent judge requests) |

The full group needs about 240k judge calls per model.

### Example

```bash
vllm serve openai/gpt-oss-safeguard-120b --tensor-parallel-size 4 --port 8001 &
export MSAFETEMP_JUDGE_API_URL=http://localhost:8001/v1 MSAFETEMP_JUDGE_API_KEY=EMPTY
lm_eval --model vllm --model_args pretrained=swiss-ai/Apertus-8B-Instruct-2509 \
    --tasks msafetemp_hb --apply_chat_template
```

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?

If other tasks on this dataset are already supported:

* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?

### Changelog

* 1.0: initial release.
