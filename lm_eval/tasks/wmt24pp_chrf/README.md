# WMT24++ (chrF / BLEU)

Reference-based scoring of the 55 English-to-X pairs in [google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp), using sacrebleu chrF and BLEU. Needs no GPU, COMET or `subset2evaluate`, unlike `wmt24pp_translation` (reference-free CometKiwi22).

```bash
lm_eval --tasks wmt24pp_chrf_en-de_DE ...   # one pair
lm_eval --tasks wmt24pp_chrf ...            # all 55 pairs
```

Rows flagged `is_bad_source` (including the canary row) are dropped.

## MetricX-24 (`wmt24pp_metricx`)

`wmt24pp_metricx_en-ja_JP` (and the other 54 pairs, group `wmt24pp_metricx`) is the same task scored with reference-based [MetricX-24](https://github.com/google-research/metricx) as well as chrF and BLEU, as in the WMT24++ paper. `metricx24` is an error score from 0 (perfect) to 25, so **lower is better**.

- Model: MetricX-24-Hybrid-XXL in bfloat16 (`google/metricx-24-hybrid-xxl-v2p6-bfloat16`, ~26 GB, downloaded on first use), tokenizer `google/mt5-xxl`. `METRICX_MODEL` / `METRICX_TOKENIZER` pick another size (e.g. `google/metricx-24-hybrid-xl-v2p6-bfloat16` with `google/mt5-xl`).
- Scoring runs once all translations are generated, on a GPU if torch has one and otherwise on every CPU core the process may use; `METRICX_BATCH_SIZE` (default 8) sets the batch.
- Inputs follow `metricx24/predict.py`: `source: … candidate: … reference: …`, at most 1536 tokens, trailing EOS removed; the score is the clamped logit of token 250089 from one decoder step.
