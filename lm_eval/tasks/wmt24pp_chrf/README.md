# WMT24++ (chrF / BLEU)

Reference-based scoring of the 55 English-to-X pairs in [google/wmt24pp](https://huggingface.co/datasets/google/wmt24pp), using sacrebleu chrF and BLEU. Needs no GPU, COMET or `subset2evaluate`, unlike `wmt24pp_translation` (reference-free CometKiwi22).

```bash
lm_eval --tasks wmt24pp_chrf_en-de_DE ...   # one pair
lm_eval --tasks wmt24pp_chrf ...            # all 55 pairs
```

Rows flagged `is_bad_source` (including the canary row) are dropped.
