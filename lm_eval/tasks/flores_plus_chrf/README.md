# FLORES+ (chrF / BLEU)

Reference-based chrF and BLEU on the `devtest` split (1012 sentences) of [openlanguagedata/flores_plus](https://huggingface.co/datasets/openlanguagedata/flores_plus). English to and from German, French, Italian and Spanish (8 directions). The dataset is gated: request access on the Hub and give the job an `HF_TOKEN`.

```bash
lm_eval --tasks flores_plus_chrf_eng_Latn-deu_Latn ...   # one direction
lm_eval --tasks flores_plus_chrf ...                     # all 8
```

Other languages: add a `flores_plus_chrf_<src>-<tgt>.yaml` using any `devtest/<lang>.jsonl` file name.
