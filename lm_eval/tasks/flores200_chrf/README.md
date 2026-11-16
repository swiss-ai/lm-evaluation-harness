# FLORES-200 (chrF / BLEU)

Reference-based chrF and BLEU on the `devtest` split (1012 sentences) of [facebook/flores](https://huggingface.co/datasets/facebook/flores). English to and from German, French, Italian and Spanish (8 directions). The dataset is gated: request access on the Hub and give the job an `HF_TOKEN`.

```bash
lm_eval --tasks flores200_chrf_eng_Latn-deu_Latn ...   # one direction
lm_eval --tasks flores200_chrf ...                     # all 8
```

Other languages: add a `flores200_chrf_<src>-<tgt>.yaml` using any `data/language/<lang>/devtest-00000-of-00001.parquet` file name.
