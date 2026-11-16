# BOUQuET (sentence level, chrF / BLEU)

Reference-based chrF and BLEU on [facebook/bouquet](https://huggingface.co/datasets/facebook/bouquet), sentence-level test split (854 sentences per direction). English to and from German, French, Italian, Spanish and Romansh (10 directions). The dataset is gated: request access on the Hub and give the job an `HF_TOKEN`.

```bash
lm_eval --tasks bouquet_eng_Latn-deu_Latn ...   # one direction
lm_eval --tasks bouquet ...                     # all 10
```

More pairs: add a `bouquet_<src>-<tgt>.yaml` using any `benchmark_data/sentence_level/test/` file name.
