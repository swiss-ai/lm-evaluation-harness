# RTP-LX

## Paper

Title: RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios? (de Wynter et al., AAAI 2025)

Abstract: https://arxiv.org/abs/2404.14397

Homepage: https://github.com/microsoft/RTP-LX

RTP-LX is a human-transcreated, human-annotated multilingual version of a
subset of RealToxicityPrompts. It covers 38 languages in 40 locale files (v1.5).
Each locale has about 1k prompts. Each prompt carries three native-speaker annotations of
8 harm categories (Toxicity 1-5; IdentityAttack, Bias, Microaggression,
Violence, SelfHarm, Insult, SexualContent 1-3). Most v1.0 locales also contain
culture-specific, manually written prompts.

### Citation

```text
@article{rtplx,
  title={RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?},
  author={de Wynter, Adrian and Watts, Ishaan and others},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39}, number={27}, pages={27940--27950}, year={2025}
}
```

## Data

The data ships as a password-protected zip in the GitHub repo, so it cannot be
loaded from the HF Hub. Prepare it once with
`multilingual-harmfulness/benchmarks/rtp_lx/prepare_data.py`. That script pins v1.5,
checks the sha256, extracts the zip safely and writes `<LOCALE>.jsonl`.
The tasks read the data from `$RTP_LX_DATA_DIR`. The default is
`/capstor/scratch/cscs/alexandersternfeld/mh-data/rtp_lx/processed`.

## Tasks

### Generation (harmfulness of the model's continuations)

- Group `rtp_lx`, with one subtask per locale: `rtp_lx_<loc>`. The locales are
  `ar bcms bg ca cs cy da de de_ch el en es et fi fr he hi hu id it ja ko lt lv nl no_nb pl pt ro ru sk sl sv sw th tr uk vi zh_hans zh_hant`.
- The prompt is given as-is, with the chat template applied by the launcher. Decoding is greedy
  with `max_gen_toks: 512` and stops at `"\n\n"`. These settings match `realtoxicitypromptsllama`.
- Judge: `meta-llama/Llama-Guard-4-12B` (override with `RTP_LX_JUDGE_MODEL`). It sees the pair
  user = prompt, assistant = completion and classifies the assistant turn. For a judge that is
  not a Llama Guard model, the Llama Guard policy prompt is sent as text instead. Control this
  with `RTP_LX_JUDGE_FORMAT=auto|native|prompted`.
- Metrics:
  - `unsafe`: fraction of completions judged unsafe (lower is better).
  - `judge_error`: fraction of unparsable or failed judge calls (lower is better).
- Per-sample judge records go to `$LM_EVAL_JUDGE_LOG_DIR/rtp_lx_<loc>.jsonl`. Each record holds the
  prompt, the English source (`prompt_en`), the subset (transcreated or cultural), the dialect,
  the human prompt annotations (majority vote and raw), the response, and the judge output.

### Classification (the paper's original setup)

- Group `rtp_lx_judge`, with one subtask per locale: `rtp_lx_judge_<loc>`.
- The evaluated model annotates each prompt. It gets the official RTP-LX annotation prompt
  (`judge_prompt.py`, verbatim from `code/constants.py`) as the system message and the
  prompt as the user message. Decoding is greedy with at most 200 tokens.
- Output is parsed with a port of the official `parse_gpt4_output`. Gold labels are the official
  `vote()` over the 3 raw human labels.
- Metrics (higher is better unless noted):
  - `kappa_<category>`: linear-weighted Cohen's kappa per category.
  - `kappa_avg`: mean over the 8 categories, the paper's headline number.
  - `acc_avg` and `acc_toxicity`: exact-match accuracy.
  - `parse_error`: lower is better. Rows with parse errors are dropped from kappa and accuracy, as in the paper.

`generate_configs.py` regenerates all per-locale yamls, the groups and
`_rtp_lx_judge_yaml`.
