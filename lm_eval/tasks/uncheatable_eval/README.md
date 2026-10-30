# UncheatableEval

### Overview

UncheatableEval measures a language model's compression rate (bits per byte)
on freshly collected text -- recent arXiv/bioRxiv papers, GitHub code, BBC
news, Wikipedia and AO3 -- so the evaluation data postdates the training
cutoff of the models being tested and cannot have been memorized. There are
no questions or answers: the model's log-likelihood of each document is
scored with rolling loglikelihood, the same mechanism as `wikitext`.

Homepage: https://github.com/Jellyfish042/UncheatableEval

### Dataset

`Jellyfish042/UncheatableEval-2026-07` (pinned, 7,500 documents, 500 per
category). Newer monthly snapshots are published as
`Jellyfish042/UncheatableEval-<YYYY>-<MM>`; to evaluate a different
snapshot, change `dataset_path` in `_template_yaml`. Scores are only
comparable between models evaluated on the same snapshot.

### Groups and Tasks

#### Groups

* `uncheatable_eval`: all 15 categories below; metrics are size-weighted
  means of the per-category values.

#### Tasks

`uncheatable_eval_<category>` for category in `arxiv_cs`, `arxiv_math`,
`arxiv_physics`, `arxiv_other`, `biorxiv_all`, `bbc_news`,
`wikipedia_english`, `wikipedia_nonenglish`, `ao3_english`,
`ao3_nonenglish`, `github_cpp`, `github_javascript`, `github_markdown`,
`github_python`, `github_other`.

### Metrics

* `bits_per_byte` (lower is better): the compression rate, the headline
  number.
* `byte_perplexity` (lower is better): `2 ** bits_per_byte`.

`word_perplexity` is deliberately omitted: whitespace-delimited "words" are
not meaningful for code or non-English text.

Each document is scored on its `content` field (at most ~18k characters).
Requires a backend that supports `loglikelihood_rolling` (e.g. `hf`, `vllm`).

### Checklist

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original source that introduced the task?
* [x] Tested: `lm_eval --tasks uncheatable_eval --model hf --model_args pretrained=sshleifer/tiny-gpt2 --limit 2`
