# Arena-Hard v0.1

### Paper

- Title: *From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline*
- Authors: Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica
- Abstract: arXiv preprint arXiv:2406.11939
- Homepage: https://lmsys.org/blog/2024-04-19-arena-hard/

The rapid evolution of Large Language Models (LLMs) has outpaced the
development of model evaluation, highlighting the need for continuous curation
of new, challenging benchmarks. However, manual curation of high-quality,
human-aligned benchmarks is expensive and time-consuming. To address this, we
introduce BenchBuilder, an automated pipeline that leverages LLMs to curate
high-quality, open-ended prompts from large, crowd-sourced datasets, enabling
continuous benchmark updates without human in the loop. We apply BenchBuilder
to datasets such as Chatbot Arena and WildChat-1M, extracting challenging
prompts and utilizing LLM-as-a-Judge for automatic model evaluation. To
validate benchmark quality, we propose new metrics to measure a benchmark's
alignment with human preferences and ability to separate models. We release
Arena-Hard-Auto, a benchmark consisting of 500 challenging prompts curated by
BenchBuilder. Arena-Hard-Auto provides 3x higher separation of model
performances compared to MT-Bench and achieves 98.6% correlation with human
preference rankings, all at a cost of $20. Our work sets a new framework for
the scalable curation of automated benchmarks from extensive data.

### Citation
```bibtex
@article{li2024crowdsourced,
  title={From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline},
  author={Li, Tianle and Chiang, Wei-Lin and Frick, Evan and Dunlap, Lisa and Wu, Tianhao and Zhu, Banghua and Gonzalez, Joseph E and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.11939},
  year={2024}
}
@misc{arenahard2024,
    title = {From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline},
    url = {https://lmsys.org/blog/2024-04-19-arena-hard/},
    author = {Tianle Li*, Wei-Lin Chiang*, Evan Frick, Lisa Dunlap, Banghua Zhu, Joseph E. Gonzalez, Ion Stoica},
    month = {April},
    year = {2024}
}
```

### Implementation Details
Arena-Hard v0.1 is an automatic LLM benchmark consisting of 500 challenging
user queries sourced from Chatbot Arena.  It measures model quality through
**pairwise comparison**: each model response is judged head-to-head against a
GPT-4-0314 baseline by a strong judge LLM.

In this integration the judge is **Llama-3.3-70B-Instruct** served via the
CSCS SwissAI API (same endpoint as the AlpacaEval task).

1. **Generation** — the 500 Arena-Hard prompts are presented as user turns.
   lm-eval's `--apply_chat_template` wraps them in the model's chat format.
2. **Two-round judging** — for each question the judge sees:
   - Round 1: baseline (GPT-4-0314) = Assistant A, model = Assistant B
   - Round 2: model = Assistant A, baseline = Assistant B (position swap)
3. **Verdict extraction** — the judge outputs `[[A>>B]]`, `[[A>B]]`, `[[A=B]]`,
   `[[B>A]]`, or `[[B>>A]]`.  Strong preferences (`>>` / `<<`) receive 3×
   weight.
4. **Scoring** — bootstrap resampling (100 rounds) over weighted scores yields
   a win rate (0–100%) with 90% confidence interval.

### Dataset

Loaded directly from HuggingFace:

- Questions: [`lmarena-ai/arena-hard-auto`](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto) (`data/arena-hard-v0.1/question.jsonl`)
- GPT-4-0314 baseline answers: same repo (`data/arena-hard-v0.1/model_answer/gpt-4-0314.jsonl`)

### Groups, Tags, and Tasks

#### Groups

* `group_name`: `Short description`

#### Tags

* `tag_name`: `Short description`

#### Tasks

- `arena_hard_v01`: 500 Arena-Hard v0.1 prompts

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
Changed to Qwen 3.5-27B judge. Make sure this model is served on the CSCS platform to be called as judge.