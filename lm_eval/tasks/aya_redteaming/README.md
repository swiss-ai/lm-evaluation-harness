# Aya Redteaming

## Paper

Title: `The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm`

Abstract: https://arxiv.org/abs/2406.18682

Homepage: https://huggingface.co/datasets/CohereLabs/aya_redteaming

Human-annotated multilingual red-teaming dataset with 7,419 harmful prompts in 8 languages and 9 harm categories. Each prompt is labelled as "global" or "local" harm.

### Citation

```text
@misc{aakanksha2024multilingualalignmentprismaligning,
      title={The Multilingual Alignment Prism: Aligning Global and Local Preferences to Reduce Harm},
      author={Aakanksha and Arash Ahmadian and Beyza Ermis and Seraphina Goldfarb-Tarrant and Julia Kreutzer and Marzieh Fadaee and Sara Hooker},
      year={2024},
      eprint={2406.18682},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.18682},
}
```

### Implementation

Each prompt is sent unchanged, in its original language, as one user turn. The launcher applies the chat template, and decoding is greedy with up to 512 new tokens. A judge LLM then answers a binary question about whether the completion is harmful. The paper calls this "standalone" grading (Appendix F).

* **Judge.** The paper used GPT-4. Here the default is `Qwen/Qwen3.8-27B` on the CSCS serving API, with thinking disabled (`AYA_REDTEAMING_JUDGE_THINKING=1` turns it on). Llama-3.3-70B-Instruct, the previous default, is no longer served. Override the judge with `AYA_REDTEAMING_JUDGE_MODEL`. The `CSCS_SERVING_API` key is required.
* **Judge prompt.** The paper does not print its GPT-4 prompt. We use the harmfulness-classification prompt that the same group published for the same GPT-4 setup in the Aya model paper (Üstün et al. 2024, arXiv:2402.07827, Appendix J.4). The `{language}` placeholder is filled with the dataset's `language` field.
* **Parsing.** The answer is read as 'Yes' (harmful) or 'No'. Light decoration and `<think>` blocks are tolerated. Anything ambiguous counts as unparseable.
* **Failures.** API failures and unparseable answers are left out of the harmful rates. They are never counted as safe, and are reported as `judge_failure_rate`.
* **Judge logs.** One JSONL row per sample is written to `$LM_EVAL_JUDGE_LOG_DIR/aya_redteaming_<lang>.jsonl`. Each row has the prompt, its English translation, the harm categories, the global/local flag, the response, the raw judge output and the label.

Environment knobs:

* `AYA_REDTEAMING_JUDGE_MAX_TOKENS` (default 50)
* `AYA_REDTEAMING_JUDGE_WORKERS` (default 16)
* `AYA_REDTEAMING_JUDGE_MAX_RETRIES` (default 10)
* `AYA_REDTEAMING_SKIP_JUDGE_CHECK=1` skips the connectivity check that otherwise runs when the task is loaded.

Regenerate the yamls with `python _generate_configs.py`.

### Metrics

All metrics are fractions in [0, 1], and lower is better for all of them (`higher_is_better: false`).

* `score`: the share of judged completions rated harmful. This is the main metric, used in the Apertus main table.
* `harmful_rate`: the same value as `score`, under a clearer name.
* `harmful_rate_global`, `harmful_rate_local`: the same rate, restricted to the "global" or "local" prompts.
* `harmful_rate_<category>`: the same rate per harm category. Prompts can carry several categories, so the categories overlap. The slugs are:
  * `bullying_harassment`
  * `discrimination_injustice`
  * `graphic_material`
  * `harms_of_representation`
  * `hate_speech`
  * `non_consensual_sexual_content`
  * `profanity`
  * `self_harm`
  * `violence_threats_incitement`
* `judge_failure_rate`: the share of samples with no usable verdict.

The group values are size-weighted means over languages. For the subset metrics, the weights are language sizes, not subset sizes.

### Groups, Tags, and Tasks

#### Groups

* `aya_redteaming`: all 8 languages

#### Tasks

* `aya_redteaming_arabic`: Arabic prompts (900)
* `aya_redteaming_english`: English prompts (987)
* `aya_redteaming_filipino`: Filipino prompts (1009)
* `aya_redteaming_french`: French prompts (813)
* `aya_redteaming_hindi`: Hindi prompts (915)
* `aya_redteaming_russian`: Russian prompts (1007)
* `aya_redteaming_serbian`: Serbian prompts (1006)
* `aya_redteaming_spanish`: Spanish prompts (782)

### Changelog

* 2.0:
  * Fixed the judge prompt. The earlier transcription mixed up the speakers ("Person A's response").
  * Failed or unparseable verdicts no longer vanish silently, and an all-failed run now gives NaN instead of 0.
  * Added the breakdown metrics, `judge_failure_rate` and the per-sample judge logs.
  * Scores from 1.0 are not comparable with 2.0.

### Checklist

For adding novel benchmarks/datasets to the library:

* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?
    * The paper releases no evaluation code. The judge prompt follows the group's published prompt (see above).
