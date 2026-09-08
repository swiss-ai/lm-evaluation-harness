# AlpacaEval with an explicit judge

This task generates responses to the AlpacaEval instructions and compares them
with the GPT-4 baseline in
[`tatsu-lab/alpaca_eval`](https://huggingface.co/datasets/tatsu-lab/alpaca_eval/blob/main/alpaca_eval_gpt4_baseline.json).
It calls the official
[`alpaca_eval`](https://github.com/tatsu-lab/alpaca_eval) scorer's
`get_length_controlled_winrate` metric.

The harness reports **a fraction from 0 to 1**: an official scorer result of
37.5% becomes `0.375`. The saved official leaderboard retains percentage units.
`avg_word_count` counts whitespace-separated words in the completion presented
to the judge.

## Setup and judge selection

Install the optional scorer dependencies:

```bash
pip install 'lm_eval[alpaca_eval]'
```

The adapter is validated against `alpaca-eval==0.6.6`. Its `pkg_resources`
import requires `setuptools<81`; the extra includes that compatibility bound.
The scorer and its dependencies load only during aggregation, so listing tasks
and preparing generated records does not require this extra.

Set an explicit annotator configuration before running the task:

```bash
export ALPACA_EVAL_ANNOTATORS_CONFIG=weighted_alpaca_eval_gpt4_turbo
export ALPACA_EVAL_OUTPUT_DIR=/path/to/evaluation/alpaca_artifacts
lm_eval --model hf --model_args pretrained=/path/to/model \
  --tasks alpaca_eval --apply_chat_template --output_path /path/to/evaluation/results
```

This example selects the official weighted GPT-4 Turbo configuration, whose
model is `gpt-4-1106-preview`. Availability and credentials must be configured
for that model. The environment variable can instead point to an official
AlpacaEval YAML file or directory containing `configs.yaml`. Custom prompt paths
follow the official scorer's path resolution; absolute paths avoid ambiguity.
Use the scorer's own decoder/client configuration for credentials, endpoints,
retries and concurrency. The adapter does not choose an endpoint, copy
credentials between environment variables, or install configuration files into
the scorer package.

**Different annotators define different evaluation protocols.** Choosing another
judge, prompt, parser, or decoding configuration does not reproduce the weighted
GPT-4 Turbo protocol merely because the metric has the same name. The imported
Swiss task used a Llama-3.3-70B judge served at CSCS; that service is not an
implicit default in this port. To reproduce that judge setup, supply its
configuration explicitly.

## Records and artifacts

The instruction prompt, GPT-4 reference dataset, deterministic generation
settings and 16,384-token generation cap match the Swiss source. Complete
`<think>...</think>` blocks are removed before judging. When the opening tag was
in the prompt, leading text through `</think>` is also removed. Whitespace is
trimmed. Empty completions are retained, with a word count of zero. Raw
completions remain in the collected records.

Each aggregation creates a unique persistent `run-*` directory under
`ALPACA_EVAL_OUTPUT_DIR`, or `./alpaca_eval_results` if it is unset. The log
reports that path. The directory contains:

- `collected_outputs.json`: raw and cleaned completions with source references.
- `model_outputs.json` and `reference_outputs.json`: the actual scorer inputs.
- `provenance.json`: scorer version, annotator names and model identities,
  configuration path and SHA-256, reference dataset/generators, sample count,
  preprocessing, status and both score units. Client credentials are omitted.
- Official scorer artifacts: annotations, annotation cache, leaderboard and GLM
  weights, when produced. Weights are directed here rather than written into
  the installed package.

Keep the original annotator configuration and any custom prompt files with the
experiment configuration; provenance records their identity without copying
potential credential fields. Missing, nonnumeric, nonfinite or out-of-range
scorer results raise an error rather than becoming a zero score. Failed scoring
runs retain their input records and a failed provenance status.

The official GLM can download its calibration data from Hugging Face. Full
benchmark generation, dataset access, judge API compatibility, and GLM scoring
require runtime validation in the evaluation environment. Use the full original
dataset order for comparable length-controlled scores; subset smoke tests do
not establish benchmark performance.

## Source and validation

Ported from Swiss AI commit
`51d6f4b62bf20e9a29dffa694b163d4f19889927` onto upstream harness v0.4.13.
Compatibility changes remove the Swiss model resolver, rate limiter and
hardcoded service; require explicit judge configuration; preserve the dataset's
reference generator identity and raw completion; and retain scoring artifacts.
The source harness's fraction scale is preserved.

Offline tests exercise record preparation, empty/thinking completions, real
configuration loading, scorer arguments, artifacts and invalid score handling.
They replace the external `alpaca_eval.evaluate` call, and do not contact a
judge. Run them with:

```bash
python -m pytest tests/test_alpaca_eval_task.py -q
```

Reference: [Length-Controlled AlpacaEval: A Simple Way to Debias Automatic
Evaluators](https://arxiv.org/abs/2404.04475).
