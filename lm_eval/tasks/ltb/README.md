# Last Translation Benchmark v1

`ltb_v1` evaluates translation generation on the official **LTBv1-eval** subset:
911 text-only examples with 1,693 human-written verification rules. It preserves
the upstream translation prompt and verifies each rule separately. Rules and
baseline translations are never included in the translation prompt.

- [Benchmark repository](https://github.com/zouharvi/last-translation-benchmark)
- [Dataset](https://huggingface.co/datasets/zouhar/last-translation-benchmark)
- [Paper](https://arxiv.org/abs/2609.04173)

The YAML config uses `dataset_path: zouhar/last-translation-benchmark` and loads
through Hugging Face Datasets. The release is pinned to dataset revision
`a483825ddbe2d7756f5bdfb1e4f611bee9026c4c`, file `data/v1.json`.
The Hub exposes the release as the `train` split; this task uses that split for
testing, selecting the `LTBv1-eval` tag. The full LTBv1 release also contains media
and examples outside the evaluation subset. Evaluation is zero-shot.

## Running

Install the model backend and the optional judge client:

```bash
pip install -e '.[hf,ltb]'
export LTB_JUDGE_API_KEY='your-api-key'
lm_eval --model hf --model_args pretrained=YOUR_MODEL \
  --tasks ltb_v1 --apply_chat_template --batch_size auto \
  --log_samples --output_path results/ltb_v1
```

The default judge is `google/gemini-3.1-pro-preview`, matching the upstream
leaderboard's judge model, served through OpenRouter. The judge receives the
source, generated translation, and one rule per request. Judge calls are made
only during scoring. Environment variables configure an alternative
OpenAI-compatible chat-completions service:

| Variable | Default |
| --- | --- |
| `LTB_JUDGE_API_KEY` | Required for scoring |
| `LTB_JUDGE_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LTB_JUDGE_MODEL` | `google/gemini-3.1-pro-preview` |
| `LTB_JUDGE_WORKERS` | `8` concurrent examples |
| `LTB_OUTPUT_DIR` | New `ltb-results-*` directory in the current directory |
| `LTB_RESPONSE_FORMAT` | `text`; use `harmony` for raw gpt-oss output |

For gpt-oss with vLLM, `LTB_RESPONSE_FORMAT=harmony` extracts the final channel
for judging and submission export. Leave the backend's `think_end_token` unset
and preserve special tokens so this task can recognize the channel boundaries.
Raw responses, including analysis, remain in the harness samples. Responses
without a final channel become `null` translations and fail instead of sending
unfinished reasoning to the judge. This extraction runs in `process_results`,
so it does not apply to `--predict_only` output.

Scoring saves `ltb_submission.json` before making judge requests and writes
`judge_responses.jsonl` after every completed request. The JSONL records contain
the dataset ID, rule index, exact request, full API response (including reasoning
and token usage when returned), parsed verdict, and per-example score. API errors
record the exception type and HTTP status; truncated responses are saved before
scoring aborts. Each record is flushed to disk so completed calls survive a later
timeout. Credentials and request headers are not logged. Existing judge logs are
never overwritten; use a fresh output directory for each evaluation.

The `evals-post-train/scripts/evaluate_ltb.sbatch` launcher sets `LTB_OUTPUT_DIR`
to the run's `harness` directory automatically. For a direct harness invocation,
set it explicitly to place these files alongside your other results. This logging
runs during scoring; `--predict_only` continues to use the harness sample files.

Use `--limit 5` for a small trial. A full evaluation makes up to 1,693 judge
requests, which may incur API charges. Generation allows 4,096 tokens without
newline stop sequences, preserving multi-paragraph translations. Adjust the
token limit through `--gen_kwargs max_gen_toks=8192` if needed.

To generate and save translations without calling a judge, add `--predict_only`
and provide `--output_path`. Once the Hugging Face dataset is cached, set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` to load it without network access.

## Metric

`ltb_pass_rate` is the fraction of examples for which **all** verification rules
pass, on a 0–1 scale. Each example has equal weight regardless of rule count.
Empty translations fail. Verdict parsing follows the official leaderboard's
permissive pass/fail parsing, including treating unrecognized responses as fail.
API exceptions after retries and truncated judge responses abort scoring so that
service failures do not silently alter the score or denominator.
On failure, scheduling stops, pending examples are cancelled, and workers skip
remaining rules. Requests already in flight finish (including SDK retries) and
are logged before the judge client and logs close; aborting is not instantaneous.

Changing the judge changes the evaluation protocol; report the judge model and
endpoint with results. These locally computed scores are not leaderboard
submissions. The official scorer uses its own API/cache infrastructure, so even
the default model does not guarantee identical verdicts across runs.

## Attribution

Dataset: CC BY 4.0. Upstream prompt and parser code: MIT, copyright (c) 2026
Vilém Zouhar. See [LICENSE](LICENSE) for the upstream code notice.

```bibtex
@misc{zouhar2026translationbenchmark,
  title={Last Translation Benchmark},
  author={LTB},
  year={2026},
  eprint={2609.04173},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
