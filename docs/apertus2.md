# Apertus 2 native chat format (`apertus2-completions`)

`apertus2-completions` renders native prompts and parses outputs client-side
with [`apertus-common`](https://github.com/swiss-ai/apertus-common), without
a Jinja chat template. It sends raw text to a `/v1/completions`
endpoint, such as vLLM serving the checkpoint and its instruct tokenizer. 
Tasks still see a chat model: messages in, reply plus tool calls out.
Apertus 2 specific logic is contained in the model class.

## Quick start

```bash
lm_eval --model apertus2-completions \
  --model_args base_url=http://host:8000/v1/completions,model=$SERVED_NAME,tokenizer=/path/to/checkpoint,max_model_len=65536,num_concurrent=32 \
  --tasks gsm8k_cot,arc_challenge_chat --apply_chat_template \
  --gen_kwargs max_gen_toks=32768 --log_samples -o out/
```

`tokenizer` must be the tokenizer the checkpoint was post-trained with (an
`Apertus_2_instruct` artifact); the server must serve the same files.
`max_model_len` is the server's context length.
`--apply_chat_template` is required (few-shot examples become turns).

## How it works

1. **Prompt.** `apertus_common.from_openai` maps task messages and the `tools`
   gen kwarg. System messages and task descriptions set `behavior`; reasoning
   effort sets `thinking`. Send BOS plus `render_str(conversation)`; literal
   special-token spellings can change re-tokenization and are counted.
2. **Request.** `skip_special_tokens: false` preserves output framing;
   `return_token_ids: true` enables exact parsing. `stop_token_ids` holds the
   wait token. Apply task `until` strings after parsing to avoid cutting reasoning.
3. **Response.** Parse think, reply, call and claim messages. Score only the
   reply, cut at the first task `until` string; append tool calls as OpenAI JSON
   after `TOOL_CALLS_MARKER`, matching `local-chat-completions`
   (`bfcl_v3_fc` works unchanged). Truncated replies keep their partial answer;
   truncated thinking yields an empty answer.

The native format separates reasoning; `think_start_token`/`think_end_token`
do not apply.

## Answer prefills

Tasks with `gen_prefix` (`arc_challenge_chat`, `bbh` cot, `drop`,
`humaneval_instruct`, `mbpp_instruct`) supply an assistant prefix. It becomes
an open reply (`<|out|>reply<|hdr|>PREFIX`, no close); tasks score its continuation.

| `prefill`               | Behaviour                                                                                                                                               |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `after_think` (default) | Generate until the first message closes; if it is a think, a second request continues the prefixed reply. Matches Apertus 2 training.                   |
| `empty_think`           | One request with an empty think before the open reply, like Qwen3's no-thinking prefill. For comparison with models that skip reasoning under prefills. |
| `reject`                | A prefill request raises before any request is sent.                                                                                                    |

Without a prefix, both modes use one request. If `after_think` gets a reply
first, return it and set `generation_prefill_skipped`. Call 2 shares call 1's
token budget and context allowance; no remaining budget yields an empty,
truncated answer.

## Token budgets

| Limit                      | Counts                                | Set by                    | At the limit                                                                                                  |
|----------------------------|---------------------------------------|---------------------------|---------------------------------------------------------------------------------------------------------------|
| `max_tokens` (per request) | Generated tokens: thinking plus reply | The task's `max_gen_toks` | Generation stops (`generation_truncated`); inside thinking the answer is empty (`generation_think_truncated`) |
| vLLM `--max-model-len`     | Prompt plus generated tokens          | Server start              | HTTP 400; the run aborts after retries                                                                        |

On harness side the budget comes from, highest first:

1. A task's `max_tokens` gen kwarg (no suite task sets one)
2. `--gen_kwargs max_gen_toks=N`, applied to every task
3. The task's `max_gen_toks` (BFCL raises it to at least 2048)
4. The `max_gen_toks` model argument (default 256)

Task budgets assume direct answers (`arc_challenge_chat` 100, `ifeval` 1280).
Apertus 2 always thinks first: pass `--gen_kwargs max_gen_toks=32768`, as for
other thinking models. Set `max_model_len` to the server's `--max-model-len`
(e.g. 65536 for that budget) so over-long requests are clamped or skipped
instead of failing. Do not set a server-wide `max_new_tokens`
(`--override-generation-config` or a `--generation-config` directory): vLLM
caps every request at it.

## Model arguments

Besides the `local-chat-completions` arguments (`base_url`, `model`,
`num_concurrent`, `max_retries`, `timeout`, ...):

| Argument                 | Default       | Meaning                                                                                                                                  |
|--------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `tokenizer`              | required      | Checkpoint tokenizer directory                                                                                                           |
| `reasoning_effort`       | `medium`      | `low`, `medium` or `high`; also read from `chat_template_args`. `enable_thinking=false` is an error.                                     |
| `prefill`                | `after_think` | See [answer prefills](#answer-prefills)                                                                                                  |
| `literal_special_tokens` | `flag`        | `flag` counts special-token text in requests; `reject` raises for text the task wrote (system, user and tool messages, tool definitions) |
| `mask_special_tokens`    | `false`       | Send a `logit_bias` blocking every special token except the output controls                                                              |
| `return_token_ids`       | `true`        | Ask for token IDs; disable if the endpoint rejects the field (text parsing then misreads control spellings written as ordinary text)     |
| `max_model_len`          | unset         | Server `--max-model-len`: clamp `max_tokens` or skip generation if the prompt fills the context; see [token budgets](#token-budgets)     |
| `log_native`             | `output`      | `output` stores the raw native output and a hash of the first request's prompt per sample; `full` stores that prompt itself              |

Likelihood and multiple-choice tasks are not supported and raise.

## Per-sample info and metrics

Responses carry `length_info`, saved in sample JSONL with `--log_samples`.
Promotion copies numeric `generation_*`/`thinking_format_*` fields into benchmark
metrics: average repeats per document, then documents providing that value.
This runs without sample logging. Promoting length fields requires
`--log_length_metrics`; `apertus2_*` strings stay sample-only.

Handled failures remain in scoring with empty/partial answers; their integer
flags become failure rates. Fatal errors abort before normal promotion and
result saving.

| Failure / flag                                       | Answer or run outcome                     | Reporting                                                                          |
|------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------------------|
| Request construction: `generation_request_failed`    | Empty answer; no POST                     | Flag; warning with exception text                                                  |
| Output mapping: `generation_map_failed`              | Empty answer                              | Flag; warning with exception text                                                  |
| Malformed format/response: `generation_parse_failed` | Keep completed messages                   | Flag and raw output when available; parser exception text neither saved nor logged |
| Context overflow: `generation_prompt_too_long`       | Empty answer; no POST                     | Flag                                                                               |
| Thinking cut off: `generation_think_truncated`       | Empty answer                              | Flag and raw output                                                                |
| HTTP errors/timeouts                                 | Retry; abort after `max_retries` attempts | Logs/raised exception; no HTTP failure flag                                        |
| Configuration error                                  | Fail before requests                      | Raised exception                                                                   |

| Key                                                            | Meaning                                                                                                                               |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `generation_truncated`                                         | Stopped at the token budget                                                                                                           |
| `generation_prefill_skipped`                                   | `after_think`, but the model did not think first                                                                                      |
| `generation_budget_clamped`                                    | `max_model_len` check lowered `max_tokens`                                                                                            |
| `generation_literal_special_tokens`                            | Special-token spellings in the request content                                                                                        |
| `generation_prompt_mismatch`                                   | Returned prompt IDs differ from native IDs; only when the server returns them                                                         |
| `generation_unknown_output` / `generation_has_claim`           | Unknown output types / claims, both dropped                                                                                           |
| `generation_call_counter_reused`, `generation_unexpected_call` | Tool calls dropped: reused call counter, or calls without `tools` in the request                                                      |
| `thinking_format_*`, `thinking_length_*`, `response_length_*`  | Parsed payloads, excluding controls, harness prefills/empty think and tool calls; thinking length only when `thinking_format_correct` |
| `apertus2_prefill_mode`, `apertus2_output`, `apertus2_prompt`  | Strings: prefill mode, raw native output, prompt (or its hash)                                                                        |

## Limitations

- Earlier reasoning is not replayed in multi-turn histories (as for chat
  models); the native profile retains it.
- Claims are dropped, not scored.
- `--use_cache` keys ignore the model: use one cache path per model
  configuration (tokenizer, effort, prefill, masking, `max_model_len`).
- Sampled repeats share the request `seed` (existing harness behaviour).

## Setup and tests

Requires `apertus-common` with open output messages (commit `e0c1581`, branch
`feat/open-output-messages`, pending merge into `apertus2`). The package declares
Python >= 3.13 and tokenizers 0.23; it also works on Python 3.12 with tokenizers
0.22.2 (validated), installed with:

```bash
pip install --no-deps --ignore-requires-python \
  "apertus-common @ git+https://github.com/swiss-ai/apertus-common@e0c1581"
```

plus `pydantic>=2.13.4` and `jsonschema>=4.26.0` if missing.
`tests/models/test_apertus2.py` runs with a synthetic tokenizer; set
`APERTUS2_TOKENIZER` to an `Apertus_2_instruct` directory for the real-artifact
checks.
