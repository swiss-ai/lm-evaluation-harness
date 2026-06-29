# Thinking / reasoning evals

How the harness handles reasoning ("thinking") models on generative tasks: the
reasoning-trace strip, and the per-response **length** and **thinking-format**
metrics it records.

For the CLI flags and `--model_args` keys that control all of this, see the
[CLI reference](./interface.md#generation-length--thinking-format-metrics). This
page is the conceptual reference for what those options do.

## Reasoning tokens and the strip

Reasoning models wrap their deliberation in an **open**/**close** token pair
(e.g. `<think>`/`</think>`, or `<|inner_prefix|>`/`<|inner_suffix|>`). Before
answer extraction, the `hf`, `vllm`, and `sglang` backends **strip** everything
up to and including the close token, so only the post-reasoning answer is scored.

- The tokens are **auto-detected** from the model's chat template (`inner_token` /
  `outer_token` declarations), or set explicitly via the `think_start_token` /
  `think_end_token` `--model_args`.
- The **close** token (`think_end_token`) is what drives the strip; on `hf` it may
  also be an integer token id.
- **Fail-loud:** when thinking is enabled but the close token cannot be resolved,
  model construction raises — pass `think_end_token=` explicitly, or set
  `enable_thinking=false`. This prevents silently scoring un-stripped reasoning.
  (On `hf` the guard defaults to enabled, so a reasoning-declaring template with an
  unresolvable close raises even if you never set `enable_thinking`.) A missing
  **open** token only warns — `has_open`/`correct` are then not tracked unless you
  pass `think_start_token=`.

> If the close token is wrong/unresolved, the reasoning is **not** stripped and
> pollutes answer extraction. The metrics below (`thinking_format_has_close`,
> `thinking_length_*`) make that visible.

## The metrics

For `generate_until` (and `multi_turn_generate`) tasks, each response is measured
on the **raw generation, before the strip**, and recorded per response in the
sample's `length_info` field (visible with `--log_samples`). Two families:

**Length** — `response_length_*` is always measured; the `thinking_length_*` pair
is added only when the response contains the reasoning close token:

| Metric | Meaning |
|--------|---------|
| `response_length_words` / `_chars` / `_tokens` | Length of the full generation (reasoning + answer). `_tokens` uses the exact generated token ids, or an exact backend token count (e.g. SGLang's `completion_tokens`), when available — otherwise the `_tokens` variant is omitted for that response. |
| `thinking_length_words` / `_chars` / `_tokens` | Length of the reasoning span (start of generation up to and including the close token). `_tokens` re-encodes the span with the tokenizer (best-effort). |

**Thinking-format** (`thinking_format_*`, 0/1 per response) — a well-formedness
signal for reasoning models:

| Metric | Meaning |
|--------|---------|
| `thinking_format_has_open` | The reasoning open token is present in the prompt **or** generation (thinking templates usually prefill the open into the prompt). |
| `thinking_format_has_close` | The reasoning close token is present in the generation (the model emitted it). |
| `thinking_format_correct` | Open present, close present, open precedes close, and the block is not re-opened after the close. |

## What is logged, where, and under which flags

| Sink | Flag | Length (`response_length_*`, `thinking_length_*`) | Format (`thinking_format_*`) |
|------|------|---------------------------------------------------|------------------------------|
| Per-sample, on disk (`length_info` in `samples_*.jsonl`) | `--log_samples` | yes | yes |
| Per-task aggregate (`results.json` + printed table) | length: `--log_length_metrics`; format: none | only with the flag | always (when token resolved) |
| W&B (native, by the harness) | `--wandb_args` (uploads whatever was aggregated) | only with `--log_length_metrics` | always (when token resolved) |

Aggregation is a per-task **mean over documents**. "Always (when token resolved)"
means `thinking_format_*` needs no `--log_length_metrics`, but each flag is only
produced when its token resolved: `has_open`/`correct` require the open token,
`has_close`/`correct` the close token (see fail-loud above).

**Aggregated key naming.** Aggregated metrics carry the `none` filter. In
`results.json` the key is `thinking_format_correct,none` (nested under the task,
e.g. `results["gsm8k"]`). In **W&B** the trailing `,none` is stripped, so it
appears as `gsm8k/thinking_format_correct`.

**Multi-turn (`multi_turn_generate`).** Each turn is a separate `generate_until`
response: the strip and the per-response length/format are measured **per turn**,
and the task creates **one `Instance` per turn**. At aggregation, all turns of a
document are grouped by `doc_id` and averaged into a **single per-doc value**
(so a 5-turn doc weighs the same as a 1-turn doc, not 5×).

## How length is handled for malformed or non-thinking responses

This describes the **aggregated** numbers (`--log_length_metrics`); the raw
per-sample `length_info` always reflects exactly what was measured. The aggregator
pairs each `thinking_length_<unit>` with its `response_length_<unit>` on an
identical per-response basis, so the two always have the same sample count and
thinking length can never exceed response length:

- **No thinking at all** (non-reasoning model/run, or `enable_thinking=false`):
  `response_length_*` is measured normally; `thinking_length_*` is **0** for every
  response (it is still emitted, paired with response length — not omitted).
- **Malformed — opened but never closed** (`thinking_format_has_close = 0`): there
  is no close token to bound the reasoning span, so `thinking_length_*` counts as
  **0** for that response (and `thinking_format_correct = 0` flags it). The full
  reasoning text still counts toward `response_length_*`, and — because the strip
  keys off the close token — an unclosed block is **not** stripped, so the
  reasoning remains in the scored text for that response.
- **Per-doc / repeats:** the per-task value is the mean over documents; within a
  doc with multiple samples, each is averaged, so a doc that closed on some
  samples and not others gets a fractional `thinking_length` (e.g. `mean(90, 0)`).
- **Measurement gaps:** a response carrying a `thinking_length_<unit>` with no
  matching `response_length_<unit>` is dropped for that unit, keeping the bases
  aligned. Consequence: the `_tokens` family can have a **smaller sample base**
  than `_words`/`_chars` on backends that don't expose token counts for every
  response.

### Impact summary (direction of bias)

As the share of **malformed (opened-but-not-closed)** responses rises:

| Quantity | Direction | Why |
|----------|-----------|-----|
| `thinking_format_has_close`, `thinking_format_correct` | **down** | flagged 0 per malformed response — this is the signal you want |
| `thinking_length_*` (aggregate) | **deflated** | malformed responses contribute 0 to the mean |
| `response_length_*` | **unaffected / relatively high** | still counts the full text, reasoning included |
| **task score** | **likely deflated** | the un-stripped reasoning stays in the scored text and pollutes answer extraction |

For a **non-thinking** run all `thinking_*` metrics sit at 0 by construction
(expected, not a problem); `response_length_*` is normal. `thinking_format_has_open`
can instead be **inflated** under few-shot / multi-turn (see Caveats).

## Caveats

- **Answer extraction.** A reasoning model often answers in its own style after
  the close token (a bare number, `\boxed{...}`), which a strict
  format-specific extractor (e.g. gsm8k's `strict-match` `#### N`) will miss even
  though the answer is correct. Prefer a robust extractor (`flexible-extract`, or
  a CoT/zero-shot task variant). `thinking_format_*` + the length split help
  distinguish "reasoned fine, extractor too strict" from a real thinking failure.
- **`thinking_format_has_open` / `correct` over-count** under **few-shot** (demo
  think tags in the prompt) and **multi-turn** (a prior turn's unclosed open leaks
  into the next turn's rendered history).
