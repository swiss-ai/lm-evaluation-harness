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
  unresolvable close raises even if you never set `enable_thinking`. `sglang` has no
  `enable_thinking` toggle — a reasoning-declaring template is always thinking-enabled.)
  A missing **open** token only warns — `has_open` is then not tracked and `correct`
  falls back to `has_close` (close-only style) unless you pass `think_start_token=`.

> If the close token is wrong/unresolved, the reasoning is **not** stripped and
> pollutes answer extraction. The metrics below (`thinking_format_has_close`,
> `thinking_length_*`) make that visible.

## The metrics

For `generate_until` (and `multi_turn_generate`) tasks, each response is measured
on the **raw generation, before the strip**, and recorded per response in the
sample's `length_info` field (visible with `--log_samples`). Two families:

**Length** — `response_length_*` is always measured; the `thinking_length_*` pair
is added only for **well-formed** responses (`thinking_format_correct = 1`), so a
mis-attributed span (e.g. a close inherited from an earlier turn) is never measured:

| Metric | Meaning |
|--------|---------|
| `response_length_words` / `_chars` / `_tokens` | Length of the full generation (reasoning + answer). `_tokens` uses the exact generated token ids, or an exact backend token count (e.g. SGLang's `completion_tokens`), when available — otherwise the `_tokens` variant is omitted for that response. |
| `thinking_length_words` / `_chars` / `_tokens` | Length of the reasoning span (start of generation up to and including the **last** close token — the same boundary the strip uses). Recorded only when `thinking_format_correct = 1`. `_tokens` is an exact token count on the `hf` int-token path, else a best-effort tokenizer re-encode of the span. |

**Thinking-format** (`thinking_format_*`, 0/1 per response) — a well-formedness
signal for reasoning models, computed on the **current turn only** (the rendered
history is never consulted, so a prior turn cannot leak into the signal):

| Metric | Meaning |
|--------|---------|
| `thinking_format_has_open` | The block was opened **this turn**: the open token is emitted in the generation, or injected by the template's generation prompt (a prefill template). Emitted only when an open token is known. |
| `thinking_format_has_close` | The reasoning close token is present in the generation (the model emitted it). |
| `thinking_format_correct` | **Open+close models:** open present (this turn), close present, open precedes the (last) close, block not re-opened after the (first) close. **Close-only models** (no open token): reduces to `has_close`. |

## What is logged, where, and under which flags

| Sink | Flag | Length (`response_length_*`, `thinking_length_*`) | Format (`thinking_format_*`) |
|------|------|---------------------------------------------------|------------------------------|
| Per-sample, on disk (`length_info` in `samples_*.jsonl`) | `--log_samples` | yes | yes |
| Per-task aggregate (`results.json` + printed table) | length: `--log_length_metrics`; format: none | only with the flag | always (when token resolved) |
| W&B (native, by the harness) | `--wandb_args` (uploads whatever was aggregated) | only with `--log_length_metrics` | always (when token resolved) |

Aggregation is a per-task **mean over documents**. "Always (when token resolved)"
means `thinking_format_*` needs no `--log_length_metrics`, but the flags are only
produced when **thinking is enabled** and the relevant token resolved: `has_open`
requires the open token; `has_close` and `correct` require the close token (`correct`
is emitted whenever the close is known — for close-only models it equals `has_close`;
see fail-loud above). With `enable_thinking=false` no `thinking_*` metric is emitted —
only `response_length_*`.

**Aggregated key naming.** Aggregated metrics carry the `none` filter. In
`results.json` the key is `thinking_format_correct,none` (nested under the task,
e.g. `results["gsm8k"]`). In **W&B** the trailing `,none` is stripped, so it
appears as `gsm8k/thinking_format_correct`.

**Multi-turn (`multi_turn_generate`).** Each turn is a separate `generate_until`
response: the strip and the per-response length/format are measured **per turn**,
and the task creates **one `Instance` per turn**. At aggregation, all turns of a
document are grouped by `doc_id` and averaged into a **single per-doc value**
(so a 5-turn doc weighs the same as a 1-turn doc, not 5×).

The open is searched in the **current turn only** (the model's generation plus, for
prefill templates, the turn's own generation-prompt prefill) — never the rendered
history — so a prior turn's unclosed open cannot inflate `has_open`/`correct`. This
holds for both prefill and non-prefill models, single- and multi-turn.

**Limitation — no cross-turn thinking blocks.** A reasoning block that opens in one
turn and closes in a *later* turn is never stitched together. Such turns are
malformed per-turn (`correct = 0`: the opening turn never closes; the closing turn
never opened this turn), so neither contributes a (mis-attributed) `thinking_length`
— only their `response_length` counts. For prefill templates this never arises (every
turn re-opens its own block).

## How length is handled for malformed or non-thinking responses

This describes the **aggregated** numbers (`--log_length_metrics`). Two separate
bases — the key design point:

- **`response_length_*`** is averaged over **all** responses.
- **`thinking_length_*`** is averaged over **well-formed responses only**
  (`thinking_format_correct = 1`). Malformed/incorrect responses simply do not carry
  a `thinking_length`, so they are **excluded from its denominator** — the average is
  never biased by them (no zero-fill).

Cases:

- **No thinking / `enable_thinking=false`:** only `response_length_*` is emitted; no
  `thinking_format_*` and no `thinking_length_*`.
- **Malformed — opened but never closed** (`has_close = 0`, `correct = 0`): no
  `thinking_length` is recorded for that response; it is excluded from the
  thinking-length average but still counts toward `response_length_*`. Because the
  strip keys off the close token, an unclosed block is **not** stripped, so the
  reasoning remains in the scored text.
- **Per-doc / repeats:** one value per doc — the mean over that doc's responses
  carrying the key. A doc with **no** well-formed response contributes no
  `thinking_length` value at all (it leaves the thinking-length denominator too).
- **Consequence — bases differ:** because the two metrics average over different
  populations, the aggregate `thinking_length` is **not bounded by** the aggregate
  `response_length` (it can even exceed it when the well-formed responses are the
  long ones). The **per-sample** invariant `thinking ≤ response` still always holds.
- **Measurement gaps:** the `_tokens` family can have a **smaller sample base** than
  `_words`/`_chars` on backends that don't expose token counts for every response.

### Impact summary (direction of bias)

As the share of **malformed (opened-but-not-closed)** responses rises:

| Quantity | Direction | Why |
|----------|-----------|-----|
| `thinking_format_has_close`, `thinking_format_correct` | **down** | flagged 0 per malformed response — this is the signal you want |
| `thinking_length_*` (aggregate) | **unbiased** | averaged over the well-formed responses only; malformed ones are excluded from the denominator, not counted as 0 |
| `response_length_*` | **unaffected / relatively high** | still counts the full text of every response, reasoning included |
| **task score** | **likely deflated** | the un-stripped reasoning stays in the scored text and pollutes answer extraction |

## Caveats

- **Answer extraction.** A reasoning model often answers in its own style after
  the close token (a bare number, `\boxed{...}`), which a strict
  format-specific extractor (e.g. gsm8k's `strict-match` `#### N`) will miss even
  though the answer is correct. Prefer a robust extractor (`flexible-extract`, or
  a CoT/zero-shot task variant). `thinking_format_*` + the length split help
  distinguish "reasoned fine, extractor too strict" from a real thinking failure.
- **`has_open` is current-turn only** (see Multi-turn above), so few-shot demo tags
  and a prior turn's unclosed open can't inflate it. On prefill templates it is
  ~always 1, so `correct` there tracks whether the turn *closed*.
- **Response caching.** The metrics are measured *during generation*, so a response
  served from the `--cache_requests` cache (which skips generation) carries no
  `length_info` and is excluded from these aggregates. Re-run without a cache hit
  (or without `--cache_requests`) to populate them for every doc.
