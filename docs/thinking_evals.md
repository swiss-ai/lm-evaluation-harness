# Thinking / reasoning evals

How the harness handles reasoning ("thinking") models on generative tasks: the
reasoning-trace **strip**, and the per-response **length** and **thinking-format**
metrics it records. For the exact CLI flags and `--model_args` keys, see the
[CLI reference](./interface.md#generation-length--thinking-format-metrics).

**Contents:** [Quick start](#quick-start) · [How it works](#how-it-works) ·
[When things turn on](#when-things-turn-on) · [Metric reference](#metric-reference) ·
[Aggregation](#aggregation) · [Multi-turn](#multi-turn) ·
[Backend differences](#backend-differences) · [Troubleshooting](#troubleshooting) ·
[Appendix: internals](#appendix-internals)

## Quick start

**The reasoning machinery is opt-in.** By default nothing is detected, nothing is
stripped, and no thinking metric is recorded — you must name a close token, or ask the
harness to find one.

```bash
# 0. Defaults: no strip, no thinking metrics. Only `response_length_*` is measured.
lm_eval --model hf --model_args pretrained=$M --tasks gsm8k --apply_chat_template

# 1. Opt in: read the reasoning tokens from the chat template, strip the trace before
#    scoring, and aggregate `thinking_format_*`.
--model_args pretrained=$M,autodetect_think_tokens=true

# 2. Or name the close token yourself (no template scan; `has_open` is not tracked).
--model_args pretrained=$M,think_end_token='</think>'

# 3. Also aggregate the lengths, and keep the per-response values on disk.
lm_eval --model hf --model_args pretrained=$M,autodetect_think_tokens=true \
        --tasks gsm8k --apply_chat_template --log_length_metrics --log_samples -o out/

# 4. Strip, but skip the thinking metrics (and their per-response cost).
--model_args pretrained=$M,autodetect_think_tokens=true,track_thinking_metrics=false
```

You get per-response values in `length_info` (sample jsonl, with `--log_samples`) and
per-task means in `results.json` / the printed table / W&B.

## How it works

Reasoning models wrap their deliberation in an **open**/**close** token pair (e.g.
`<think>`/`</think>`, or `<|inner_prefix|>`/`<|inner_suffix|>`). Once a close token is
known, the `hf`, `vllm` and `sglang` backends **strip** everything up to and including
the **last** close token, so only the post-reasoning answer is scored. No close token is
known unless you pass `think_end_token=` or opt in to `autodetect_think_tokens=true`.

Each response is measured *before* the strip and recorded in the sample's `length_info`
field — one entry per response — for `generate_until` and `multi_turn_generate` tasks.

```text
generation   <think>2 + 2 is 4.</think>The answer is 4.
             └───── thinking span ────┘└─── answer ───┘
             └────────────── response span ───────────┘

scored       The answer is 4.
```

- `response_length_*` covers the response span (reasoning **+** answer); always recorded.
- `thinking_length_*` covers the thinking span — the same boundary the strip uses — and is
  recorded **only for well-formed responses**, so a mis-attributed span is never measured.

> **Why this matters.** If the close token is wrong or unresolved, the reasoning is
> **not** stripped and pollutes answer extraction. `thinking_format_has_close` and
> `thinking_length_*` are what make that visible.

## When things turn on

These four switches are **independent**.

| Question | Answer |
|---|---|
| Does the model think? | `enable_thinking` — a **chat-template argument only**. See [`enable_thinking`, precisely](#enable_thinking-precisely). |
| Are the reasoning tokens detected? | `autodetect_think_tokens` — **opt-in, default `false`**. The only lever over template scanning. |
| Does the strip run? | Iff a **close** token is known: passed via `think_end_token`, or auto-detected. |
| Are thinking metrics recorded? | `track_thinking_metrics`. Default: on iff a close token is known. |
| Is response length recorded? | **Always**, for every `generate_until` response — thinking or not. |

So an unconfigured run does nothing special: no detection, no strip, no thinking metrics
— just `response_length_*`. You opt in with `autodetect_think_tokens=true` or by naming
`think_end_token=` yourself.

`enable_thinking` appears in exactly one of those rows. It does **not** gate detection,
the strip, or metric *tracking*. It can still move metric *values*, because it changes
the prompt the model is given — see [`enable_thinking`, precisely](#enable_thinking-precisely).

### `enable_thinking`, precisely

`enable_thinking` is a **chat-template argument**: it decides whether the model reasons.
Whether it actually reaches the template differs per backend:

| Backend | Unset | Set to `true` / `false` |
|---|---|---|
| `vllm` | Forwarded as `true` (the parameter's default) → thinking on | Forwarded |
| `hf` | **Not forwarded** — the template's own default decides | Forwarded |
| `sglang` | Never forwarded (no such argument) — the template's default decides | *n/a* |

In particular, `enable_thinking` neither arms nor disables the strip: only a known close
token does that. Turning thinking on does not make the harness look for the tokens, and
turning it off does not stop it.

**One indirect effect.** Because `enable_thinking` changes the *rendered prompt*, it can
move metric **values** — never the switches. On a **prefill** template that injects the
open token into the generation prompt only while thinking is on, `enable_thinking=false`
means the open is no longer prefilled, so `thinking_format_has_open` (and therefore
`correct`) reflects that. The metric is measuring the prompt you actually used, which is
the intended behaviour.

### Resolving the reasoning tokens

- Forced with `think_start_token` / `think_end_token`, or — **only if you opt in with
  `autodetect_think_tokens=true`** — read from the chat template's `inner_token` /
  `outer_token` declarations. On `hf` the close may also be an integer token id.
- **Auto-detection is off by default**, so an unconfigured run scans nothing: no close,
  hence no strip and no thinking metrics. An explicit `think_end_token` is a deliberate
  strip request and is honoured with or without auto-detection.
- When on, detection happens at **model construction**, reading `tokenizer.chat_template`
  directly — it does not require `--apply_chat_template`. The strip can therefore be armed
  on a run that never renders the template.
- **Fail-loud:** if auto-detection is on and the template declares reasoning tokens but
  the close cannot be resolved, construction raises. Pass `think_end_token=`, or drop
  `autodetect_think_tokens=true`. (This escape works on all three backends, including
  `sglang`.) With auto-detection off — the default — the guard can never fire.
- A missing **open** token only warns: `has_open` is not tracked and `correct` falls back
  to `has_close` (close-only style). Pass `think_start_token=` to track it.

### `track_thinking_metrics`

Whether a close token *exists* and whether you *want thinking metrics* are separate
questions. This `--model_args` key (all three backends) decides the second:

| Value | Behaviour |
|---|---|
| unset (default) | **Derive**: track iff a close token is known. |
| `true` | Force on. |
| `false` | Force off — drop `thinking_format_*` / `thinking_length_*` and their per-response cost. |

A **close token is required either way** (it defines `has_close` and the thinking span),
so forcing it on without one warns and stays off. A missing **open** is fine: the format
metric degrades to close-only (`correct == has_close`) and `has_open` is not emitted. An
open *without* a close measures nothing — it warns, and the strip stays off too.
`response_length_*` is never affected.

This flag governs the **metrics, not the strip**: `track_thinking_metrics=false` still
strips. To disable both, simply don't name a close token — neither via `think_end_token`
nor via `autodetect_think_tokens=true`. That is the default.

## Metric reference

**Length** — measured on the response, before the strip:

| Metric | Meaning |
|--------|---------|
| `response_length_words` / `_chars` / `_tokens` | The whole response (reasoning + answer). |
| `thinking_length_words` / `_chars` / `_tokens` | The reasoning span: start of the generation up to and including the **last** close token. Recorded only when `thinking_format_correct = 1`. |

**Thinking-format** (`thinking_format_*`, 0/1 per response) — a well-formedness signal,
computed on the **current turn only**:

| Metric | Meaning |
|--------|---------|
| `thinking_format_has_open` | The block was opened *this turn*: the open token is emitted in the generation, or injected by the template's generation prompt (a prefill template). Emitted only when an open token is known. |
| `thinking_format_has_close` | The close token is present in the generation (the model emitted it). |
| `thinking_format_correct` | **Open+close models:** open present (this turn), close present, open precedes the (last) close, and the block is not re-opened after the (first) close. **Close-only models** (no open token known): reduces to `has_close`. |

### Token counts (`_tokens`)

`_tokens` is the only family whose provenance differs per backend:

| Backend | `response_length_tokens` | `thinking_length_tokens` |
|---|---|---|
| `hf`, integer close | Exact generated ids, bounded to the response; **excludes** the terminating EOS. | Exact (ids up to the close). |
| `hf`, string close | Same as above. | Re-encode of the span. |
| `vllm` | `len(output.outputs[0].token_ids)` — includes the stop-string tokens and the EOS, which the text does not. | Re-encode of the span. |
| `sglang` | `meta_info.completion_tokens` — includes the terminating token; **omitted** when absent. | Re-encode of the span. |

Consequences:

- Per-sample `thinking ≤ response` holds **exactly** for `_chars` and `_words` (the
  thinking span is a literal prefix of the response). For `_tokens` it holds only on the
  `hf` integer-close path; elsewhere re-encoding the span in isolation can tokenize its
  boundary differently and land 1–2 tokens over — most likely when the answer after the
  close is very short.
- `_tokens` can have a **smaller sample base** than `_chars`/`_words` (`sglang`, when
  `completion_tokens` is absent).
- **Prefer `_chars`** for strict ratios and for comparing across backends.

## Aggregation

Where the numbers land:

| Sink | Flag | Length | Format |
|------|------|--------|--------|
| Per-sample (`length_info` in `samples_*.jsonl`) | `--log_samples` | yes | yes |
| Per-task (`results.json` + printed table) | length: `--log_length_metrics` | only with the flag | always |
| W&B (native) | `--wandb_args` (uploads whatever was aggregated) | only with `--log_length_metrics` | always |

"Always" still requires that the metrics are **tracked** (see
[When things turn on](#when-things-turn-on)) and the relevant token resolved.

Aggregation is a per-task **mean over documents**, emitted under the `none` filter; each
metric also gets a `_stderr,none` companion. **One value per doc**: the mean over that
doc's responses *carrying the key* (so `repeats > 1` and multi-turn collapse to one).

Two different denominators — the key design point:

- **`response_length_*`** averages over **all** responses.
- **`thinking_length_*`** averages over **well-formed responses only**
  (`thinking_format_correct = 1`). Malformed responses simply do not carry the key, so
  they leave its denominator rather than being counted as zero.

Cases:

- **Not tracked** (no close token — the default, unless you pass `think_end_token=` or
  `autodetect_think_tokens=true` — or `track_thinking_metrics=false`): only
  `response_length_*` is emitted.
- **Malformed — opened but never closed** (`has_close = 0`): no `thinking_length` for
  that response; it still counts toward `response_length_*`. Because the strip keys off
  the close, an unclosed block is **not** stripped — the reasoning stays in the scored text.
- **A doc with no well-formed response** contributes no `thinking_length` value at all.
- **Bases differ**, so the aggregate `thinking_length` is **not bounded by** the aggregate
  `response_length` — it can exceed it when the well-formed responses are the long ones.

### Direction of bias

As the share of **malformed (opened-but-not-closed)** responses rises:

| Quantity | Direction | Why |
|----------|-----------|-----|
| `thinking_format_has_close`, `thinking_format_correct` | **down** | flagged 0 per malformed response — this is the signal you want |
| `thinking_length_*` (aggregate) | **unbiased** | averaged over well-formed responses only; malformed ones leave the denominator, not counted as 0 |
| `response_length_*` | **unaffected / relatively high** | still counts every response in full, reasoning included |
| **task score** | **likely deflated** | the un-stripped reasoning stays in the scored text and pollutes answer extraction |

### Key naming

Aggregated metrics carry the `none` filter. In `results.json` the key is
`thinking_format_correct,none`, nested under the task (`results["gsm8k"]`). In **W&B**
the trailing `,none` is stripped: `gsm8k/thinking_format_correct`.

## Multi-turn

For `multi_turn_generate`, each turn is a separate `generate_until` response and the task
creates **one `Instance` per turn**, so the strip and the per-response length/format are
measured **per turn**. At aggregation all turns of a document are grouped by `doc_id` and
averaged into a **single per-doc value** — a 5-turn doc weighs the same as a 1-turn doc,
not 5×.

The open is searched in the **current turn only**: the model's generation plus, for
prefill templates, that turn's own generation-prompt prefill. The rendered history is
never consulted, so few-shot demo tags and a prior turn's unclosed open cannot inflate
`has_open` / `correct`. This holds for prefill and non-prefill models alike, single- and
multi-turn. On prefill templates `has_open` is ~always 1, so `correct` there effectively
tracks whether the turn *closed*.

**Limitation — no cross-turn blocks.** A reasoning block that opens in one turn and
closes in a *later* turn is never stitched together. Both turns read `correct = 0` (the
opening turn never closes; the closing turn never opened this turn), so neither
contributes a mis-attributed `thinking_length` — only their `response_length` counts.
Prefill templates never hit this, since every turn re-opens its own block.

## Backend differences

All three measure the response before the strip, but they bound "the response"
differently:

| | `hf` | `vllm` | `sglang` |
|---|---|---|---|
| Stops each sequence independently | **no** — the whole batch | yes | yes |
| What bounds "the response" | pad/eos tail trimmed, then truncated at this sequence's own first stop | engine text (already stop-trimmed) | engine text (already stop-trimmed) |

**Why `hf` needs bounding.** Its stop criteria halt the *whole batch* only once **every**
sequence has stopped, so a sequence that finishes early keeps its row filled — with
pad/eos after an EOS, or with **real tokens** generated past its own stop if it matched
only a *string* stop. Measuring that raw row would inflate lengths by up to the batch
maximum and let a longer batch sibling flip another response's `thinking_format_*`. So
`hf` trims the tail, truncates at the sequence's own first stop, and maps that bound back
onto the exact generated ids; the integer close is then searched inside that bounded view.

`vllm` and `sglang` stop each sequence independently, so no sibling can inflate another's
output and there is nothing to trim.

**Scoring is untouched by any of this** — the bounding affects the metrics only. See
[Token counts](#token-counts-_tokens) for the resulting comparability caveats.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **No `thinking_*` metrics at all, and the reasoning is not stripped** | The default: auto-detection is **off**, so no close token was resolved | Pass `autodetect_think_tokens=true`, or name the close with `think_end_token=` |
| `ValueError` at model construction, "reasoning close … could not be auto-detected" | You enabled `autodetect_think_tokens=true`, and the chat template declares `inner_token`/`outer_token` but not as a quoted literal | Pass `think_end_token=` instead, or drop `autodetect_think_tokens=true` |
| Reasoning is stripped even though I passed `enable_thinking=false` | `enable_thinking` never controlled the strip — a close token is known (explicit, or auto-detected because you opted in), and the strip keys off the close | Remove `autodetect_think_tokens=true` / `think_end_token=` |
| Task score low, but the reasoning looks fine | A strict extractor (e.g. gsm8k `strict-match`, `#### N`) misses the model's post-close answer style (a bare number, `\boxed{…}`) | Use a robust extractor (`flexible-extract`) or a CoT/zero-shot task variant |
| `thinking_format_correct` is 0 everywhere, yet `has_close` is 1 | The open token **is** known but `has_open` reads 0 — the model never emits it and the prefill probe failed silently | No override for the probe today. Either read `has_close` directly, or drop the open (name only `think_end_token=`, without `autodetect_think_tokens=true`) so the metric degrades to close-only, where `correct == has_close` |
| `thinking_format_has_close` is 0 everywhere and the model never seems to reason | A close is known, but the chat template defaults thinking **off** (`enable_thinking` was never forwarded to it) | On `hf`/`vllm`, pass `enable_thinking=true` to actually reason (`sglang` has no such argument — the template decides); or silence the metrics with `track_thinking_metrics=false` |
| `has_open` is never emitted | No open token is known — the format metric degraded to close-only, so `correct == has_close` | Pass `think_start_token=` |
| No `thinking_length_*` for a task | No response was well-formed | Check `thinking_format_correct` |
| `thinking_length_tokens` > `response_length_tokens` | Re-encode drift on the non-`hf`-integer paths | Use `_chars` |
| Aggregate `thinking_length` > aggregate `response_length` | Different denominators, by design | See [Aggregation](#aggregation) |
| No `length_info` at all in the samples | Responses were served from the `--use_cache` response cache, which skips generation | Re-run without a cache hit. `--cache_requests` is unaffected — it caches only request *building* |
| No `thinking_*` metrics though the model thinks | `track_thinking_metrics=false`, or no close token is known | Pass `think_end_token=` / `track_thinking_metrics=true` |

## Appendix: internals

- `Instance.length_info` (`lm_eval/api/instance.py`) is a `list[dict]`, one entry per
  response, mirroring how `resps` are appended.
- **Producers:** `generate_until` in `lm_eval/models/{huggingface,vllm_causallms,sglang_causallms}.py`,
  via the helpers in `lm_eval/models/utils.py` (`build_length_info`,
  `compute_generation_length_info`, `compute_thinking_format_info`, `resolve_think_tokens`,
  `resolve_track_thinking_metrics`, `detect_open_prefilled`). Measurement never raises —
  a failure yields no metrics for that response rather than breaking generation.
- **Consumer:** `promote_generation_info_metrics` (`lm_eval/evaluator_utils.py`) groups by
  `doc_id`, caps each instance at `repeats` (dropping data-parallel padding clones), and
  emits under the `none` filter into the usual gather → mean → consolidate → W&B path.
- The `hf` integer-close path builds the same dict **inline**, because its boundary is a
  token id rather than a string. Keep it in sync with `build_length_info`.
