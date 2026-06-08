# Multi-Turn Rollout

`output_type: multi_turn_generate` is a new task-level capability for
**scripted multi-turn benchmarks** — datasets where each row specifies
a sequence of user turns the model must respond to, with each
generation conditioned on the model's own prior responses. It is the
foundation for the new S-RuLES sub-suite of `realguardrails` and is
designed to unblock SysBench, MT-Bench turn-2, Multi-IF turns 2-3,
MT-Eval, and the judge-based multi-turn suites of RealGuardrails /
SystemCheck (handwritten, distractors, Monkey Island) once judge
integration is added.

This document describes the architecture, the task contract, the
limits, and how to add a new multi-turn benchmark.

## Why this exists

The previous (single-shot) task model assumed one generation per doc.
For benchmarks like S-RuLES, where each dataset row has 1-7 trailing
user turns and the upstream evaluator iterates per-turn with
early-exit on rule violation, single-shot rollout silently produces
different numbers from the paper on the ~17% of rows with multiple
trailing user turns (100% of S-RuLES `basic_helpful`, ~10% of
`redteam_helpful`, ~3% of `redteam_harmless`). The architectural blocker
was also keeping ~6 other multi-turn benchmarks deferred.

`multi_turn_generate` solves this with one reusable evaluator-level
feature instead of per-task workarounds, while reusing every backend's
existing `generate_until(list[Instance]) -> list[str]` interface — no
backend wrapper changes.

## Architecture in one sentence

A new evaluator phase (`run_multi_turn_rollout`) runs **before** the
standard one-shot dispatch loop, iterates the conversation turn-by-turn
across all pending docs, and appends turn-indexed `Instance` objects to
each task's `_instances` list — so the existing filter + `process_results`
+ sample-log path consumes the variable-length per-doc instance lists
the same way it consumes `multiple_choice`'s N-instances-per-doc.

## Data dependency, and why each turn is a synchronization barrier

Turn `t+1`'s prompt contains turn `t`'s response. This dependency is
intrinsic; there is no way to dispatch turn `t+1` until the backend has
returned turn `t`. Per-turn synchronization is therefore unavoidable —
but **within** each turn, all pending docs across all multi-turn tasks
share a single batched `generate_until` call. vLLM continuous batching,
HF auto-batch, and `local-chat-completions` `num_concurrent` operate
inside one turn as in single-shot eval.

## Components

### `lm_eval/api/multi_turn.py` — `MultiTurnState`

Per-doc rollout state, owned by the task during `evaluate()`:

```
@dataclass
class MultiTurnState:
    doc: dict
    history: list[Message]   # mutated in place; system + prefilled + appended
    done: bool = False
```

`task._multi_turn_states: dict[int, MultiTurnState]` is keyed by
`doc_id`. The driver mutates `history` and `done` in place; the task
owner does not need to manage state itself.

### `ConfigurableTask` hooks (`lm_eval/api/task.py`)

Three task-overridable methods, declared as defaults on
`ConfigurableTask`. A task that sets `output_type: multi_turn_generate`
MUST override `build_initial_messages` and `next_user_turn`;
`should_stop` is optional (defaults to `False`).

```python
def build_initial_messages(self, doc: dict) -> list[Message]:
    """[system, ...prefilled turns up to and including the LAST prefilled
    assistant turn]. Does NOT include the queued user turns the model
    will respond to — those are yielded one at a time by next_user_turn.
    Raises NotImplementedError by default; multi-turn tasks must override.
    """

def next_user_turn(
    self, doc: dict, history: list[Message], turn_idx: int
) -> str | None:
    """The next user turn the model should answer. Return None when the
    rollout is exhausted for this doc. ``turn_idx == 0`` is the first
    model-generation; history at that point ends with the last prefilled
    assistant message. Raises NotImplementedError by default.
    """

def should_stop(
    self, doc: dict, history: list[Message], turn_idx: int
) -> bool:
    """Called AFTER appending the model's assistant response for
    turn_idx. Return True to early-exit (this doc is done and no further
    turns will be generated). Default: False.
    """
```

### `run_multi_turn_rollout` driver (`lm_eval/evaluator.py`)

Public function (intentionally not underscore-prefixed; consumers and
tests reach it as `lm_eval.evaluator.run_multi_turn_rollout`). The
algorithm:

```
pending = [(task_output, doc_id, state) for each multi-turn task's docs]
turn_idx = 0
while pending and turn_idx < MAX_SAFETY_TURNS:
    batch_this_turn = []
    for (to, doc_id, state) in pending:
        user = to.task.next_user_turn(state.doc, state.history, turn_idx)
        if user is None:
            state.done = True; continue
        state.history.append(Message("user", user))
        prompt = lm.apply_chat_template(
            [m.to_dict() for m in state.history], add_generation_prompt=True,
        )
        inst = Instance(
            request_type="generate_until",
            doc=state.doc,
            arguments=(prompt, deepcopy(task.config.generation_kwargs or {})),
            idx=turn_idx,
            metadata=(task.config.task, doc_id, task.config.repeats or 1),
        )
        batch_this_turn.append((to, state, inst))

    if not batch_this_turn:
        break

    responses = lm.generate_until([inst for _, _, inst in batch_this_turn])

    for (to, state, inst), resp in zip(batch_this_turn, responses):
        inst.resps.append(resp)
        state.history.append(Message("assistant", resp))
        to.task._instances.append(inst)
        if to.task.should_stop(state.doc, state.history, turn_idx):
            state.done = True

    pending = [(to, doc_id, st) for (to, doc_id, st) in live if not st.done]
    turn_idx += 1
```

Key properties:
+ One synchronous barrier per turn (intrinsic to the data dependency).
+ Cross-task batching at the per-turn level maximizes occupancy.
+ Reuses `lm.generate_until` unchanged — no backend wrapper changes.
+ Reuses the existing `Instance` shape — variable-per-doc count flows
  cleanly through the existing post-process loop.

### Integration with the existing single-shot path

The driver runs **before** `evaluator.evaluate()`'s standard
`for reqtype, reqs in requests.items()` dispatch loop. Single-shot tasks
are unaffected; the new branch fires only if at least one task in the run
has `output_type: multi_turn_generate`.

After the driver returns, the standard post-process loop in `evaluate()`:
1. `task.apply_filters()` — operates on each Instance's `resps` (a
   singleton list per turn), same as single-shot. Default `take_first`
   produces `inst.filtered_resps["none"]` = the string response.
2. `instances_by_doc_id` groups by `doc_id`, sorts by `idx`, calls
   `task.process_results(doc, [filtered_resp_t0, filtered_resp_t1, …])`.
3. Sample logger writes one record per doc with `arguments`/`resps` as
   variable-length lists (one entry per generated turn). No logger
   change needed; the shape is already supported by `multiple_choice`.

## Backend compatibility

| Backend | Parallelism path | Per-turn batching | Status |
|---|---|---|---|
| HF transformers, single rank (CPU, MPS, 1 GPU) | n/a | `_detect_batch_size` runs once at first call; cached value reused across turns. | ✓ Supported |
| HF transformers, `parallelize=True` (single-rank model-parallel) | Single process spans GPUs; harness sees `world_size == 1`. | Same as single-rank. | ✓ Supported |
| HF transformers, `accelerate launch --num_processes N` (data-parallel) | `world_size == N`; harness shards docs across ranks via `task.doc_iterator`; driver gathers live counts each turn and pads with sentinel dummies (`max_gen_toks=1`) so ranks stay even-batched. | One batched call per turn per rank; global gather per turn. | ✓ Supported |
| vLLM `tensor_parallel_size=T` | One model copy across T GPUs (intra-model). Configured via `--model_args`. Harness sees `world_size == 1`. | Continuous batching within each turn; intrinsic sync barrier between. | ✓ Supported |
| vLLM `data_parallel_size=D` | D model replicas; vLLM internally routes prompts across replicas inside `generate_until(list)`. Harness sees `world_size == 1`. | Continuous batching across all D replicas. | ✓ Supported |
| SGLang `num_concurrent=N` | Backend-internal concurrency; harness sees `world_size == 1`. | Concurrent dispatch within a turn; intrinsic sync barrier between. | ✓ Supported |
| `local-chat-completions` + `num_concurrent=N` (any OpenAI-compatible server) | Same as SGLang. | Same. | ✓ Supported |
| OpenAI / OpenAI-compatible chat APIs via `openai-chat-completions` | Provider-side concurrency via `num_concurrent`. | Same. | ✓ Supported |
| Anthropic / Google native backends (`anthropic_llms.py` etc.) | Provider-side concurrency. | n/a | ✗ Not supported (no `apply_chat_template`; driver hard-errors). Route through `local-chat-completions` against a wrapper instead. |

No backend wrapper changes are required because each turn issues exactly
one `generate_until(list[Instance])` call — the same interface single-shot
eval uses. The two parallelism axes (cross-rank under `accelerate launch`
and within-rank under the backend's own batching) compose: each rank's
per-turn batch is dispatched through whichever backend-internal mechanism
that rank's `LM` instance uses.

### Choosing a parallelism path

- **vLLM available** → prefer `data_parallel_size=N` (and
  `tensor_parallel_size>1` only if the model doesn't fit one GPU). The
  per-turn batch crosses all replicas inside one `generate_until` call;
  no harness-level coordination needed.
- **HF only, multiple GPUs** → `accelerate launch --num_processes=N`.
  Cross-rank coordination costs one small gather per turn plus dummy
  generations on ranks whose docs early-exited; expected to be tolerable
  when docs are uniformly distributed across ranks (which the default
  `doc_iterator` index-mod-rank sharding gives for shuffled datasets —
  not yet measured against a real workload). Avoid combining with
  `parallelize=True` on the same model (both would try to spread it).

  Concrete example:

  ```bash
  accelerate launch --num_processes 8 -m lm_eval run \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,dtype=bfloat16,strip_system_boilerplate=true \
    --tasks realguardrails_s_rules_br \
    --apply_chat_template \
    --batch_size 8 \
    --output_path ./results/
  ```
- **Single GPU / single host** → no special config needed; the
  in-process per-turn batched call already exploits the device.

## Generation parameters & overrides

Every turn is a `generate_until` call built from the task's single
`generation_kwargs` block (the driver deep-copies
`task.config.generation_kwargs` per turn), so sampling params behave as for
`generate_until` tasks:

- **Defaults** (YAML omits `generation_kwargs`): greedy — `temperature=0.0`,
  `do_sample=False`, `max_gen_toks=256` — with **`until=[]`**. This is the
  same greedy default as `generate_until` except for `until`:
  `generate_until` defaults to `until=[fewshot_delimiter]` (`"\n\n"`), which
  would truncate a multi-turn response at the first blank line, so
  multi-turn instead relies on the model's own turn-end (EOS) token.
- **CLI overrides are respected**: `--gen_kwargs key=value` *merges* into the
  task's `generation_kwargs` (CLI wins on conflicting keys, other YAML keys
  are kept), identical to `generate_until`. The override is gated on
  `output_type in ("generate_until", "multi_turn_generate")` in
  `simple_evaluate`.
- Per-turn *varying* params are not supported (one block for all turns) —
  see the limit below.

### Turn termination depends on the model's EOS — mind mis-configured fine-tunes

Each turn stops at the model's EOS token or any `until` string. **If a
model's tokenizer `eos_token` is not its chat turn-end token, generation
runs on past the turn boundary** — the model emits its turn-end token, the
backend does not treat it as a stop, and it keeps generating (usually
hallucinating the next user/assistant turns up to `max_gen_toks`),
silently corrupting the per-turn response the evaluator scores.

Observed with fine-tunes whose `eos_token` is `<|endoftext|>` while the
chat template ends turns with `<|im_end|>` (no `generation_config.json` to
reconcile them). Remedies:

- **vLLM**: add the turn-end token as a stop, `--gen_kwargs 'until=<|im_end|>'`
  — works because vLLM detokenizes with special tokens visible, so the stop
  string matches.
- **HF**: needs a **token-level** stop. A stop *string* does NOT work (HF
  strips special tokens from the decoded output, so `<|im_end|>` never
  appears to match), and setting the *tokenizer's* `eos_token` does NOT help
  either — HF's `model.generate()` stops on the *model's*
  `generation_config.json`/`config.json` `eos_token_id`, not the tokenizer's.
  Pass the turn-end **token id** instead: `--gen_kwargs 'eos_token_id=151645'`
  (`HFLM._model_generate` forwards it to `model.generate()`), or correct the
  model's `generation_config.json` `eos_token_id`.

Well-formed chat models — tokenizer `eos_token` == chat turn-end (Llama
`<|eot_id|>`, stock Qwen2.5 `<|im_end|>`) or a `generation_config.json`
listing the turn-end id — need none of this.

## Limits

These are **intentional v1 trade-offs**. None of them is a structural
problem; each has a documented remediation path.

### Distributed: HF data-parallel tail-pad cost

Cross-rank synchronisation is implemented (see the row for
`accelerate launch` above), but it has a cost worth understanding.

**Why a barrier is unavoidable.** Different ranks hold docs with
different turn counts and different early-exit timings, so per-rank
live-doc counts diverge. The standard FSDP/DDP requirement that every
rank issue an evenly-sized batch every collective forces ranks with
fewer live docs to pad. The driver:

1. Issues `accelerator.gather` at the entry guard so no rank enters
   the loop alone (would deadlock the next gather).
2. Issues `accelerator.gather` at the start of every turn to compute
   `global_max = max(live_n across ranks)` and to detect global
   termination (`global_max == 0` → all ranks break together).
3. Pads each rank's per-turn batch up to `global_max` with a sentinel
   `dummy_instance` (`max_gen_toks=1`, `until=[]`). The padded slots
   never reach `task._instances`; their responses are discarded.

**Tail cost.** As docs early-exit unevenly, a rank that finishes early
still issues padded-only batches until the slowest rank is done. With
a balanced random shard the tail is short; with biased early-exit
(e.g. all redteam docs landing on one rank) it can dominate. Doc
sharding by `doc_iterator` is index-mod-rank, which is essentially
random for shuffled datasets. No mitigation is implemented; if a
benchmark shows tail-heavy behaviour, the right fix is task-level
balanced sharding.

**Backends that don't go through `accelerate launch`** — vLLM
`data_parallel_size`, SGLang / `local-chat-completions` /
`api_models` with `num_concurrent`, single-rank HF — expose
`world_size == 1` to the harness and bypass this entire mechanism;
the backend's own internal parallelism handles routing inside the
single `generate_until(list[Instance])` call.

### `repeats > 1` not supported

Each `(doc, turn)` maps to exactly one `Instance`. If a YAML sets
`repeats: 3`, `_build_initial_multi_turn_states` raises `ValueError`
with a remediation message. The standard dispatch loop's
`cloned_reqs.extend([req] * req.repeats)` cloning is not replicated in
the driver. For multi-turn benchmarks where stochastic resampling is
needed (rare — most are T=0), this restriction is a real constraint.

### `--cache_requests` (Instance cache) is a no-op for multi-turn tasks

`build_all_requests` short-circuits for `multi_turn_generate` before
reaching the cache write path. The response cache (`--use_cache`) still
works per-turn — each turn's prompt is distinct so cache keys differ
naturally.

### `MAX_SAFETY_TURNS = 64` hardcoded

A defensive cap against pathological tasks that never return `None` from
`next_user_turn`. No real benchmark exceeds ~10 turns; 64 is comfortable
headroom. Currently a module-level constant in `lm_eval/evaluator.py`;
expose via task config if a future benchmark needs deeper rollouts.

### Per-doc history is rebuilt every turn (HF backend)

Each turn re-renders the chat template with the full growing history.
On HF, this means re-tokenizing the whole prefix every turn — O(N²)
tokenization across an N-turn rollout. Negligible for ≤7 turns; would
need attention if a benchmark with ≥20 turns lands.

### Sample-log shape change

`arguments` and `resps` are now variable-length lists (one entry per
turn). Downstream paper-comparison scripts that hardcode list length 1
need updating. `multiple_choice` already produced variable lengths, so
`wandb_logger` and `evaluation_tracker` handle this correctly.

### Per-turn `generation_kwargs` not supported

The driver uses the task's single `generation_kwargs` block for every
turn. Benchmarks that need different `max_gen_toks` per turn (rare)
would need a `next_gen_kwargs(turn_idx)` hook on the task — straightforward
to add.

### No bootstrap confidence intervals

The harness emits point estimates only. For paper-comparison reporting
(e.g. paper Tables with 95% CIs from `bootstrap=10000`), post-process the
per-doc booleans in `samples_*.jsonl`. Out of scope for the driver.

## Performance characteristics

For S-RuLES `basic_helpful` (250 rows, average ~2.5 trailing user turns)
on `meta-llama/Llama-3.2-3B-Instruct`:

| Backend | Wall-clock per turn | Total wall-clock |
|---|---|---|
| HF + MPS (M4 Mac, bs=4) | ~30 s | ~1.5–2 min for the smoke (10 docs); ~25–40 min for full 250 docs |
| vLLM (single GH200, `num_concurrent=64`) | ~3–5 s per turn | ~10–15 min for full 250 docs |
| HF + `accelerate launch --num_processes N` | not measured | not measured (expected ~N× single-rank for evenly-distributed docs; less in the tail-pad regime) |

vLLM continuous batching gives the biggest speedup; cross-doc batching
inside a turn approximates the throughput of single-shot eval on a
larger doc set (because every turn is a batched call across all
still-pending docs).

## Adding a new multi-turn benchmark

1. **YAML.** Set `output_type: multi_turn_generate`. Generation params,
   metric list, dataset path, and (optionally) `class: !function task.MyTask`
   work as in single-shot. The task DOES NOT need `description` or
   `doc_to_text` — those are unused by the driver.

2. **Task class.** Subclass `ConfigurableTask` (or any `ConfigurableTask`
   subclass) and implement at minimum:

   ```python
   from lm_eval.api.task import ConfigurableTask
   from lm_eval.api.utils import Message

   class MyMultiTurnTask(ConfigurableTask):
       def build_initial_messages(self, doc):
           return [Message("system", doc["system_prompt"])]

       def next_user_turn(self, doc, history, turn_idx):
           turns = doc["user_turns"]   # list of strings
           return turns[turn_idx] if turn_idx < len(turns) else None

       def should_stop(self, doc, history, turn_idx):
           return False   # default
   ```

3. **`process_results`.** Receives `(doc, results: list[str])` where
   `results[i]` is the model's i-th assistant response. Reconstruct the
   full conversation if your scorer needs it; the harness does not
   pass the conversation directly to keep the signature compatible with
   single-shot tasks.

4. **CLI.** Always pass `--apply_chat_template`. Without it,
   `_build_initial_multi_turn_states` raises a runtime error explaining
   the requirement. The system role is load-bearing for any multi-turn
   benchmark; the flat-text fallback would silently change the task.

### Hook contract — clarifications

These details are implicit in the driver code but worth spelling out
for first-time multi-turn task authors:

- **What's in `history` when `next_user_turn` is called.** At
  `turn_idx == 0`, `history` is exactly what `build_initial_messages`
  returned (typically system + any prefilled user/assistant turns up
  to and including the last prefilled assistant message). At
  `turn_idx == k > 0`, `history` is the initial messages followed by
  the alternating sequence
  ``user_0, assistant_0, user_1, assistant_1, …, user_{k-1}, assistant_{k-1}``.
  The driver appends each ``user`` Message immediately after calling
  `next_user_turn` and each ``assistant`` Message immediately after
  the model generates. `should_stop` runs AFTER the assistant append
  for that turn, so `history[-1].role == "assistant"` when it's called.
- **Returning `None` immediately is valid.** If `next_user_turn(doc,
  history, 0)` returns `None`, the doc is marked done with zero model
  generations; no `Instance` is created and `process_results(doc, [])`
  receives an empty list. Tasks that filter docs by a runtime
  predicate (e.g. "skip rows with no trailing user turns") can use
  this as the natural skip mechanism — no early-exit hook needed.
- **`generation_kwargs` is shared across all turns.** The driver
  re-uses the task's single `generation_kwargs` block (`max_gen_toks`,
  `until`, `temperature`, etc.) on every turn. Per-turn variation
  isn't supported in v1 — see the Limits section. For tasks that need
  it (rare), add a `next_gen_kwargs(turn_idx)` hook on the task and a
  matching line in the driver.
- **Driver state is task-owned, not user-facing.** `task._instances`
  is mutated in place as turns are generated; `task._multi_turn_states`
  is built by `_build_initial_multi_turn_states` and torn down by the
  driver at the natural end or at every early-return. Don't read or
  write either from custom hook overrides — read `doc` and `history`
  (the parameters) instead. A hook that inspects `task._instances`
  mid-rollout sees the in-progress list and will produce
  rank-dependent behavior under DP, since each rank only holds its
  shard.

See `lm_eval/tasks/realguardrails/s_rules/task.py:SRulesTask` for a
worked example covering all four hooks plus an `__init__` that strips
the YAML `class` key and a `download` override that anchors relative
`data_files` paths against `Path(__file__).parent`.

## Comparison to alternatives considered

| Approach | Why not |
|---|---|
| Per-doc async dispatch (each doc rolled out independently via async/await) | Requires async support on every backend; HF doesn't expose it; vLLM offline mode is barrier-driven. Marginal throughput gain at major code-churn cost. |
| New `LM.multi_turn_generate()` method on each backend | Pushes rollout into every backend wrapper. 5+ implementations of the same outer loop. Breaks the principle that `LM` is a small, stateless interface. |
| Speculatively construct all turns with placeholder histories | Doesn't work — turn `t+1`'s prompt literally contains turn `t`'s output. The data dependency is intrinsic. |
| Multi-pass evaluator loop (chosen) | Backend-agnostic; reuses every backend's existing batching machinery; one sync barrier per turn (which is intrinsic anyway); cross-task batching maximizes occupancy. |

## Files touched

The complete change set:

+ `lm_eval/api/instance.py` — `OutputType` Literal extended with `"multi_turn_generate"`.
+ `lm_eval/api/task.py` — `ALL_OUTPUT_TYPES` extended; three new hooks on `ConfigurableTask` (`build_initial_messages`, `next_user_turn`, `should_stop`); `_build_initial_multi_turn_states` helper; `build_all_requests` early-returns for multi-turn; `construct_requests` returns `[]` for multi-turn.
+ `lm_eval/api/registry.py` — `DEFAULT_METRIC_REGISTRY` extended.
+ `lm_eval/api/multi_turn.py` (new) — `MultiTurnState` dataclass.
+ `lm_eval/evaluator.py` — `run_multi_turn_rollout(lm, mt_task_outputs)` driver; `_any_rank_has_pending` / `_gather_max_live` / `_clear_multi_turn_states` helpers; 4-line insertion in `evaluate()` to call the driver before the standard request dispatch.
+ `lm_eval/config/task.py` — multi-turn default `generation_kwargs` (`until=[]`); CLI `--gen_kwargs` override extended to multi-turn (upstream commit `504922b38`).
+ `lm_eval/tasks/realguardrails/s_rules/{task.py,utils.py,_s_rules_template.yaml}` — SRulesTask migration to the new path.
+ `lm_eval/tasks/realguardrails/README.md` — Limits section: multi-turn parallelism pointer + Anthropic/Google caveat.
+ `tests/test_multi_turn_rollout.py` (new) — 20 driver tests (10 single-rank + 4 distributed + 4 helper unit + asymmetric-DP + safety-cap) plus 1 documented real-DP smoke placeholder.
+ `tests/test_s_rules.py` — multi-turn hook + `process_results` tests added.

## Future work

Tracked as follow-up items, in rough priority order:

1. Judge integration from the rollout loop (unlocks RealGuardrails
   handwritten / distractors / Monkey Island and SysBench).
2. `next_gen_kwargs(turn_idx)` hook for per-turn generation params.
3. Bootstrap CI post-processor for paper-comparison reporting.
4. Balanced doc sharding (or work-stealing) for tail-heavy multi-turn
   benchmarks on HF + `accelerate launch` — current `doc_iterator`
   index-mod-rank can produce uneven tails when early-exit correlates
   with doc bucket.
5. Optionally factor `pending` / `batch_this_turn` tuples into
   `NamedTuple`s for legibility.
