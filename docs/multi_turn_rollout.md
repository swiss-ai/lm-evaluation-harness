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
`for reqtype, reqs in requests.items()` dispatch (around
`evaluator.py:583`). Single-shot tasks are unaffected; the new branch
fires only if at least one task in the run has
`output_type: multi_turn_generate`.

After the driver returns, the standard post-process path
(`evaluator.py:608+`):
1. `task.apply_filters()` — operates on each Instance's `resps` (a
   singleton list per turn), same as single-shot. Default `take_first`
   produces `inst.filtered_resps["none"]` = the string response.
2. `instances_by_doc_id` groups by `doc_id`, sorts by `idx`, calls
   `task.process_results(doc, [filtered_resp_t0, filtered_resp_t1, …])`.
3. Sample logger writes one record per doc with `arguments`/`resps` as
   variable-length lists (one entry per generated turn). No logger
   change needed; the shape is already supported by `multiple_choice`.

## Backend compatibility

| Backend | Per-turn batching | Status |
|---|---|---|
| HF transformers | `_detect_batch_size` runs once at first call; cached batch size reused across turns. Smaller late-turn batches due to early-exit (acceptable). | ✓ Supported |
| vLLM | Continuous batching within each turn; synchronous barrier between (intrinsic). | ✓ Supported |
| `local-chat-completions` + `num_concurrent` | Concurrent dispatch within a turn; same sync barrier. | ✓ Supported |
| OpenAI / Anthropic / Google API backends | Same as `local-chat-completions`. | ✓ Supported |

No backend wrapper changes are required because each turn issues exactly
one `generate_until(list[Instance])` call — the same interface single-shot
eval uses.

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

### Distributed multi-rank runs not supported

The driver hard-errors when `lm.world_size > 1` with `NotImplementedError`.

**Why:** the per-rank `while pending:` loop terminates at different
times across ranks when (a) ranks hold docs with different turn counts
or (b) some ranks hold no docs at all (small dataset + `--limit`).
Collective ops (`accelerator.gather`, `wait_for_everyone`) require every
rank to participate every iteration — asymmetric exit deadlocks.

**What works today:** single-rank (CPU, MPS, single-GPU), single-process
multi-GPU via vLLM `tensor_parallel_size=N`, single-process model-parallel
via HF `parallelize=True`, and `local-chat-completions` against any
backend (vLLM server, openai, etc.) with `num_concurrent`.

**Path to enable distributed:** add a cross-rank "any pending?" gather
at the top of each turn (one `accelerator.gather(torch.tensor(int(not pending)))`),
pad idle ranks with no-op requests to participate in the within-turn
batched `generate_until`, terminate only when every rank reports done.
Estimated effort: ~half a day. See `lm_eval/evaluator.py:run_multi_turn_rollout`
inline comments.

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
| HF + accelerate launch DP | not supported (yet) | — |

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
+ `lm_eval/evaluator.py` — `run_multi_turn_rollout(lm, mt_task_outputs)` driver; 4-line insertion in `evaluate()` to call it before the standard request dispatch.
+ `lm_eval/tasks/realguardrails/s_rules/{task.py,utils.py,_s_rules_template.yaml}` — SRulesTask migration to the new path.
+ `tests/test_multi_turn_rollout.py` (new) — 11 driver tests.
+ `tests/test_s_rules.py` — multi-turn hook + `process_results` tests added.

## Future work

Tracked as follow-up items, in rough priority order:

1. Cross-rank synchronisation for distributed runs (lifts the
   single-rank limit).
2. Judge integration from the rollout loop (unlocks RealGuardrails
   handwritten / distractors / Monkey Island and SysBench).
3. `next_gen_kwargs(turn_idx)` hook for per-turn generation params.
4. Bootstrap CI post-processor for paper-comparison reporting.
5. Optionally factor `pending` / `batch_this_turn` tuples into
   `NamedTuple`s for legibility.
