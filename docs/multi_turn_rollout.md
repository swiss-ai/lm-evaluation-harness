# Multi-Turn Rollout

`output_type: multi_turn_generate` is a task-level capability for
**multi-turn benchmarks** — datasets where the model must respond to a
sequence of turns, with each generation conditioned on the model's own
prior responses. It powers MT-Bench (turn 1 + turn 2), BFCL v3's
multi-turn function-calling categories, and the S-RuLES sub-suite of
`realguardrails`, and is the foundation for further multi-turn suites
(SysBench, Multi-IF turns 2-3, MT-Eval, …).

This document describes the architecture, the two task contracts, the
limits, and how to add a new multi-turn benchmark.

## Why this exists

The single-shot task model assumes one generation per doc. Multi-turn
benchmarks need turn `t+1`'s prompt to contain turn `t`'s response, so
the rollout cannot be expressed as one upfront prompt. `multi_turn_generate`
solves this with one reusable evaluator-level driver, reusing every
backend's existing `generate_until(list[Instance]) -> list[str]`
interface — no backend wrapper changes.

## Architecture in one sentence

A dedicated evaluator phase (`run_multi_turn_rollout`) owns the
turn-by-turn loop: it asks each task for the next prompt, batches all
pending episodes across all multi-turn tasks into a single
`generate_until` call per wave, feeds each response back into the task's
state, and finally stores one rollout-result object on the doc's
`Instance` for the standard `process_results` path to consume.

## Data dependency, and why each turn is a synchronization barrier

Turn `t+1`'s prompt contains turn `t`'s response. This dependency is
intrinsic; there is no way to dispatch turn `t+1` until the backend has
returned turn `t`. Per-wave synchronization is therefore unavoidable —
but **within** each wave, all pending episodes across all multi-turn
tasks share a single batched `generate_until` call. vLLM continuous
batching, HF auto-batch, and `local-chat-completions` `num_concurrent`
operate inside one wave exactly as in single-shot eval.

## The two task contracts

There is **one** driver and **one** core contract. A convenience layer
on top covers the common "scripted" case so simple benchmarks stay
short.

### Core contract (general / agentic)

The driver consults these five hooks. The task owns its episode state
(any object it likes) and returns the next **fully-rendered** prompt, so
agentic tasks — e.g. BFCL, which issues several model calls per user turn
and feeds tool-execution results back into the conversation — are
expressible.

```python
def init_multiturn_state(self, doc, ctx, gen_kwargs,
                         apply_chat_template=False, chat_template=None):
    """Return the per-doc episode state object."""

def multiturn_next_request(self, state) -> tuple[str, dict] | None:
    """Return (prompt, gen_kwargs) for the next model call, or None when
    this episode has no more requests this step."""

def multiturn_consume_response(self, state, response: str) -> None:
    """Advance the episode given the model's response (parse it, run
    tools, append to history, decide whether the episode is done)."""

def multiturn_is_done(self, state) -> bool:
    """Whether the episode is finished."""

def multiturn_result(self, state):
    """The object handed to process_results for this doc."""
```

`chat_template` is the model's `lm.apply_chat_template` callable (or
`None` if `--apply_chat_template` was not passed); the task renders its
own history with it. See `lm_eval/tasks/bfcl_v3/task.py` and
`lm_eval/tasks/mt_bench/task.py` for worked examples.

### Scripted convenience hooks

For *scripted* benchmarks — a fixed sequence of user turns with exactly
one model generation per turn — override these three instead, and inherit
the five core hooks from `ConfigurableTask`'s defaults (which drive a
scripted episode through these methods):

```python
def build_initial_messages(self, doc) -> list[Message]:
    """[system, ...prefilled turns up to the last prefilled assistant
    turn]. Does NOT include the queued user turns."""

def next_user_turn(self, doc, history, turn_idx) -> str | None:
    """The next user turn the model should answer; None when exhausted.
    turn_idx == 0 is the first model generation."""

def should_stop(self, doc, history, turn_idx) -> bool:
    """Called AFTER appending the model's assistant response for turn_idx.
    Return True to early-exit this doc. Default: False."""
```

The default core-hook implementations use `MultiTurnState`
(`lm_eval/api/multi_turn.py`) for episode state, render history with the
chat template each turn, and return the list of assistant turns from
`multiturn_result`. See `lm_eval/tasks/realguardrails/s_rules/task.py`
for a worked example (system + prefilled context, trailing user turns,
early-exit on scenario failure).

> Scripted tasks **require** `--apply_chat_template`: without the chat
> template the per-doc system prompt becomes a plain-text prefix and the
> task silently changes meaning. `init_multiturn_state` raises if it is
> missing.

## The driver (`run_multi_turn_rollout`, `lm_eval/evaluator.py`)

```text
episodes = [one per (doc, repeat) for every multi_turn_generate Instance]
for each episode: state = task.init_multiturn_state(doc, ctx, gen_kwargs, …)

for _ in range(max_steps):                 # max_steps = max task.MAX_MULTITURN_STEPS (default 64)
    wave = []
    for episode in episodes:
        if task.multiturn_is_done(state): continue
        nxt = task.multiturn_next_request(state)        # (prompt, gen_kwargs) | None
        if nxt is None: continue
        wave.append(Instance("generate_until", arguments=nxt, …))
    if not wave: break
    responses = lm.generate_until(wave)                 # one batched call across all episodes
    for response, episode in zip(responses, wave):
        task.multiturn_consume_response(state, response)

for episode in episodes:
    instance.resps.append(task.multiturn_result(state)) # one result object per doc
```

Key properties:

+ One synchronous barrier per wave (intrinsic to the data dependency).
+ Cross-task batching at the per-wave level maximizes occupancy.
+ Reuses `lm.generate_until` unchanged — no backend wrapper changes.
+ One `Instance` per doc; the rollout result is appended to its `resps`,
  so `process_results(doc, [result])` receives the result object at
  `results[0]` — the same plumbing as a single-shot `generate_until`
  task.

`MAX_MULTITURN_STEPS` is a per-task class attribute (a defensive cap;
MT-Bench sets `2`, BFCL `128`, the default is `64`). It also bounds the
number of waves for agentic tasks that issue multiple model calls per
user turn.

### Integration with the single-shot path

`multi_turn_generate` requests are popped out of the standard
`for reqtype, reqs in requests.items()` dispatch loop in `evaluate()` and
handled by `run_multi_turn_rollout`. Single-shot tasks are unaffected;
the branch fires only if at least one task in the run has
`output_type: multi_turn_generate`. After the driver returns, the
standard post-process loop runs unchanged:
`apply_filters` (default `take_first`) → group by `doc_id` →
`process_results(doc, [filtered_result])`.

## Generation parameters & overrides

Every model call is a `generate_until` call built from the task's single
`generation_kwargs` block, so sampling params behave as for
`generate_until` tasks:

+ **Defaults** (YAML omits `generation_kwargs`): greedy — `temperature=0.0`,
  `do_sample=False`, `max_gen_toks=256` — with **`until=[]`**. This is the
  same greedy default as `generate_until` except for `until`:
  `generate_until` defaults to `until=[fewshot_delimiter]` (`"\n\n"`), which
  would truncate a multi-turn response at the first blank line, so
  multi-turn instead relies on the model's own turn-end (EOS) token.
+ **CLI overrides are respected**: `--gen_kwargs key=value` *merges* into the
  task's `generation_kwargs` (CLI wins on conflicting keys), gated on
  `output_type in ("generate_until", "multi_turn_generate")` in
  `simple_evaluate`.
+ Tasks that need per-doc or per-turn variation (e.g. MT-Bench's
  category-based temperature) set it inside their own core hooks — the
  task owns its `gen_kwargs` and can vary it per episode/turn.

### Turn termination depends on the model's EOS — mind mis-configured fine-tunes

Each turn stops at the model's EOS token or any `until` string. **If a
model's tokenizer `eos_token` is not its chat turn-end token, generation
runs on past the turn boundary** — the model emits its turn-end token, the
backend does not treat it as a stop, and it keeps generating (usually
hallucinating the next user/assistant turns up to `max_gen_toks`),
silently corrupting the per-turn response the evaluator scores.

Observed with fine-tunes whose `eos_token` is `<|endoftext|>` while the
chat template ends turns with `<|im_end|>`. Remedies:

+ **vLLM**: add the turn-end token as a stop, `--gen_kwargs 'until=<|im_end|>'`
  — vLLM detokenizes with special tokens visible, so the stop string matches.
+ **HF**: needs a **token-level** stop (a stop *string* does NOT work — HF
  strips special tokens from decoded output). Pass the turn-end **token id**
  instead: `--gen_kwargs 'eos_token_id=151645'`, or correct the model's
  `generation_config.json` `eos_token_id`.

Well-formed chat models (tokenizer `eos_token` == chat turn-end, e.g. Llama
`<|eot_id|>` / Qwen2.5 `<|im_end|>`) need none of this.

## Backend compatibility

Each wave issues exactly one `generate_until(list[Instance])` call — the
same interface single-shot eval uses — so all backends work without
wrapper changes, and each backend's own batching applies within a wave:

| Backend | Within-wave batching |
|---|---|
| HF transformers (CPU / MPS / single GPU, or `parallelize=True`) | auto-batch; batch size detected once and reused across waves |
| vLLM `tensor_parallel_size` / `data_parallel_size` | continuous batching across all replicas |
| SGLang / `local-chat-completions` / OpenAI-compatible | `num_concurrent` concurrent dispatch within a wave |

Backends without `apply_chat_template` (e.g. Anthropic / Google native
backends) are not supported for scripted multi-turn tasks; route through
`local-chat-completions` against a wrapper instead.

## Limits

These are **intentional trade-offs** of the unified single-driver design.

### Single-node only (no distributed multi-turn)

`run_multi_turn_rollout` raises `NotImplementedError` for `world_size > 1`
(HF `accelerate launch` data-parallel). Run multi-turn tasks with
`world_size == 1`; backends with internal parallelism — vLLM
`data_parallel_size`/`tensor_parallel_size`, SGLang /
`local-chat-completions` / API backends with `num_concurrent` — expose
`world_size == 1` to the harness and route the per-wave batch internally,
so they are unaffected. (Cross-rank `accelerate launch` support was
dropped when the two upstream multi-turn implementations were unified;
it can be reintroduced on the driver if a workload needs it.)

### `--cache_requests` (Instance cache)

The per-doc placeholder `Instance` is built before the rollout runs, so
the request cache stores only that placeholder, not the per-turn prompts.
The response cache (`--use_cache`) still works per turn — each turn's
prompt is distinct so cache keys differ naturally.

### Per-doc history is rebuilt every turn (HF backend)

Each turn re-renders the chat template with the full growing history; on
HF this re-tokenizes the whole prefix every turn — O(N²) tokenization
across an N-turn rollout. Negligible for short rollouts.

### Sample-log shape

`process_results` receives the rollout-result object at `results[0]`
(for scripted tasks, the list of assistant turns). Downstream
paper-comparison scripts that assume one string per doc need updating.

## Adding a new multi-turn benchmark

1. **YAML.** Set `output_type: multi_turn_generate`. Generation params,
   metric list, dataset path, and (optionally) `class: !function task.MyTask`
   work as in single-shot. Always run with `--apply_chat_template`.

2. **Task class.** Subclass `ConfigurableTask`. For a **scripted**
   benchmark, override the three convenience hooks:

   ```python
   from lm_eval.api.task import ConfigurableTask
   from lm_eval.api.utils import Message

   class MyMultiTurnTask(ConfigurableTask):
       def build_initial_messages(self, doc):
           return [Message("system", doc["system_prompt"])]

       def next_user_turn(self, doc, history, turn_idx):
           turns = doc["user_turns"]
           return turns[turn_idx] if turn_idx < len(turns) else None

       def should_stop(self, doc, history, turn_idx):
           return False
   ```

   For an **agentic** benchmark (multiple model calls per turn, tool
   execution, custom rendering), override the five core hooks instead and
   keep whatever episode state object you need. See
   `lm_eval/tasks/bfcl_v3/task.py`.

3. **`process_results(doc, results)`.** `results[0]` is the object your
   `multiturn_result` returned (for scripted tasks, the list of assistant
   responses in turn order). Reconstruct the full conversation if your
   scorer needs it.

## Comparison to alternatives considered

| Approach | Why not |
|---|---|
| Per-doc async dispatch | Requires async support on every backend; HF doesn't expose it; vLLM offline mode is barrier-driven. |
| New `LM.multi_turn_generate()` on each backend | Pushes the rollout loop into every backend wrapper (5+ copies). |
| Speculatively construct all turns upfront | Impossible — turn `t+1`'s prompt literally contains turn `t`'s output. |
| Single evaluator driver (chosen) | Backend-agnostic; reuses every backend's batching; one sync barrier per wave (intrinsic anyway); cross-task batching maximizes occupancy. |

## Files

+ `lm_eval/api/instance.py` — `OutputType` includes `"multi_turn_generate"`.
+ `lm_eval/api/task.py` — five core hooks (abstract on `Task`, scripted-convenience
  defaults on `ConfigurableTask`) + three scripted hooks; `construct_requests`
  builds one placeholder Instance per doc.
+ `lm_eval/api/multi_turn.py` — `MultiTurnState` (scripted convenience layer state).
+ `lm_eval/api/registry.py` — `DEFAULT_METRIC_REGISTRY` entry.
+ `lm_eval/evaluator.py` — `run_multi_turn_rollout` driver + dispatch.
+ `lm_eval/config/task.py` — multi-turn default `generation_kwargs` (`until=[]`)
  and CLI `--gen_kwargs` override.
+ Tasks: `lm_eval/tasks/mt_bench/`, `lm_eval/tasks/bfcl_v3/`,
  `lm_eval/tasks/realguardrails/s_rules/`.
+ Tests: `tests/test_multiturn.py`, `tests/test_mt_bench.py`,
  `tests/test_bfcl_v3.py`, `tests/test_s_rules.py`.
