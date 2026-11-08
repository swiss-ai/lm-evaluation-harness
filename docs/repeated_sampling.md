# Repeated sampling

Repeated scoring is opt-in and supported for API models: `apertus2-completions`
and the other TemplateAPI models (`local-completions`, `local-chat-completions`,
`openai-completions`, `openai-chat-completions`, LiteLLM). They send every
attempt as its own request, a sampled one with an offset seed (`anthropic-chat`
and `sglang-generate` send no seed). Local backends (HF, vLLM, SGLang, Megatron)
and API adapters outside TemplateAPI are rejected before any request. The
harness sends `m` attempts per document (the task's `repeats`), runs the task's
extraction and grader on each, then reduces the scores. `repeats` alone retains
the task's legacy scoring behavior.

> [!WARNING]
> All 560 leaf tasks of evals-post-train's thinking suite
> (`configs/apertus/tasks_posttrain_thinking.txt`) declare `repeat_support`, and a
> `repeat_filter` where needed. This means they are configured for repeat scoring;
> it does not test their datasets, judges or a live model. Other tasks need the
> same declarations first (see "Making a task repeat-scorable").

## Configuration

Three settings control repeat scoring, in a run file, on the command line or as
`simple_evaluate` arguments:

- `repeats`: attempts per document.
- `repeat_scoring`: the run policy, with `enabled`, `mean` and `pass_at_k`
  (metric name to a list of k).
- `task_overrides`: per leaf task, its own `repeats` and `repeat_scoring`.

### Run settings

Use a run YAML with the existing `lm-eval run --config eval.yaml` command:

```yaml
model: local-chat-completions
model_args:
  model: my-model
  base_url: http://localhost:8000/v1/chat/completions
tasks: [aime25, minerva_math500, ifeval]
apply_chat_template: true
gen_kwargs:
  temperature: 0.6
  do_sample: true
  max_gen_toks: 32768
repeats: 5
repeat_scoring:
  enabled: true
  mean: true
  pass_at_k: {}
task_overrides:
  aime25:
    repeats: 10
    repeat_scoring:
      pass_at_k:
        exact_match: [2, 5, 10]
  minerva_math500:
    repeat_scoring:
      pass_at_k:
        math_verify: [2, 5]
  ifeval:
    repeats: 1
output_path: results/repeated
```

### On the command line

`--repeat_scoring` and `--task_overrides` take JSON objects:

```bash
lm-eval run --model local-chat-completions \
  --model_args model=my-model,base_url=http://localhost:8000/v1/chat/completions \
  --tasks aime25,minerva_math500,ifeval --apply_chat_template \
  --gen_kwargs temperature=0.6,do_sample=True,max_gen_toks=32768 \
  --repeats 5 \
  --repeat_scoring '{"enabled": true, "mean": true}' \
  --task_overrides '{"aime25": {"repeats": 10, "repeat_scoring": {"pass_at_k": {"exact_match": [2, 5, 10]}}}}'
```

Command-line values update a run file's: with `--config eval.yaml`, `--repeats 8`
changes only the count, and `--repeat_scoring` or `--task_overrides` change only
the fields they name.

### Precedence

For each task, later layers win:

1. Task YAML defaults (`repeats`, `repeat_scoring`).
2. The run policy, for the tasks it covers (see the next section).
3. The task's entry in `task_overrides`.

- Missing policy fields inherit from the layer below. An explicit `pass_at_k`
  map replaces the inherited one, and `{}` clears it.
- The attempt count resolves the same way: the override's `repeats`, then the
  run-level `repeats` where it applies, then the YAML `repeats`, then 1.
- `enabled: false` turns repeat scoring off and keeps the resolved count
  (legacy repeats).
- Override names are leaf tasks after group expansion.

### Tasks without repeat settings / mixed runs

- A task without repeat settings runs as before: its YAML `repeats` (1 by
  default), scored by its own filters and metrics.
- The run policy covers only tasks that can be repeat-scored:
  - generative tasks
  - must declare `repeat_support`
  - must score one attempt at a time (no multi-sample filter, or a `repeat_filter` that selects a single-attempt one)
- Its `pass_at_k` applies to the binary metrics each task declares. Without
  `mean`, tasks that declare none of them stay unscored.
- Every other task keeps ordinary scoring and its own `repeats`, so a suite can
  mix both kinds. With repeat scoring on, the run-level `repeats` does not reach
  these tasks; with it off, it applies to all generative tasks (legacy repeats).
- If a group averages over a task the policy cannot cover, the policy leaves the
  whole group unscored, so the group score stays consistent.
- A group aggregates each repeat metric over the tasks that report it, as
  ordinary groups do. It reports a metric only on filters that every task
  defining the filter uses for repeat scoring.

### Errors and warnings

All checks run before any request unless marked otherwise.

| Level                    | Condition                                                                                                                                                                                                                                                                               |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Error                    | Repeat scoring enabled (task YAML or override) for a task that cannot support it: not generative, no `repeat_support`, or a multi-sample filter without a single-attempt `repeat_filter`                                                                                                |
| Error                    | A `pass_at_k` value outside 1 to the task's `repeats` (1 if no layer sets it, so any k above 1 fails)                                                                                                                                                                                   |
| Error                    | `pass_at_k` on a metric the task does not declare as `binary` (task default or override)                                                                                                                                                                                                |
| Error                    | A policy that requests no metrics (`mean: false` and no `pass_at_k`)                                                                                                                                                                                                                    |
| Error                    | Invalid settings: non-object policy, unknown policy keys, non-boolean `enabled`/`mean`, `pass_at_k` not a map of lists, `repeats` not a positive integer, override keys other than `repeats`/`repeat_scoring`, override names that are not selected leaf tasks, unknown `repeat_filter` |
| Error                    | Invalid declaration of a repeat-scored task: a `repeat_support` kind other than `binary`, `mean` or `native`, or a `native` metric without an aggregation                                                                                                                               |
| Error                    | A group aggregate whose tasks are only partly repeat-scored                                                                                                                                                                                                                             |
| Error                    | `predict_only`, or a model other than the API models above                                                                                                                                                                                                                              |
| Error (after generation) | A missing attempt, or a grader that omits a declared metric, returns a non-finite value, or returns a non-binary value for a `binary` metric                                                                                                                                            |
| Warning                  | An override with scoring fields on a task that stays unscored (ignored)                                                                                                                                                                                                                 |
| Warning                  | A run-level `pass_at_k` metric that no covered task declares as binary                                                                                                                                                                                                                  |
| Warning                  | A group metric its tasks report on no common filter (the group omits it)                                                                                                                                                                                                                |
| Warning                  | Repeats with greedy decoding (temperature 0): near-identical attempts                                                                                                                                                                                                                   |

### Making a task repeat-scorable

The same steps apply to existing and new tasks. Task authors declare
capabilities (these cannot be overwritten by per run configurations):

```yaml
repeat_support:
  exact_match: binary
  numeric_score: mean
  weighted_score: native
repeat_filter: score-first
```

- `repeat_support` maps each repeat-scored metric to `binary` (0 or 1
  correctness, enables pass@k), `mean` (a numeric average) or `native` (the
  task's own corpus aggregation, run once per repeat).
- `native` metrics need an aggregation that accepts one complete repeat and
  returns a scalar. An unavailable native score stays NaN; repeats are never
  dropped.
- `repeat_filter` selects an existing single-attempt extraction path. It is
  required when the task's filters include a multi-sample filter, one with `k`
  above 1 such as `take_first_k`.
- Optionally set `repeats` and `repeat_scoring` in the YAML as defaults for runs
  that set nothing for the task.
- For code tasks, extraction still builds a one-candidate test input; the task's
  normal code grader determines success.
- Deferred judges are called once per full repeat. Their dependencies and
  credentials remain required.

## Scores and artifacts

- Binary and numeric means average document scores within each repeat, then
  average repeats. For binary scores this estimates sampled pass@1.
- Native metrics retain their corpus/weighted aggregation within each repeat.
- With `c` successes among `m` attempts, pass@k is
  `1 - C(m-c, k) / C(m, k)`, averaged across documents. Each grader has its own `c`.
- Keys are `<metric>_mean_repeats,<filter>` and `<metric>_pass@k,<filter>`.
  Majority voting remains a task-specific legacy mode.
- Repeated-score standard errors are currently `N/A`. Per-repeat variation is
  available from scalar artifacts; it is not a confidence interval over problems.

Greedy repeats do not become independent samples merely because seeds differ.
Set decoding settings explicitly and compare protocols with the same prompt,
reasoning budget, grader and temperature. Provider seed support does not guarantee
bit-for-bit replay.

### What is logged

> [!NOTE]
> Every task's entry in the results' `configs` holds its effective `repeats` and
> `repeat_scoring` after all layers, including `enabled: false` for tasks that
> ran ordinary scoring.

- `repeat_protocols`, for each repeat-scored task: the protocol version, the
  resolved policy, the declared metrics, `repeats`, the number of documents,
  whether the task samples (temperature above 0) and the provider seed support.
  Group protocols add each leaf's budget and the tasks behind each metric.
- `repeat_results`: scalar task scores by repeat index, even with sample logging
  disabled. Tasks with fewer repeats are absent from later indices.
- Sample logs keep raw and filtered responses and score lists ordered by
  `repeat_indices`. Existing private-dataset redaction still applies.
- Generation lengths, truncation and parsing diagnostics keep their existing
  aggregation.

A task's protocol in the results file:

```json
"aime25": {
  "version": 1,
  "repeats": 8,
  "policy": {"enabled": true, "mean": true, "pass_at_k": {"exact_match": [2, 4, 8]}},
  "metrics": {"exact_match": "binary", "degeneration": "mean"},
  "documents": 30,
  "sampling_enabled": true,
  "seed_support": "request-seed"
}
```

## API support and validation

The supported API paths share one scorer. The backend tests mock the HTTP
endpoint or SDK, without a live model, and check attempts, seeds, retries and
caching:

| Backend                           | Attempts and seeds                                                                                                               | Unit tests                                                                                        |
|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Apertus 2 completions             | Seed offset by repeat index in its own payload; native reasoning replay                                                          | `tests/models/test_apertus2.py`: synthetic native format, sync/async, retries, multi-turn         |
| Local/OpenAI completions and chat | Sampled attempts: seed offset by repeat index before batching, kept on retries; greedy requests keep their seed and stay batched | `tests/models/test_api.py`: mocked HTTP, concurrency, retries, attempt caching and partial resume |
| LiteLLM chat                      | Shared attempt identity; SDK may drop unsupported seed parameters                                                                | `tests/models/test_repeat_backends.py`: mocked SDK, sync/async dispatch                           |

Scoring itself (means, pass@k, precedence, groups and errors) is tested in
`tests/test_repeat_scoring.py`, independent of the backend. Model availability,
accepted sampling fields and reasoning formats differ between endpoints, so
check a new provider with a short run before relying on it.

### Caching (`--use_cache`)

`--use_cache <file>` stores model answers so a rerun does not request them again.
With repeat scoring:

- Each attempt is stored on its own, so attempts stay distinct across reruns. A
  rerun requests only missing attempts; raising `repeats` keeps the stored ones.
- Stored attempts match on model name and arguments, chat template settings,
  task, document, attempt, turn, prompt and generation settings. Concurrency,
  timeouts, retries, rate limits and credentials can change between runs.
- Answers are written when a generation batch returns. Failed requests are not
  stored and are retried on the next run; empty answers are stored.
- A retry reuses its attempt and seed.
- Not cached: `apertus2-completions` (its reasoning replay must see every answer
  as produced) and models passed as objects (no name to key on). Local backends
  keep the prompt-keyed cache of greedy generations.

`--cache_requests` caches prepared requests, not answers; changing `repeats`
rebuilds them.

## Note: tasks with their own repeats

Some tasks generated several samples before repeat scoring existed and score
them with their own filters or metrics:

- `aime24` and `aime25`: `repeats: 8` with score-first, mean@8, maj@8 and pass@8
  filters.
- `humaneval` and the tasks that include it, such as `humaneval_instruct`:
  `repeats: 20`, with pass@1 and pass@10 from the task's own metric.
- `gsm8k_cot_self_consistency`: 64 samples with maj@64 and maj@8.
- `gsm8k_cot_zeroshot_self_consistency` and `minerva_math500_self_consistency`:
  4 samples with mean@4, maj@4 and pass@4 filters.
- `humaneval_instruct_self_consistency` and the `mbpp*_instruct_self_consistency`
  tasks: 8 samples, with an `all` filter that scores them together.

How they behave:

- Without repeat scoring they run exactly as before.
- With repeat scoring enabled for them, their `repeat_filter` selects the
  single-attempt path, each attempt is graded on its own, and their multi-sample
  filters are not used.
- A `repeats` set by the run or an override replaces their YAML count; otherwise
  the YAML count stays (8 attempts for aime).
- For aime with 8 attempts, `exact_match_mean_repeats` and `exact_match_pass@8`
  correspond to its own mean@8 and pass@8 filters.
- Tasks with a multi-sample filter and no `repeat_filter`
  (`gsm8k_cot_self_consistency`, `gsm8k_platinum_cot_self_consistency`) stay
  legacy under a run policy and still generate their full sample count.

Open question: should the legacy multi-sample paths (their mean@k and pass@k
filters and sample counts) be removed once repeat scoring is the standard,
keeping majority voting as the only task-specific mode? That would leave one way
to compute each score, but it changes the legacy results of these tasks.
