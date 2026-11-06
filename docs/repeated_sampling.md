# Repeated sampling

Repeated scoring is opt-in and supported for API models: `apertus2-completions`
and the other TemplateAPI models (`local-completions`, `local-chat-completions`,
`openai-completions`, `openai-chat-completions`, LiteLLM). They send every
attempt as its own request, a sampled one with an offset seed (`anthropic-chat`
and `sglang-generate` send no seed). Local backends (HF, vLLM, SGLang, Megatron)
and API adapters outside TemplateAPI are rejected before any request. The harness creates `m` attempts, runs the task's
extraction and grader on each, then reduces the scores. `repeats` alone retains
the task's legacy scoring behavior.

## Configuration

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

`--repeats`, `--repeat_scoring` and `--task_overrides` are also CLI options;
the latter two take JSON objects. `simple_evaluate` accepts the same Python
arguments. Task YAML can supply `repeats` and `repeat_scoring` defaults. The
run-level `repeat_scoring` applies to generative tasks that declare
`repeat_support` and score one attempt at a time (no multi-sample filter, or a
`repeat_filter` that selects a single-attempt one), except the tasks of a group
that aggregates over a task that does not; the others keep ordinary scoring, so
mixed suites run. Its
`pass_at_k` applies to the binary metrics each task declares; without `mean`,
tasks that declare none of them keep ordinary scoring. A run-level
`repeats` applies to the tasks it repeat-scores, or to all generative tasks when
it does not enable repeat scoring (legacy repeats). Other tasks keep their own
count unless a task override sets one.

Precedence is per-task override, run policy, task default. Missing policy fields
inherit; an explicit `pass_at_k` map replaces the whole inherited map, including
`{}` to clear it. CLI fields override run-file fields. Overrides name selected
leaf tasks after group expansion. Unknown names, unsupported tasks, invalid
counts, nonbinary pass metrics, `predict_only` and non-API models fail before
generation, and so does a group aggregate that only some of its tasks
repeat-score (enable repeat scoring for all of a group's tasks or none). As
for ordinary groups, a group aggregates each metric over the tasks that report
it, and each filter covers the tasks that define it, which must all select it
for repeats; a warning before generation names a group metric that this leaves
without a filter.
`enabled: false` selects legacy scoring and keeps the resolved repeat count.

Task authors declare capabilities, which runs cannot override:

```yaml
repeat_support:
  exact_match: binary
  numeric_score: mean
  weighted_score: native
repeat_filter: score-first
```

`repeat_filter` optionally selects an existing single-attempt extraction path.
It is required when the default filter set includes a fixed multi-sample filter.
For code tasks, extraction still builds a one-candidate test input; the task's
normal code grader determines success. `native` metrics require an aggregation
function that accepts one complete repeat and returns a scalar. Deferred judges
are called once per full repeat. Their dependencies and credentials remain
required. An unavailable native score remains NaN; repeats are never dropped.

## Scores and artifacts

- Binary and numeric means average document scores within each repeat, then
  average repeats. For binary scores this estimates sampled pass@1.
- Native metrics retain their corpus/weighted aggregation within each repeat.
- With `c` successes among `m` attempts, pass@k is
  `1 - C(m-c, k) / C(m, k)`, averaged across documents. Each grader has its own `c`.
- Keys are `<metric>_mean_repeats,<filter>` and `<metric>_pass@k,<filter>`.
  Majority voting remains a task-specific legacy mode.
- `repeat_results` contains scalar task scores by repeat index, even with sample
  logging disabled. Tasks with fewer repeats are absent from later indices.
- `repeat_protocols` records version, policy, capabilities, repeats, documents,
  whether the task samples (temperature above 0) and provider seed support.
  Group protocols include leaf budgets and metric contributors.
- Sample logs retain raw/filtered responses and score lists ordered by
  `repeat_indices`. Existing private-dataset redaction still applies. Generation
  lengths, truncation and parsing diagnostics retain their existing aggregation.
- Repeated-score standard errors are currently `N/A`. Per-repeat variation is
  available from scalar artifacts; it is not a confidence interval over problems.

Greedy repeats do not become independent samples merely because seeds differ.
Set decoding settings explicitly and compare protocols with the same prompt,
reasoning budget, grader and temperature. Provider seed support does not guarantee
bit-for-bit replay.

## API support and validation

The supported API paths share one scorer. The following coverage is offline;
no live provider has been certified for this change.

| Backend | Attempt/seed behavior | Offline evidence |
| --- | --- | --- |
| Apertus 2 completions | Seed offset by repeat index in its own payload; native reasoning replay | Synthetic native format, sync/async, retries, multi-turn |
| Local/OpenAI completions and chat | Seed offset by repeat index before batching; retries keep it; requests with one seed stay batched | Mock HTTP, concurrency, cache hits/partial resume |
| LiteLLM chat | Shared attempt identity; SDK may drop unsupported seed parameters | Mock SDK sync/async dispatch |

Provider-specific model availability, accepted sampling fields and reasoning
formats still require an endpoint smoke test.

API generations are cached per attempt when generation returns, keyed by model namespace,
task/document/repeat/turn identity and request arguments; Apertus 2 and models
passed as objects are not cached. Other models keep the prompt-keyed caching of
greedy generations. Missing attempts are not cached; empty answers are. The
request cache key includes the repeat count. Retries do not consume a new
attempt. A missing attempt of a repeat-scored task fails the evaluation.

The thinking-suite configuration inventory covers all 560 leaf tasks in the
post-training suite: knowledge/math, code, instruction following, multilingual,
safety, BFCL and deferred judges. This is declaration and extraction coverage,
not proof that every private dataset, judge or live model endpoint is available.
Other tasks must declare compatible capabilities before enabling this mode.
