# Repeated sampling

Repeated scoring is opt-in and independent of the model backend. The harness
creates `m` attempts, runs the task's extraction and grader on each, then reduces
the scores. `repeats` alone retains the task's legacy scoring behavior.

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
arguments. Task YAML can supply `repeats` and `repeat_scoring` defaults.

Precedence is per-task override, run policy, task default. Missing policy fields
inherit; an explicit `pass_at_k` map replaces the whole inherited map, including
`{}` to clear it. CLI fields override run-file fields. Overrides name selected
leaf tasks after group expansion. Unknown names, unsupported tasks, invalid
counts and nonbinary pass metrics fail before generation. `enabled: false`
selects legacy scoring and keeps the resolved repeat count.

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
- `repeat_protocols` records version, policy, capabilities, budgets, completed
  attempts and available provider sampling/seed metadata. Group protocols include
  leaf budgets and metric contributors. Incompatible policies or filters cannot
  silently remove expected contributors from a reported group score.
- Sample logs retain raw/filtered responses and score lists ordered by
  `repeat_indices`. Existing private-dataset redaction still applies. Generation
  lengths, truncation and parsing diagnostics retain their existing aggregation.
- Repeated-score standard errors are currently `N/A`. Per-repeat variation is
  available from scalar artifacts; it is not a confidence interval over problems.

Greedy repeats do not become independent samples merely because seeds differ.
Set decoding settings explicitly and compare protocols with the same prompt,
reasoning budget, grader and temperature. Provider seed support does not guarantee
bit-for-bit replay. OpenAI reasoning adapters record their forced sampling mode.

## API support and validation

All registered API paths use the same scorer. The following coverage is offline;
no live provider or served Apertus checkpoint has been certified for this change.

| Backend | Attempt/seed behavior | Offline evidence |
| --- | --- | --- |
| Apertus 2 completions | Explicit episode identity; one seed offset; native reasoning replay | Synthetic native format, sync/async, retries, reordered attempts, multi-turn |
| Local/OpenAI completions and chat | Attempt seed before batching; retries keep it | Mock HTTP, concurrency, cache hits/partial resume and forced sampling |
| Azure OpenAI chat | Shared HTTP path, forwarded sampling/concurrency; `api-key` auth | Constructor and payload regression; shared transport tests |
| LiteLLM chat | Shared attempt identity; SDK may drop unsupported seed parameters | Mock SDK sync/async dispatch |
| Anthropic chat/completions | Independent requests; no seed support | Block extraction and bounded retry tests; payload review |
| SGLang HTTP | Maps request seed to `sampling_seed` | Payload regression; shared transport tests |
| GGUF | Uses task seed if supplied; one call per attempt | Reordered request and parameter regression |
| TextSynth | Uses task seed if supplied; zero means unseeded | Reordered request and parameter regression |
| Watsonx | Maps seed to `random_seed`; preserves model defaults | Mock text dispatch and task-parameter regression |

The Azure header follows the [Azure OpenAI REST reference](https://learn.microsoft.com/en-us/azure/foundry/openai/latest).
Provider-specific model availability, accepted sampling fields and reasoning
formats still require an endpoint smoke test. HF/vLLM/native backends can use the
scorer for eligible tasks, but are outside this API certification milestone.

Response caching is opt-in for stateless API adapters, keyed by model namespace,
task/document/repeat/turn identity and request arguments. Legacy argument-only
completion entries are not reused. Apertus's stateful generation and native
backends bypass response caching. Request caches include task/generation policy,
repeat count, renderer/template arguments, dataset fingerprints and few-shot seed.
Retries do not consume a new attempt. Exhausted transport errors, missing
responses and incomplete response counts fail the evaluation.

The thinking-suite configuration inventory covers all 560 leaf tasks in the
post-training suite: knowledge/math, code, instruction following, multilingual,
safety, BFCL and deferred judges. This is declaration and extraction coverage,
not proof that every private dataset, judge or live model endpoint is available.
Other tasks must declare compatible capabilities before enabling this mode.
