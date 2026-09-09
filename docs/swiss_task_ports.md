# Swiss AI task ports on upstream v0.4.13

Branch `yxu/dev` starts at EleutherAI v0.4.13, commit
`ddd67220430a2470529f25fd5c05a576ca1057a0`. Task source is
[swiss-ai/lm-evaluation-harness at 51d6f4b](https://github.com/swiss-ai/lm-evaluation-harness/tree/51d6f4b62bf20e9a29dffa694b163d4f19889927).
This branch selectively ports the definitions below without merging the Swiss
fork's engine, backends, or judge service APIs. Source copyright notices remain
with the imported files.

| Task | Imported source directory | Scope |
| --- | --- | --- |
| `global_mmlu_gen_0shot` | `global_mmlu/gen_0shot` | Global-MMLU-Lite, 15 languages, generated answers |
| `truthfulqa_multilingual_mc2` | `okapi/truthfulqa_multilingual/truthfulqa_multilingual_mc2.yaml` | Group existing upstream tasks in 31 languages |
| `include_base_44_gen_0shot` | `include/gen_0shot` | INCLUDE-44, generated answers |
| `include_base_new_45_gen_0shot` | `include_new/gen_0shot` | Swiss AI's additional 45-language dataset |
| `switzerland_qa_0shot` | `switzerland_qa/gen_0shot` | Five languages, generated answers |
| `blend_sample` | `blend/sampled` and shared processors | Swiss sampled BLEnD dataset, 16 regions |
| `cultural_bench` | `cultural_bench` | Swiss easy/hard multiple-choice configuration |
| `multi-if` | `multi_if` | **First turn only**, as in the Swiss source |
| `alpaca_eval` | `alpaca_eval` | Generated answers plus optional pairwise judge scoring |

## Compatibility changes

- Port the `ordered_regex` filter required by the four generative multilingual
  families. Patterns run in priority order; `group_select` chooses the match
  occurrence within the first matching pattern. The imported tasks use `-1`.
  Prompt wording, extraction patterns, language sets and group weighting remain
  as in the pinned source.
- Replace INCLUDE-44's parent-relative Python function strings with qualified
  module imports supported by upstream's YAML loader.
- Materialize Multi-IF's already constructed document list directly, avoiding
  an unnecessary disk-backed generator cache. Record `first_turn_only` in task
  metadata. This task does not measure following instructions across three turns.
- Adapt AlpacaEval to the official scorer package with explicit annotator
  configuration and retained run artifacts. Remove the hardcoded CSCS service
  and Swiss-only resolver/rate-limiter imports. The scorer's percentage is
  converted to the harness's fraction metric. See its task README for details.
- Fix upstream `mmlu_flan_cot_zeroshot` group aggregation: its generated-answer
  tasks emit `exact_match`, but the group requested `acc`. All extraction
  filters now produce weighted group scores.
- Fix upstream MBPP extraction after the 8B evaluation exposed lost initial
  Python keywords. The task's `gen_prefix` already opens a Python fence; the
  old extractor prepended another bare fence and misread `def`, `import`,
  `async`, or `class` as its language label. Preserve bare completion text up
  to the closing fence, including valid code without a closing fence, and
  strip a leading opening fence only when it is actually present. This does
  not remove explanatory prose or repair malformed Python. Task prompts,
  generation settings, and the pass-at-one metric remain unchanged. Saved
  completions from the affected runs can be rescored without new inference;
  their original scores and filtered samples describe the old extractor.
- Apply repository formatting to imported code and YAML. No alternative Swiss
  prompt variants or task generators are needed for these ports.

## Instruction-model protocol corrections

The six task families below incorporate the audited Swiss report configuration
at `4ac31dae8f4070be2cd85b92c8497f81b4f86c87`, including the narrow math
extraction fix from [Swiss PR #88](https://github.com/swiss-ai/lm-evaluation-harness/pull/88).
Upstream's evaluator, backends and newer correctness fixes remain in place.

| Task | Protocol on this branch | Comparison metric |
| --- | --- | --- |
| `mmlu_flan_cot_zeroshot` | All 57 subjects request concise reasoning and a final option; ordered extraction prioritizes explicit answers | `exact_match,ordered-extract` |
| `mmlu_pro` | Bracketed choices and ordered extraction; five demonstrations, 2048 generated tokens | `exact_match,ordered-extract` |
| `hendrycks_math` | Six fixed demonstrations, 2048 generated tokens; shared upstream Minerva scorer | `math_verify,none` |
| `minerva_math` | Four corrected upstream demonstrations, explicit 1024 generated tokens | `math_verify,none` |
| `drop` | Three training demonstrations, answer-only instruction, assistant answer prefix and prompt-boundary stops | `f1,none` |
| `mathqa` | Assistant answer prefix; retain upstream's `regisss/math_qa` data mirror | `acc,none` |

MMLU-Flan retains the backend's default generation budget (256 in the suite's
vLLM backend). MMLU-Pro retains upstream's separation of demonstration questions
and assistant answers; the older Swiss helper put the answer inside the user
message. Existing alternative extraction metrics remain available in raw results.

Both MATH families accept the final-answer phrase without its optional suffix,
then fall back to the last boxed answer. They share upstream's corrected LaTeX
demonstrations. Unlike the Swiss regex, the phrase matcher treats its final
period as optional literal punctuation, so an unpunctuated `42` stays `42`
instead of losing its final digit. The shared scorer retains upstream's
tuple/thousands normalization, exact-string equality and
full-solution Math Verify scoring. The two additional Hendrycks demonstrations
come from Swiss, with accidental backspace escapes repaired. The same narrow
thousands-separator check is applied to the leaderboard math helper, whose
existing tests exposed that it still fused bare tuples such as `0,1` into `01`.

The DROP scorer is unchanged; its demonstration targets now contain the actual
answers instead of dictionary keys. It still expects separate spans when scoring
multi-span predictions. Task versions are incremented where semantics change;
included MATH-500 variants inherit their parent protocol changes as well.

Upstream BBQ answer remapping is deliberately retained and regression-tested.
The audited Swiss mapping incorrectly credited concrete answers when the gold
was unknown in the first two original answer positions. Matching that score
would reproduce a scoring bug. These task changes improve protocol fidelity;
they do not establish that a model reproduces a published score. Comparisons
must identify the model, chat template, task version, selected metric and cohort.

## Install and run

Install the backend appropriate to your environment separately. For Multi-IF
and AlpacaEval task dependencies, plus the MATH scorers:

```bash
pip install -e '.[ifeval,alpaca_eval,math]'
python -m nltk.downloader punkt_tab
lm-eval ls tasks
```

AlpacaEval requires `ALPACA_EVAL_ANNOTATORS_CONFIG` to select the scorer's judge
configuration. Credentials and endpoint settings follow that configuration;
no judge calls are made by task discovery. Set `ALPACA_EVAL_OUTPUT_DIR` to retain
scorer artifacts under your evaluation output directory. Different judge
configurations define different evaluation protocols and must be reported.

Dataset repositories include `swiss-ai/include-base-new-45`,
`swiss-ai/switzerland_qa`, and `swiss-ai/blend-sample`. Dataset access and any
required authentication must be checked in the execution environment. Finding
a task in the index does not prove dataset access or a successful model run.

`acp_bench` remains upstream's tag over Boolean and multiple-choice tasks;
its component results use different extraction filters. It does not emit one
overall score. HumanEval and MBPP retain upstream's explicit code-execution
opt-in. IFBench is provided separately by MLLM-eval-suite's custom-task path.

## Validation

Tests exercise extraction priority, numeric target conversion, region/topic
selection, Multi-IF's first-turn scoring, recursive task/function resolution,
MMLU group aggregation and AlpacaEval's scoring boundary. Upstream filter,
group and task-manager regression tests cover the integration. Validation
does not require a GPU, running generated code, or making paid judge calls.
MBPP extraction tests replace only the external code-evaluation scorer during
module import and check literal extracted strings and response nesting.

Instruction-protocol tests also cover boxed-answer scoring, all MMLU subject
prompts, assistant prefixes, MMLU-Pro demonstration roles and BBQ answer
remapping. The existing upstream math regressions run with the math extra.
