# Swiss task ports implementation plan

> For agentic workers: use the parallel-agent workflow for the independent scorer and suite integration tasks; review their results before publishing.

**Goal:** Publish `yxu/dev` in the existing Swiss AI harness fork, based on upstream v0.4.13, with the nine missing task definitions, then register the requested benchmark set in MLLM-eval-suite.

**Architecture:** Keep upstream engine behavior and add focused task directories and the ordered answer-extraction filter. Use the existing suite registry and launcher. AlpacaEval remains an optional scorer integration with explicit judge configuration; preserve the imported Multi-IF first-turn scope.

**Tech stack:** Python, lm-eval v0.4.13, YAML, pytest, existing Bash/Slurm launchers.

**Spec:** User-approved in this conversation: upstream base, selective Swiss AI ports, branch named `yxu/dev`.

## Global constraints

- Upstream base: `ddd67220430a2470529f25fd5c05a576ca1057a0` (v0.4.13).
- Swiss source: `51d6f4b62bf20e9a29dffa694b163d4f19889927`.
- Preserve existing main branches and checkouts; use isolated clones under `/tmp`.
- Do not claim full multi-turn Multi-IF support: the source evaluates turn 1.
- Keep task prompts, source datasets, language sets, aggregation and extraction semantics reproducible; document deliberate compatibility changes.
- No live judge calls or full benchmark runs are needed for source integration; report runtime checks separately from definitions and offline tests.

## Task 1: Task ports and extraction

Files: `lm_eval/tasks/{global_mmlu/gen_0shot,include/gen_0shot,include_new/gen_0shot,switzerland_qa/gen_0shot,blend,cultural_bench,multi_if}`, the MC2 group YAML under `okapi/truthfulqa_multilingual`, `lm_eval/filters/extraction.py`, `tests/test_swiss_task_ports.py`, `tests/test_ordered_regex.py`, and a source/protocol note in `docs/swiss_task_ports.md`.

- [ ] Write ordered-regex tests with literal expected outputs: preferred explicit answer beats incidental letters; final match of first matching pattern wins; absent matches yield fallback; response nesting is retained.
- [ ] Run `python -m pytest tests/test_ordered_regex.py -q` and observe the absent filter failure.
- [ ] Port only `OrderedRegexFilter` and its registration from the pinned Swiss source, preserving public configuration fields.
- [ ] Copy the requested task subtrees and their referenced helpers; omit unrelated Swiss tasks and alternative prompting variants.
- [ ] Add meaningful fixture checks for include answer-index conversion, language/country subset processing, Multi-IF first-turn interpretation, and imported task-group resolution without downloading datasets.
- [ ] Run the new checks plus upstream filter/task-index tests; run Ruff on changed Python and YAML parsing through the real task loader.
- [ ] Record provenance, optional dependencies, dataset access limitations and Multi-IF scope in documentation.

## Task 2: Portable AlpacaEval adapter

Files: `lm_eval/tasks/alpaca_eval/{alpaca_eval.yaml,metric.py,README.md}`, `tests/test_alpaca_eval_task.py`; optional-dependency changes are coordinated by the controller in `pyproject.toml`.

Interface: YAML calls `alpaca_eval_process(doc, results)` and `alpaca_eval_agg(items)`. Adapter uses the official `alpaca_eval` package and converts its 0–100 length-controlled win rate to the source harness's 0–1 metric. Optional dependencies must be imported only when scoring runs. Source inspection supersedes the misleading Swiss docstring about return units.

- [ ] Test real record preparation, empty completion handling, scorer configuration, and missing/invalid score failures; replace only the external scoring call with a controlled fixture.
- [ ] Port the task, removing dependence on Swiss-only `lm_eval.api.model_resolver` and `rate_limiter` APIs. Require an explicit annotator configuration and use the scorer package's own endpoint/credential configuration.
- [ ] Keep judge identity in task/result metadata or saved scorer artifacts and document that different judges produce different protocols.
- [ ] Run adapter tests and Ruff without making network judge calls.

## Task 3: Suite registration and launch integration

Files: `suite/tasks.toml`, `task_suites/lm-eval/text_requested.txt`, `launchers/lm-eval/eval.sh`, `slurm/lm-eval/eval_job.slurm`, related tests and `docs/lm_eval_task_coverage.md`; controller owns `.gitmodules` and final submodule pin.

- [ ] Register all 31 requested task names with meaningful dashboard metrics, preserving existing entries and defaults. State Multi-IF first-turn scope in documentation.
- [ ] Use existing judge preflight for AlpacaEval and forward its configuration through the launcher environment. Do not hardcode a service credential.
- [ ] Add explicit opt-in code-execution flag forwarding for HumanEval/MBPP; default launches must remain unchanged.
- [ ] Test actual launcher argument propagation with existing fake scheduler patterns and run the suite's full offline tests.

## Task 4: Integrate, review and publish

- [ ] Confirm upstream v0.4.13 remains an ancestor and inspect the complete diff for unintended source changes.
- [ ] Run focused harness tests, installed task-index checks, suite tests and relevant formatting checks.
- [ ] Review all source and integration changes independently; fix material findings.
- [ ] Publish harness `yxu/dev` without force, update suite submodule URL/pin to the published commit, and publish suite `yxu/dev` without force.
- [ ] Open a reviewable suite PR against the existing hardening branch; do not merge.
- [ ] Report exact branch/commit links, validation scope, dependency setup and any tasks whose datasets or live scoring remain unvalidated.
