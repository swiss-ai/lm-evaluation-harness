# POLAR-Bench: non-reasoning integration

This task evaluates selective disclosure on the released 7,852-instance dataset.
Protocols 1–4 replay fixed attacks; protocol 5 uses a fixed external attacker B.
A is the evaluated model. B is not a judge: scoring is deterministic.

## Minimal integration

The baseline is Swiss-AI harness commit
`51d6f4b62bf20e9a29dffa694b163d4f19889927`.
The only modified existing file is `lm_eval/tasks/_task_index.json`, registering
`polar_bench`. All other changes add POLAR-specific task, launcher, documentation,
and offline verification files. The evaluator, vLLM backend, shared dependency
configuration and `evals-post-train` are unchanged.

The task uses the original `multi_turn_generate` hooks. This version supports
**non-reasoning instruction models only**, with `POLAR_THINKING_MODE=final_only`.
Normal visible answers need no thinking closing marker. The launcher disables
thinking, token autodetection and thinking metrics. This does not turn a
reasoning-only model into a non-reasoning model. Reasoning support is deferred.
Unexpected inline `<think>...</think>` text is stripped defensively; unfinished
inline reasoning yields an empty answer. Prefilled or natively stripped reasoning
cannot be verified through this original interface and is outside this version's
scope.

## Protocol and scoring

- A: **8,192 generated tokens per round**; B: **32,768** by default. Override A with
  `--gen_kwargs max_gen_toks=16384` if needed. Set
  `POLAR_ATTACKER_MAX_TOKENS=65536` to change B's per-request budget; its resolved
  value is recorded in task metadata. For reasoning B services, this budget can
  include reasoning as well as visible output; it is not a visible-answer quota.
- Visible A answers enter subsequent A/B history and the scorer.
- B's `<STOP>` ends the dialogue before the next A call. A refusal or privacy leak
  alone does not stop it. Empty or truncated B attacks fail explicitly.
- The turn budget follows the reference: sample-specific turns when present,
  otherwise 6, capped by the fixed script length when applicable.
- The harness defensive step cap raises an error instead of scoring a partial run.
- `privacy`, `utility`, and `overall` are 0–1 scores, higher is better. The scorer
  considers all A turns and excludes B turns. Overall is the arithmetic mean of
  privacy and utility. Also report `final_answer_rate` and `zero_turn_rate`:
  an empty dialogue can still receive privacy 1, utility 0, overall 0.5.

These are not drop-in reproductions of old **1,200-token** results. The original
prompt builders, target extraction, seeds and scorer are vendored in `utils.py`,
with the source snapshot SHA256 and MIT notice. That snapshot is from the user's
working evaluator, including unpublished changes, not a released commit.

No A termination-reason or truncation-rate metric is claimed: the original
multi-turn interface does not pass that metadata to the task. Check output logs
and context limits during cluster validation. Avoid response caching (`--use_cache`)
until its compatibility with the dynamic dialogue has been validated.

## Cluster smoke run

Use this checkout inside an allocated GPU job with a compatible vLLM environment.
Install with `pip install -e '.[vllm]' openai`; when vLLM is already supplied by the
cluster container, install `pip install -e . openai` in an environment that exposes
those container packages. This script does not submit SLURM jobs or start B.
A separate wrapper may reuse `evals-post-train`'s container and environment builder
without modifying that repository.

```bash
export POLAR_ATTACKER_BASE_URL='https://YOUR_B_SERVICE/v1'
# Set POLAR_ATTACKER_API_KEY securely if the endpoint requires it.
export POLAR_ATTACKER_MODEL='meta-llama/Llama-3.3-70B-Instruct'
export POLAR_ATTACKER_MAX_TOKENS=32768
export POLAR_DATASET_PATH='/path/to/polar-smoke-50.json'
export POLAR_OUTPUT_PATH='results/polar-nonreasoning-smoke'
export POLAR_VLLM_ARGS='tensor_parallel_size=4,data_parallel_size=1'
bash scripts/run_polar_vllm.sh /path/to/non-reasoning-A-model
```

Use a subset covering all five attack protocols and ten domains rather than
the first 50 dataset rows. It is an integration test, not a representative
leaderboard estimate. B must be reachable from the evaluation node, with its
served model ID matching `POLAR_ATTACKER_MODEL`. A must be non-reasoning. A reasoning B service may be tested when it returns
its final answer separately in `message.content`; the client does not forward
separate `reasoning`/`reasoning_content` fields. A larger budget does not guarantee
completion. The B timeout defaults to 600 seconds per attempt; override with
`POLAR_ATTACKER_TIMEOUT`. A slow endpoint may still time out. The service must
support the requested output limit, and its context must fit both input and output.
The default 32,768-token B budget differs from the original 500-token protocol;
record the B model and budget when comparing results.

Run one harness process; vLLM tensor parallelism is supported. Do not wrap this
launcher in `accelerate launch`. The context must fit the document, policy,
growing conversation and output budget. The launcher records A's name for seed
derivation; `POLAR_SEED` defaults to 42, `POLAR_MAX_ROUNDS` to 6.

Without `POLAR_DATASET_PATH`, the task downloads the HF JSON at revision
`2fe6a18ac50ab86c64b2e5935031901ca35a2d18`. The exact file hash and row count are
logged. Invalid rows fail rather than silently changing the population.

## Offline verification

```bash
pip install -e . openai pytest
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest \
  tests/test_polar_bench.py tests/test_multiturn.py \
  tests/test_mt_bench.py tests/test_thinking_format_aggregation.py -q

```

The historical reference comparison (B budget 500) is recorded in `validation_report.json`:
all 7,852 rows were checked for targets, eligibility and system prompts, and
50 representative dialogues (160 A turns) were compared using fake A/B replies.
The one-off comparison script is not included in this integration.
The regression tests above require no inference, credentials or GPUs.

## Real-model integration validation

A Clariden run on 2026-10-01 completed all 250 coverage samples with
Qwen/Qwen2.5-7B-Instruct as A and locally served
meta-llama/Llama-3.3-70B-Instruct as B. The subset covers every combination of
10 domains, 5 protocols, and 5 privacy levels. It produced 771 A turns in
378 seconds of evaluation time, excluding environment preparation and B startup.
The run used harness commit `1002f1ed0bdb1611613d2f29cf89d293a22d977b`,
A/B generation budgets of 8,192/32,768 tokens, and A/B context limits of
32,768/65,536 tokens.

| Metric | Value |
| --- | ---: |
| privacy | 0.3315714286 |
| utility | 0.8632349206 |
| overall | 0.5974031746 |
| final_answer_rate | 1.0 |
| zero_turn_rate | 0.0 |

[W&B run](https://wandb.ai/apertus/polar-integration-test/runs/polar-qwen25-7b-llama33-70b-cover250-20261001-205856-9e6a-001)
(access may require project permission). Online upload was confirmed after a
separate upload-only retry; the original Slurm job retains its failed upload status.
This validates the integration on a coverage subset, not the full 7,852-instance
benchmark or a representative leaderboard estimate. The subset and two-node
Slurm wrapper are not included in this task integration.

The wrapper reused `evals-post-train` commit
`b1b192ecab12f3310ad2da673c7a195a7e3c9fb7` without modifying its source.
The launch helper in this PR runs within an existing GPU allocation and requires
an already reachable B service; it does not submit Slurm jobs, deploy B, or upload
to W&B. Shared dashboard inclusion remains a separate maintainer decision.
