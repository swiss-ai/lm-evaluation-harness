---
name: lm-eval-clariden-testing
description: Run, validate, and debug lm-evaluation-harness tests on the CSCS Clariden/Alps Slurm cluster, including ordinary pytest, Hugging Face GPU tests, mocked Megatron adapter tests, four-rank distributed Megatron tests, and real checkpoint smoke evaluations. Use whenever a user asks to test lm-eval changes, run its Megatron tests, reproduce a failure in a CSCS container, select an EDF, or diagnose Slurm, dependency, cache, NCCL, and rank-layout problems in this repository.
compatibility: CSCS Clariden/Alps, Slurm, CSCS container environments, uv, pytest, and a user-selected compatible EDF.
---

# Test lm-evaluation-harness on CSCS Clariden

Run compute-heavy and GPU-dependent validation on compute nodes. Use the repository's current configuration rather than carrying commands from another branch.

## Read the project before choosing tools

Inspect these files before running or formatting anything:

- `pyproject.toml`
- `.pre-commit-config.yaml`
- the target test and its fixtures
- `lm_eval/models/megatron_lm.py` for Megatron adapter work
- `tests/models/test_megatron_lm.py`
- `tests/models/test_megatron_lm_distributed.py`

This repository currently uses Ruff through pre-commit. Do not assume Black. Use the Ruff version pinned by `.pre-commit-config.yaml`, and respect `[tool.ruff]` in `pyproject.toml`.

## Choose the smallest test tier

1. **Pure harness logic:** run focused pytest tests in one process. GPU allocation is unnecessary unless imports require the container's Torch stack.
2. **Mocked Megatron adapter:** run `tests/models/test_megatron_lm.py` in one process with Torch available. It tests adapter logic without loading a checkpoint.
3. **Distributed adapter:** run `tests/models/test_megatron_lm_distributed.py` with four Slurm tasks and one GPU per task. Tests that require another world size skip themselves.
4. **Real adapter smoke test:** set a real `MEGATRON_PATH`, checkpoint, tokenizer, and model arguments. Start with the smallest topology that represents the bug.
5. **Full evaluation:** run only after focused and distributed tests pass.

Do not use a full benchmark as the first debugging step.

## Select account, partition, and container explicitly

There is no skill-wide default EDF. Use the environment selected for the current experiment or ask the user.

```bash
sinfo -a -o "%P %a"
scontrol show reservation

export SLURM_ACCOUNT=<project-account>
export SLURM_PARTITION=<an-up-partition>
export SLURM_ENVIRONMENT=<edf-path-or-site-environment-name>
```

Rules:

- Select only a partition that is currently `up`.
- Use `--reservation` only after verifying that it is active and covers the account, resources, and requested time. Otherwise omit it.
- If `SLURM_ENVIRONMENT` is an absolute path, verify it with `test -r "$SLURM_ENVIRONMENT"`.
- Put `--environment="$SLURM_ENVIRONMENT"` on `srun`.
- Use `--network=disable_rdzv_get` and `--mpi=pmix` for containerized distributed jobs unless a tested case needs something else.
- Verify the EDF can import the required Torch and Megatron versions before interpreting test failures.

## Keep all job writes off home

```bash
export REPO=/users/anowak/developer/lm-evaluation-harness
export SCRATCH_ROOT=${SCRATCH:-/iopsstor/scratch/cscs/$USER}
export SESSION="$SCRATCH_ROOT/tmp/lm-eval-tests-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SESSION"/{logs,tmp}

if test -d /ritom/scratch/cscs/$USER; then
  export CACHE_ROOT=/ritom/scratch/cscs/$USER/cache
else
  export CACHE_ROOT="$SESSION/cache"
fi
mkdir -p "$CACHE_ROOT"/{uv,xdg,torch,huggingface}

export UV_CACHE_DIR="$CACHE_ROOT/uv"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export UV_PROJECT_ENVIRONMENT="$SESSION/.venv"
```

Install or sync only inside the selected container. A clean scratch environment can reuse the container's Torch installation:

```bash
uv venv --system-site-packages "$UV_PROJECT_ENVIRONMENT"
uv sync --frozen --group dev --no-install-package torch
```

If the repository environment is already prepared for that exact EDF, use `uv run --no-sync`. Do not mutate a `$HOME` `.venv` from a job.

Set unique node-local compiler caches before Python or Torch imports:

```bash
rank=${SLURM_PROCID:-0}
runtime_cache_root="/tmp/lm-eval-${SLURM_JOB_ID:-manual}-$rank"
export TORCH_EXTENSIONS_DIR="$runtime_cache_root/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$runtime_cache_root/torchinductor"
export TRITON_CACHE_DIR="$runtime_cache_root/triton"
export FLASHINFER_WORKSPACE_BASE="$runtime_cache_root/flashinfer"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR"   "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
```

Mount the repository and session path plus `$HOME`, `/capstor`, `/iopsstor`, and `/ritom` when available. Reading the checkout from `$HOME` is acceptable; logs, environments, caches, downloads, and generated artifacts must go elsewhere.

## Run focused tests

For tests that only need one process and one GPU:

```bash
srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=72 \
  --gres=gpu:1 \
  --time=00:30:00 \
  --exclusive \
  --network=disable_rdzv_get \
  --mpi=pmix \
  --environment="$SLURM_ENVIRONMENT" \
  --container-mounts="$REPO:$REPO,$SESSION:$SESSION,$HOME:$HOME,/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom" \
  -u bash -lc "cd '$REPO' && uv run --no-sync pytest -q tests/models/test_megatron_lm.py" \
  2>&1 | tee "$SESSION/logs/megatron-unit.log"
```

A useful focused regression set for evaluator and Megatron changes is:

```bash
uv run --no-sync pytest -q \
  tests/models/test_lm.py \
  tests/models/test_megatron_lm.py \
  tests/test_evaluator_utils.py \
  tests/test_multiturn.py \
  tests/test_wandb_model_metrics.py \
  tests/test_cli_model_args.py
```

Add `tests/test_evaluator.py` when the environment has the Hugging Face dependencies, Torch, model access, and enough cache space. Distinguish missing dependencies or model access from assertion failures.

## Run the four-rank distributed adapter tests

These tests read `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`. Use one Slurm task per GPU. Do not wrap the command in `torchrun`.

Create a launch script under `$SESSION/tmp` so rank setup is visible and reusable:

```bash
#!/usr/bin/env bash
set -euo pipefail

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

rank=$SLURM_PROCID
runtime_cache_root="/tmp/lm-eval-${SLURM_JOB_ID}-$rank"
export TORCH_EXTENSIONS_DIR="$runtime_cache_root/torch-extensions"
export TORCHINDUCTOR_CACHE_DIR="$runtime_cache_root/torchinductor"
export TRITON_CACHE_DIR="$runtime_cache_root/triton"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd /users/anowak/developer/lm-evaluation-harness
uv run --no-sync pytest -q tests/models/test_megatron_lm_distributed.py
```

Launch it on one four-GPU node:

```bash
srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  --nodes=1 \
  --ntasks=4 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  --gpus-per-task=1 \
  --cpus-per-task=72 \
  --time=00:30:00 \
  --exclusive \
  --network=disable_rdzv_get \
  --mpi=pmix \
  -l \
  --environment="$SLURM_ENVIRONMENT" \
  --container-mounts="$REPO:$REPO,$SESSION:$SESSION,$HOME:$HOME,/capstor:/capstor,/iopsstor:/iopsstor,/ritom:/ritom" \
  -u bash "$SESSION/tmp/run-distributed-tests.sh" \
  2>&1 | tee "$SESSION/logs/megatron-distributed.log"
```

For repeated commands, reuse one allocation. If the global `interactive-srun` skill is available, override its account, partition, environment, GPU count, task count, mounts, time, and allocation-log path explicitly. Record the job ID and stop the allocation when finished.

## Run a real Megatron checkpoint smoke test

Before launching:

1. Verify `MEGATRON_PATH` points to the intended checkout inside the container.
2. Verify the checkpoint and tokenizer paths are mounted and readable.
3. Match `devices`, TP, PP, and EP to the allocated world size.
4. Decide whether checkpoint metadata is authoritative. Keep `use_checkpoint_args=true` by default; set it false only when supplying the complete architecture explicitly.
5. Start with a small task and `--limit`.
6. Keep Hugging Face data and request caches under `CACHE_ROOT` or `SESSION`.

Prefer Slurm's one-task-per-GPU layout for real distributed evaluations. Use `SLURM_PROCID` as global rank, `SLURM_LOCALID` as local rank, and `SLURM_NTASKS` as world size. Do not add `torchrun` when Slurm already creates every rank.

## Format and lint correctly

Read `.pre-commit-config.yaml` first. Run the repository's pinned Ruff hooks on changed Python files:

```bash
uv run --no-sync pre-commit run ruff-check --files <changed.py> ...
uv run --no-sync pre-commit run ruff-format --files <changed.py> ...
```

Alternatively, invoke the exact Ruff version pinned by pre-commit. Do not substitute another globally installed Ruff version, and do not run Black unless the repository configuration changes to require it.

Run:

```bash
git diff --check
```

If it reports whitespace inherited unchanged from the branch being merged, report that separately from new whitespace introduced by the current edits.

## Report results precisely

Record and report:

- repository commit
- account and partition
- EDF or environment name
- Slurm job ID
- node/task/GPU topology
- exact pytest or evaluation command
- pass/fail/skip counts
- log path
- whether failures are assertions, dependency/import failures, model-access failures, NCCL/rendezvous failures, preemption, or timeouts

Preserve logs under `$SESSION`. Stop or cancel only the allocation recorded for this test.
