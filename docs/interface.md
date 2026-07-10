# User Guide

This document details the interface exposed by `lm-eval` and provides details on what flags are available to users.

## Command-line Interface

The `lm-eval` CLI is organized into subcommands:

| Command | Description |
|---------|-------------|
| `lm-eval run` | Run evaluations on language models |
| `lm-eval ls` | List available tasks, groups, subtasks, or tags |
| `lm-eval validate` | Validate task configurations |

Run the library via the `lm-eval` entrypoint or `python -m lm_eval`.

Use `-h` or `--help` to see available options:

```bash
lm-eval -h              # Show all subcommands
lm-eval run -h          # Show options for run command
lm-eval ls -h           # Show options for list command
```

> **Legacy Compatibility**: The original single-command interface still works. Running `lm-eval --model hf --tasks hellaswag` automatically inserts the `run` subcommand.

---

## Quick Start

```bash
# List available tasks
lm-eval ls tasks

# Basic evaluation
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag

# With few-shot examples
lm-eval run --model hf --model_args pretrained=gpt2 --tasks arc_easy --num_fewshot 5

# Save results and model outputs
lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag --output_path ./results/ --log_samples

# Use a config file
lm-eval run --config eval_config.yaml
```

---

## `lm-eval run`

Run evaluations on language models.

```bash
lm-eval run --model <model> --tasks <task> [options]
```

### Quick Examples

```bash
# Basic evaluation with HuggingFace model
lm-eval run --model hf --model_args pretrained=gpt2 dtype=float32 --tasks hellaswag

# Multiple tasks with few-shot examples
lm-eval run --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks arc_easy arc_challenge --num_fewshot 5

# Custom generation parameters
lm-eval run --model hf --model_args pretrained=gpt2 --tasks lambada --gen_kwargs temperature=0.8 top_p=0.95

# Use a YAML configuration file
lm-eval run --config my_config.yaml --tasks mmlu
```

### Model and Tasks

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-M` | Model type/provider name (default: `hf`). See [supported models](https://github.com/EleutherAI/lm-evaluation-harness#model-apis-and-inference-servers). |
| `--model_args` | `-a` | Model constructor arguments as `key=val key2=val2` or `key=val,key2=val2`. For HuggingFace models, see [`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py) for available arguments. |
| `--tasks` | `-t` | Space or comma-separated list of task names or groups. Use `lm-eval ls tasks` to see available tasks. |
| `--apply_chat_template` | | Apply chat template to prompts. Use without argument for default template, or specify template name. |
| `--limit` | `-L` | Limit examples per task. Integer for count, float (0.0-1.0) for percentage. **For testing only.** |
| `--use_cache` | `-c` | Path prefix for SQLite cache of model responses (e.g., `/path/to/cache_`). |

### Evaluation Settings

| Argument | Short | Description |
|----------|-------|-------------|
| `--num_fewshot` | `-f` | Number of few-shot examples in context. |
| `--batch_size` | `-b` | Batch size: integer, `auto`, or `auto:N` to auto-tune N times (default: 1). |
| `--max_batch_size` | | Maximum batch size when using `--batch_size auto`. |
| `--device` | | Device to use: `cuda`, `cuda:0`, `cpu`, `mps` (default: `cuda`). |
| `--gen_kwargs` | | Generation arguments as `key=val key2=val2`. Values parsed with `ast.literal_eval`. Example: `temperature=0.8 'stop=["\n\n"]'` |

### Data and Output

| Argument | Short | Description |
|----------|-------|-------------|
| `--output_path` | `-o` | Output directory or JSON file for results. Required with `--log_samples`. |
| `--log_samples` | `-s` | Save all model inputs/outputs for post-hoc analysis. |
| `--samples` | `-E` | JSON mapping task names to sample indices, e.g., `'{"task1": [0,1,2]}'`. Incompatible with `--limit`. |
| `--log_length_metrics` | | Aggregate per-response response/thinking length into per-task metrics in the results and W&B. See [Generation length & thinking-format metrics](#generation-length--thinking-format-metrics). |

### Caching and Performance

| Argument | Description |
|----------|-------------|
| `--cache_requests` | Cache preprocessed prompts: `true`, `refresh`, or `delete`. Cached files stored in `lm_eval/cache/.cache` or path set by `LM_HARNESS_CACHE_PATH` env var. |
| `--check_integrity` | Run task test suite validation before evaluation. |

### Prompt Formatting

| Argument | Description |
|----------|-------------|
| `--system_instruction` | Custom system instruction prepended to prompts. |
| `--fewshot_as_multiturn` | Format few-shot examples as multi-turn conversation. Auto-enabled with `--apply_chat_template`. Set to `false` to disable. |

### Task Management

| Argument | Description |
|----------|-------------|
| `--include_path` | Additional directory containing external task YAML files. |

### Logging and Tracking

| Argument | Short | Description |
|----------|-------|-------------|
| `--verbosity` | `-v` | **(Deprecated)** Use `LMEVAL_LOG_LEVEL` env var instead. |
| `--write_out` | `-w` | Print prompts for first few documents (for debugging). |
| `--show_config` | | Display full task configuration after evaluation. |
| `--wandb_args` | | Weights & Biases arguments as `key=val`. E.g., `project=my-project name=run-1`. |
| `--wandb_config_args` | | Additional W&B config arguments. |
| `--hf_hub_log_args` | | HuggingFace Hub logging arguments. See [HF Hub Logging](#huggingface-hub-logging). |

### Advanced Options

| Argument | Short | Description |
|----------|-------|-------------|
| `--predict_only` | `-x` | Save predictions only, skip metric computation. Implies `--log_samples`. |
| `--seed` | | Random seeds as single integer or comma-separated list for `python,numpy,torch,fewshot`. Default: `0,1234,1234,1234`. Use `None` to skip. Example: `--seed 42` or `--seed 0,None,8,52`. |
| `--trust_remote_code` | | Allow executing remote code from HuggingFace Hub. |
| `--confirm_run_unsafe_code` | | Confirm understanding of risks for tasks executing arbitrary Python. |
| `--metadata` | | JSON string passed to TaskConfig. Required for some tasks like RULER. Example: `--metadata '{"max_seq_length": 4096}'`. |

### Configuration File

| Argument | Short | Description |
|----------|-------|-------------|
| `--config` | `-C` | Path to YAML configuration file. CLI arguments override config file values. See [Configuration Files](config_files.md). |

### HuggingFace Hub Logging

The `--hf_hub_log_args` argument accepts these keys:

| Key | Description |
|-----|-------------|
| `hub_results_org` | Organization name on HF Hub. Defaults to token owner. |
| `details_repo_name` | Repository name for detailed results. |
| `results_repo_name` | Repository name for aggregated results. |
| `push_results_to_hub` | `True`/`False` - push results to Hub. |
| `push_samples_to_hub` | `True`/`False` - push samples to Hub. Requires `--log_samples`. |
| `public_repo` | `True`/`False` - make repository public. |
| `leaderboard_url` | URL to associated leaderboard. |
| `point_of_contact` | Contact email for results dataset. |
| `gated` | `True`/`False` - gate the details dataset. |

---

### Generation length & thinking-format metrics

Options that control the per-response length / thinking-format metrics and the
reasoning-trace strip for thinking models. See **[Thinking / reasoning
evals](./thinking_evals.md)** for what these metrics mean, what is logged where,
malformed/non-thinking handling, and multi-turn behavior.

**Overview — four independent switches:**

- **The model thinks** when its **chat template** says so. `enable_thinking` is purely a
  template argument: `vllm` forwards it (default `true`), `hf` forwards it only when you
  set it explicitly, and `sglang` never does — otherwise the template's own default
  applies. It does **not** gate the strip, token detection, or metric tracking.
- **The reasoning tokens are auto-detected** from the chat template only when you opt in
  with `autodetect_think_tokens=true` (default `false`, so nothing is scanned).
- **The strip runs** whenever a reasoning **close** token is known — set explicitly via
  `think_end_token`, or auto-detected. Everything up to and including the close is
  dropped before the answer is scored. By default no close is known, so **nothing is
  stripped**.
- **Thinking length & correctness** (`thinking_length_*`, `thinking_format_*`) are
  tracked iff a close token is known — or whatever `track_thinking_metrics` forces.
  `thinking_format_*` reaches results/W&B on its own; `thinking_length_*` needs
  `--log_length_metrics`.
- **Response length** (`response_length_*`) is recorded for **every** `generate_until`
  response, thinking or not, and reaches results/W&B with `--log_length_metrics`.

Per-response values are always written to `length_info` in the sample jsonl under
`--log_samples`, independently of the aggregation flags.

Added by this feature:

| Option | Kind | Description |
|--------|------|-------------|
| `--log_length_metrics` | CLI | Aggregate `response_length_*` / `thinking_length_*` into per-task metrics (results.json, table, W&B). Off by default; `thinking_format_*` is aggregated regardless. |
| `think_start_token` | `--model_args` | Force the reasoning **open** token (str); otherwise auto-detected from the chat template only when `autodetect_think_tokens=true`. Used for `thinking_format_has_open`; without it the format metric degrades to close-only. |
| `autodetect_think_tokens` | `--model_args` | `true`/`false` (**default `false`**) — opt in to auto-detecting the reasoning open/close from the chat template. Left off, the template is never scanned: with no explicit `think_end_token` there is then **no strip and no thinking metrics**, and the fail-loud guard cannot fire. |
| `track_thinking_metrics` | `--model_args` | `true`/`false` — force the `thinking_format_*` / `thinking_length_*` metrics on or off. Default (unset) derives them: on iff a close token is known. Requires a close token even when forced on (a missing **open** is fine — the metric degrades to close-only). `response_length_*` is unaffected. |

Pre-existing options that also affect these metrics:

| Option | Kind | Description |
|--------|------|-------------|
| `--log_samples` | CLI | Write the per-response `length_info` (all length + thinking-format fields) into the sample jsonl. |
| `--wandb_args` | CLI (value) | Native W&B upload of the aggregated metrics, e.g. `--wandb_args project=p name=run1`. |
| `--apply_chat_template` | CLI | Required for the chat/thinking template to render (the reasoning tokens live in it). |
| `enable_thinking` | `--model_args` | `true`/`false` — a **chat-template argument only** (`hf`/`vllm`; `sglang` has none). Forwarded by `vllm` (default `true`) and by `hf` only when set explicitly; otherwise the template's default decides. It does **not** gate the strip, token detection, or metric tracking. (It changes the rendered prompt, so on a prefill template it does affect the `has_open` value.) |
| `think_end_token` | `--model_args` | Force the reasoning **close** token (str; also int token id on `hf`); drives the strip. Honoured with or without `autodetect_think_tokens`. |

Useful combinations:

```bash
# Nothing (the default): no detection, no strip, no thinking metrics
--model_args pretrained=M

# Opt in: detect the tokens, strip the trace, record the thinking metrics
--model_args pretrained=M,autodetect_think_tokens=true

# Name the close yourself (no template scan; `has_open` is not tracked)
--model_args pretrained=M,think_end_token='</think>'

# Strip, but drop the thinking metrics (and their per-response cost)
--model_args pretrained=M,autodetect_think_tokens=true,track_thinking_metrics=false
```

---

## `lm-eval ls`

List available tasks, groups, subtasks, or tags.

```bash
lm-eval ls [tasks|groups|subtasks|tags] [--include_path DIR]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `tasks` | List all available tasks (groups, subtasks, and tags). |
| `groups` | List only task groups (e.g., `mmlu`, `glue`, `superglue`). |
| `subtasks` | List only individual subtasks (e.g., `mmlu_anatomy`, `hellaswag`). |
| `tags` | List task tags (e.g., `reasoning`, `knowledge`). |
| `--include_path` | Additional directory for external task definitions. |

### Task Organization

- **Groups**: Collections of related tasks with aggregated metrics across subtasks (e.g., `mmlu` contains 57 subtasks)
- **Subtasks**: Individual evaluation tasks (e.g., `mmlu_anatomy`, `hellaswag`)
- **Tags**: Categories for filtering tasks without aggregated metrics (e.g., `reasoning`, `language`)

### Examples

```bash
# List all tasks
lm-eval ls tasks

# List only task groups
lm-eval ls groups

# Include external tasks
lm-eval ls tasks --include_path /path/to/external/tasks
```

---

## `lm-eval validate`

Validate task configurations before running evaluations.

```bash
lm-eval validate --tasks <task1,task2> [--include_path DIR]
```

### Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--tasks` | `-t` | **(Required)** Comma-separated list of task names to validate. |
| `--include_path` | | Additional directory for external task definitions. |

### Validation Checks

The validate command performs:

- **Task existence**: Verifies all specified tasks are available
- **Configuration syntax**: Checks YAML/JSON configuration files
- **Dataset access**: Validates dataset paths and configurations
- **Required fields**: Ensures all mandatory task parameters are present
- **Metric definitions**: Verifies metric functions and aggregation methods
- **Filter pipelines**: Validates filter chains and their parameters
- **Template rendering**: Tests prompt templates with sample data

### Examples

```bash
# Validate a single task
lm-eval validate --tasks hellaswag

# Validate multiple tasks
lm-eval validate --tasks arc_easy,arc_challenge,hellaswag

# Validate a task group
lm-eval validate --tasks mmlu

# Validate external tasks
lm-eval validate --tasks my_custom_task --include_path ./custom_tasks
```

---

## Python API

For programmatic usage, see the [Python API Guide](python-api.md).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LMEVAL_LOG_LEVEL` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `LM_HARNESS_CACHE_PATH` | Path for cached requests (default: `lm_eval/cache/.cache`). |
| `HF_TOKEN` | HuggingFace Hub token for private datasets/models. |
| `TOKENIZERS_PARALLELISM` | Set to `false` to avoid tokenizer warnings (auto-set by CLI). |
