#!/usr/bin/env python3
"""Evaluate a training checkpoint with the SGLang backend.

This script
-----------
1. Detects the *format* of an arbitrary checkpoint directory
   (HuggingFace, Megatron‑LM core/torch‑dist, or unknown).
2. Converts non‑HF checkpoints **in‑place** to HuggingFace format using
   Megatron‑LM's official conversion utilities.  The converted model is
   written next to the original checkpoint under a sibling directory
   `<original_name>_hf` (or a custom `--hf-out` path).
3. Launches the *lm‑eval‑harness* with the **sglang** backend to run a
   benchmark suite of your choice (defaults to *gsm8k_cot*).

The script is intended to be lightweight glue; all heavy lifting is
performed by existing Megatron‑LM and SGLang tools.

Example
-------
```bash
python evaluate_checkpoint.py \
  /path/to/iter_123000 \
  --model-size 8b \
  --tasks arc_easy,hellaswag \
  --dp-size 2 --tp-size 4
```
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Sequence


# DEFAULT_TASKS = "winogrande,lambada_openai,piqa,social_iqa,openbookqa,arc_easy,commonsense_qa,triviaqa,mmlu_continuation,gsm8k,global_mmlu_ar,global_mmlu_bn,global_mmlu_de,global_mmlu_en,global_mmlu_es,global_mmlu_fr,global_mmlu_hi,global_mmlu_id,global_mmlu_it,global_mmlu_ja,global_mmlu_ko,global_mmlu_pt,global_mmlu_sw,global_mmlu_yo,global_mmlu_zh,wikitext,lambada,hellaswag,longbench,ruler"
DEFAULT_TASKS = "longbench,ruler"
# DEFAULT_TASKS = "longbench,ruler,hellaswag,piqa,winogrande,arc_easy,arc_challenge,lambada_openai,triviaqa,gsm8k,mathqa"


###############################################################################
# Generic helpers
###############################################################################

def _run(cmd: Sequence[str] | str, **kwargs) -> None:
    """Wrapper around *subprocess.run* with nicer printing and error handling."""
    if isinstance(cmd, (list, tuple)):
        cmd_str = " ".join(cmd)
    else:
        cmd_str = cmd
    print(f"[cmd] {cmd_str}", flush=True)
    completed = subprocess.run(cmd_str, shell=True, **kwargs)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd_str}")

###############################################################################
# Checkpoint detection & conversion
###############################################################################

CheckpointType = Literal["hf", "megatron", "unknown"]


def detect_ckpt_type(path: Path) -> CheckpointType:
    """Very small heuristic classifier for checkpoint layout."""
    # megatron: path should contain folder iter_xxx
    if any(path.glob("iter_*")):
        return "megatron"
        
    if list(path.glob("*.safetensors")):
        return "hf"
    else:
        return "unknown"

def convert_megatron_to_hf(
    ckpt_dir: Path,
    megatron_lm_dir: Path,
    hf_out: Path,
    pipeline_mp_size: int | None = None,
    iter: int = -1,
    overwrite_max_seq_length: int = -1,
) -> None:
    """Convert *ckpt_dir* (Megatron‑LM) → HuggingFace under *hf_out*.

    This calls **torchdist_2_torch.py** followed by **convert.py** exactly the
    same way the SwissAI helper scripts do, but with fewer special‑case
    assumptions so it works for arbitrary checkpoints.
    """

    ckpt_dir = ckpt_dir.resolve()
    hf_out = hf_out.resolve()
    hf_out.mkdir(parents=True, exist_ok=True)
    iter_str = f"_iter{iter}" if iter != -1 else ""

    torch_tmp = Path(str(hf_out) + f"_torch{iter_str}")
    torch_tmp.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    current_pythonpath = env.get('PYTHONPATH', '')
    env["PYTHONPATH"] = f"{megatron_lm_dir}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(megatron_lm_dir)

    torchdist_script = megatron_lm_dir / "scripts/conversion/torchdist_2_torch.py"
    convert_py = megatron_lm_dir / "tools/checkpoint/convert.py"

    # ---------------------------------------------------------------------
    # Step 1: Megatron torch‑dist ⇒ single‑node torch
    # ---------------------------------------------------------------------
    # TODO: support 70B model
    torch_cmd = [
        "CUDA_DEVICE_MAX_CONNECTIONS=1",
        "torchrun",
        "--nproc-per-node=1",  # you may expose more GPUs if you wish
        str(torchdist_script),
        "--bf16",
        "--attention-backend",
        "fused",
        # "--fix-old-xielu",
        "--load",
        str(ckpt_dir),
        "--ckpt-convert-save",
        str(torch_tmp),
    ]
    if iter != -1:
        torch_cmd += ["--ckpt-step", str(iter)]
    if pipeline_mp_size:
        torch_cmd += ["--pipeline-model-parallel-size", str(pipeline_mp_size)]

    _run(torch_cmd, env=env)

    # ---------------------------------------------------------------------
    # Step 2: torch ⇒ HuggingFace
    # ---------------------------------------------------------------------
    core_dir = torch_tmp / "torch"
    hf_cmd = [
        # "NVTE_DEBUG=1",
        # "NVTE_DEBUG_LEVEL=2",
        sys.executable,
        str(convert_py),
        "--model-type",
        "GPT",
        "--loader",
        "core",
        "--saver",
        "swissai_hf",
        # "--test-logits", # no logit test for long context models, OOMs
        "--load-dir",
        str(core_dir),
        "--save-dir",
        str(hf_out),
        "--hf-tokenizer", 
        "alehc/swissai-tokenizer",
    ]
    _run(hf_cmd, env=env)

    # Clean up intermediate torch checkpoint to save space
    shutil.rmtree(torch_tmp, ignore_errors=True)
    
    if overwrite_max_seq_length != -1:
        hf_out_config = json.load(open(hf_out / "config.json", "r"))
        # WARNING: hardcoded value to detect base model
        if hf_out_config['max_position_embeddings'] < overwrite_max_seq_length:
            hf_out_config['max_position_embeddings'] = overwrite_max_seq_length
            json.dump(hf_out_config, open(hf_out / "config.json", "w"))
            print(f"[info] overwritten max_position_embeddings to {overwrite_max_seq_length} for sglang eval")

###############################################################################
# Benchmarking with lm‑eval‑harness + SGLang backend
###############################################################################

def run_lm_eval(
    hf_dir: Path,
    tasks: Sequence[str],
    dp_size: int,
    tp_size: int,
    context_length: int,
    dtype: str = "auto",
    batch_size: str | int = "auto",
    extra_lm_eval_args: Sequence[str] | None = None,
) -> None:
    # if isinstance(batch_size, str) and batch_size != "auto":
    #     raise ValueError("batch_size string value must be 'auto'")
        
    if 'longbench' in tasks:
        context_length = max(context_length, 131072) # longbench need > 64k context length

    model_args = (
        f"pretrained={hf_dir},mem_fraction_static=0.7,trust_remote_code=True,tokenizer_path=alehc/swissai-tokenizer,dp_size={dp_size},tp_size={tp_size},dtype={dtype},context_length={context_length}"
    )

    cmd = [
        "lm_eval",
        "--model",
        "sglang",
        "--model_args",
        model_args,
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(batch_size),
    ]
    if extra_lm_eval_args:
        cmd.extend(extra_lm_eval_args)

    _run(cmd)

###############################################################################
# Main CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect, convert, and benchmark checkpoints with SGLang backend.")
    p.add_argument("checkpoint", type=Path, help="Path to checkpoint directory (any format)")
    p.add_argument("--megatron-lm-dir", type=Path, help="Path to Megatron‑LM source", default=os.environ.get("MEGATRON_LM_DIR", ""))
    p.add_argument("--hf-out", type=Path, help="Where to write (or reuse) the converted HuggingFace checkpoint")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma‑separated lm‑eval‑harness tasks")
    p.add_argument("--dp-size", type=int, default=2, help="Data parallel size for SGLang server")
    p.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size for SGLang server")
    p.add_argument("--dtype", default="auto", help="Model dtype to request from SGLang (auto|bf16|fp16|fp32)")
    p.add_argument("--batch-size", default="auto", help="Batch size for lm‑eval‑harness (int or 'auto')")
    p.add_argument("--ruler-lengths", type=str, default="4096,8192,16384", help="Comma-separated list of sequence lengths for ruler task")
    p.add_argument("--context-length", type=int, default=32768, help="Context length for SGLang evaluation")
    p.add_argument("--skip-eval", action="store_true", help="Only convert, do not run lm‑eval‑harness")
    p.add_argument("--wandb-entity", default="tiachen", help="Wandb entity")
    p.add_argument("--wandb-project", default="swissai-eval-long-context-8b-mixture-long-ctx", help="Wandb project")
    p.add_argument("--wandb-id", required=True, help="Wandb id")
    p.add_argument("--iter", type=int, default=-1, help="Iteration to evaluate")
    p.add_argument("--overwrite-max-seq-length", type=int, default=-1, help="Overwrite max sequence length to avoid generation errors in sglang (see https://github.com/sgl-project/sglang/blob/4953f4ca9a3a440168cb4a0e9d1e4ae883c97d52/python/sglang/srt/layers/rotary_embedding.py#L130). This does not affect rope scaling (at least in sglang).")

    args = p.parse_args()
    
    # Validate Megatron-LM directory if needed
    if args.megatron_lm_dir:
        megatron_path = Path(args.megatron_lm_dir).expanduser().resolve()
        if not megatron_path.exists():
            p.error(f"Megatron-LM directory not found: {megatron_path}")
        args.megatron_lm_dir = megatron_path
        
    return args


def main() -> None:
    args = parse_args()

    ckpt_dir = args.checkpoint.expanduser().resolve()
    if not ckpt_dir.exists():
        sys.exit(f"Checkpoint path not found: {ckpt_dir}")

    ckpt_type = detect_ckpt_type(ckpt_dir)
    print(f"[info] detected checkpoint type: {ckpt_type}")

    hf_out = (
        args.hf_out.expanduser().resolve()
        if args.hf_out is not None
        else ckpt_dir.with_name(ckpt_dir.name + "_hf")
    )

    if ckpt_type == "hf":
        print("[info] checkpoint already in HuggingFace format — skipping conversion")
    elif ckpt_type == "megatron":
        if not args.megatron_lm_dir:
            sys.exit("Need --megatron-lm-dir (or MEGATRON_LM_DIR env) to convert Megatron checkpoints")
        print(f"[info] converting Megatron checkpoint → HF under {hf_out}")
        convert_megatron_to_hf(
            ckpt_dir=ckpt_dir,
            megatron_lm_dir=args.megatron_lm_dir.expanduser().resolve(),
            hf_out=hf_out,
            iter=args.iter,
            overwrite_max_seq_length=args.overwrite_max_seq_length,
        )
    else:
        sys.exit("Unrecognised checkpoint structure — aborting")

    if args.skip_eval:
        print("[info] --skip-eval passed; exiting after conversion")
        return

    extra_args = [
        "--wandb_args",
        f"entity={args.wandb_entity},project={args.wandb_project},id={args.wandb_id}",
        "--metadata='{\"max_seq_lengths\":[" + args.ruler_lengths + "]}'",
    ]
    run_lm_eval(
        hf_dir=hf_out,
        tasks=[t.strip() for t in args.tasks.split(",") if t.strip()],
        dp_size=args.dp_size,
        tp_size=args.tp_size,
        context_length=args.context_length,
        dtype=args.dtype,
        batch_size=args.batch_size,
        extra_lm_eval_args=extra_args,
    )


if __name__ == "__main__":
    main()
