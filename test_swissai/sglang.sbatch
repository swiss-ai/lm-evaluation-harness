#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/image/sglang/sglang.toml
#SBATCH --job-name=1b-sglang
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:59:00
#SBATCH --partition=normal
#SBATCH --exclusive

# Step 0: Some useful logs.
export MASTER_ADDR=$(hostname)
echo "Using nodes: $SLURM_JOB_NODELIST"

TEST_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/lm-evaluation-harness/test_swissai
# srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")'

# Parse MODEL_TAG as input argument
MODEL_TAG="1B-base-rope-fix"
CKPT_PATH=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
# MODEL_TAG="1B-base-no-rope-fix"
# CKPT_PATH=/iopsstor/scratch/cscs/schlag/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
# MODEL_TAG="1B-concat"
# CKPT_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/framework_prepare/Megatron-LM/logs/Meg-Runs/long-context-ablation-1b/apertus3-1b-concat/checkpoints/
# MODEL_TAG="1B-no-concat"
# CKPT_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/framework_prepare/Megatron-LM/logs/Meg-Runs/long-context-ablation-1b/apertus3-1b-no-concat/checkpoints/
# MODEL_TAG="1B-random-concat"
# CKPT_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/framework_prepare/Megatron-LM/logs/Meg-Runs/long-context-ablation-1b/apertus3-1b-random-concat/checkpoints/
# MODEL_TAG="1B-tsp-concat"
# CKPT_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/framework_prepare/Megatron-LM/logs/Meg-Runs/long-context-ablation-1b/apertus3-1b-tsp-concat/checkpoints/


MEGATRON_LM_DIR=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_DIR=$TEST_PATH/$MODEL_TAG/wandb
mkdir -p $WANDB_DIR
export HF_HOME=/iopsstor/scratch/cscs/ctianche/huggingface
# set -e

mkdir -p /iopsstor/scratch/cscs/ctianche/.tmp
# HF_TEMP_PATH=\$(mktemp -d -p /iopsstor/scratch/cscs/ctianche/.tmp)  # To store hf conversion (if needed).
# TORCH_NODIST_PATH=\$(mktemp -d -p /iopsstor/scratch/cscs/ctianche/.tmp)  # To store torch no dist checkpoint converted (if needed).
# REPOS_PATH=\$(mktemp -d -p /iopsstor/scratch/cscs/ctianche/.tmp)  # To git clone repos.

# export TOKEN_ID_OUT_OF_RANGE_BEHAVIOR="fix_warn"
# pip install /capstor/scratch/cscs/ctianche/swissai_long_context/eval/transformers

cd $TEST_PATH
python eval_ckpt.py $CKPT_PATH \
    --megatron-lm-dir $MEGATRON_LM_DIR \
    --hf-out $HF_HOME/$MODEL_TAG \
    --dp-size 1 --tp-size 4 --wandb-id $MODEL_TAG

SUCCESS=1
echo Evaluation finished.
