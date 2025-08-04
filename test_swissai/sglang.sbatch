#!/bin/bash
#SBATCH --account=a-infra01-1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --environment=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/image/sglang/sglang.toml
#SBATCH --job-name=8b-sglang
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --exclusive


if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_tag> <iters> <ckpt_path>"
    exit 1
fi
MODEL_TAG="$1"
ITERS="$2"
CKPT_PATH="$3"

# =====================
# Environment Setup
# =====================
export MASTER_ADDR=$(hostname)
echo "Using nodes: $SLURM_JOB_NODELIST"

TEST_PATH=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/lm-evaluation-harness/test_swissai
# srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")'

MEGATRON_LM_DIR=/capstor/scratch/cscs/ctianche/swissai_long_context/eval/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
# prepend iter number to 7 digits
export WANDB_DIR=$TEST_PATH/$MODEL_TAG/iter_$(printf "%07d" $ITERS)
mkdir -p $WANDB_DIR
export HF_HOME=/iopsstor/scratch/cscs/ctianche/huggingface

mkdir -p /iopsstor/scratch/cscs/ctianche/.tmp

cd /capstor/scratch/cscs/ctianche/swissai_long_context/eval/lm-evaluation-harness
python -m pip install -e .[api,longbench,ruler,wandb] 

cd $TEST_PATH

export HF_DATASETS_TRUST_REMOTE_CODE=true


if [ $ITERS -eq -1 ]; then
    HF_OUT=$HF_HOME/$MODEL_TAG
else
    HF_OUT=$HF_HOME/$MODEL_TAG/iter_${ITERS}
fi

python eval_ckpt.py $CKPT_PATH \
    --megatron-lm-dir $MEGATRON_LM_DIR \
    --hf-out $HF_OUT \
    --dp-size 1 --tp-size 4 --wandb-id ${MODEL_TAG}_iter_${ITERS}_job${SLURM_JOB_ID} \
    --batch-size 32 --iter $ITERS --context-length 65536 --ruler-lengths 4096,8192,16384,32768 \
    --overwrite-max-seq-length 32768

SUCCESS=1
echo Evaluation finished.
