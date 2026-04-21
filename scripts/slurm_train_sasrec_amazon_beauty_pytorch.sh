#!/bin/bash
#SBATCH --job-name=sasrec_pt_beauty
#SBATCH --output=logs/sasrec_pt_beauty_%j.out
#SBATCH --error=logs/sasrec_pt_beauty_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Optional: activate your environment
# source ~/.bashrc
# conda activate afl

export DATASET="${DATASET:-./data/amazon_beauty}"
export TRAIN_DIR="${TRAIN_DIR:-amazon_beauty_pytorch_preln}"
export DEVICE="${DEVICE:-cuda}"

export BATCH_SIZE="${BATCH_SIZE:-128}"
export LR="${LR:-0.001}"
export MAXLEN="${MAXLEN:-50}"
export HIDDEN_UNITS="${HIDDEN_UNITS:-50}"
export NUM_BLOCKS="${NUM_BLOCKS:-2}"
export NUM_EPOCHS="${NUM_EPOCHS:-1000}"
export NUM_HEADS="${NUM_HEADS:-1}"
export DROPOUT_RATE="${DROPOUT_RATE:-0.5}"
export L2_EMB="${L2_EMB:-0.0}"

bash scripts/train_sasrec_amazon_beauty_pytorch.sh
