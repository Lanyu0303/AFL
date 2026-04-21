#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="${DATASET:-Beauty}"
TRAIN_DIR="${TRAIN_DIR:-amazon_beauty_pytorch_preln}"
DEVICE="${DEVICE:-cuda}"

# Hyperparameters requested for Amazon Beauty reproduction
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
MAXLEN="${MAXLEN:-50}"
HIDDEN_UNITS="${HIDDEN_UNITS:-50}"
NUM_BLOCKS="${NUM_BLOCKS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1000}"
NUM_HEADS="${NUM_HEADS:-1}"
DROPOUT_RATE="${DROPOUT_RATE:-0.5}"
L2_EMB="${L2_EMB:-0.0}"

echo "Running SASRec PyTorch implementation"
echo "DATASET=${DATASET}"
echo "TRAIN_DIR=${TRAIN_DIR}"
echo "DEVICE=${DEVICE}"
echo "Artifacts will be written under: $(basename "${DATASET}")_${TRAIN_DIR}/"
echo "BATCH_SIZE=${BATCH_SIZE}, LR=${LR}, MAXLEN=${MAXLEN}, HIDDEN_UNITS=${HIDDEN_UNITS}, NUM_BLOCKS=${NUM_BLOCKS}, NUM_EPOCHS=${NUM_EPOCHS}, NUM_HEADS=${NUM_HEADS}, DROPOUT_RATE=${DROPOUT_RATE}, L2_EMB=${L2_EMB}"
echo "norm_first=TRUE (Pre-LN)"

python afl/sasrec_pytorch/main_pytorch.py \
  --dataset "${DATASET}" \
  --train_dir "${TRAIN_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --maxlen "${MAXLEN}" \
  --hidden_units "${HIDDEN_UNITS}" \
  --num_blocks "${NUM_BLOCKS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_heads "${NUM_HEADS}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --l2_emb "${L2_EMB}" \
  --device "${DEVICE}" \
  --norm_first
