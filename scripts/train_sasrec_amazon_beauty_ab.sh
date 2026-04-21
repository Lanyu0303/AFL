#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Train SASRec on Amazon Beauty train/val and generate prior CSV for *_ab user simulation.
DATA_DIR="${DATA_DIR:-./data/amazon_beauty}"
AB_DATA_DIR="${AB_DATA_DIR:-./data/amazon_beauty_ab}"
MODEL_PATH="${MODEL_PATH:-./output/amazon_beauty_ab/SASRec_AB.pt}"
PRIOR_FILE="${PRIOR_FILE:-./data/prior_amazon/Beauty_prior_amazon.csv}"
PRIOR_FILE_AB="${PRIOR_FILE_AB:-./data/prior_amazon/SASRec_AB_20.csv}"

LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
L2_GRID="${L2_GRID:-1e-3,1e-4,1e-5,1e-6,1e-7}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"
AB_TRAIN_CANS_NUM="${AB_TRAIN_CANS_NUM:-20}"

echo "Training SASRec for Amazon AB prior generation"
echo "DATA_DIR=${DATA_DIR}"
echo "AB_DATA_DIR=${AB_DATA_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PRIOR_FILE=${PRIOR_FILE}"
echo "PRIOR_FILE_AB=${PRIOR_FILE_AB}"
echo "AB_TRAIN_CANS_NUM=${AB_TRAIN_CANS_NUM}"

python afl/prepare_amazon_artifacts.py \
  --data_dir "${DATA_DIR}" \
  --ab_data_dir "${AB_DATA_DIR}" \
  --ab_train_cans_num "${AB_TRAIN_CANS_NUM}" \
  --output_model_path "${MODEL_PATH}" \
  --output_prior_file "${PRIOR_FILE}" \
  --output_prior_file_ab "${PRIOR_FILE_AB}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --l2_grid "${L2_GRID}" \
  --seed "${SEED}"

echo "Done. Generated:"
echo "  - ${MODEL_PATH}"
echo "  - ${PRIOR_FILE}"
echo "  - ${PRIOR_FILE_AB}"
