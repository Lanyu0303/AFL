#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Use this script when Amazon data is already preprocessed.
DATA_DIR="${DATA_DIR:-./data/amazon_beauty}"
MODEL_PATH="${MODEL_PATH:-./output/amazon_beauty/SASRec.pt}"
PRIOR_FILE="${PRIOR_FILE:-./data/prior_amazon/Beauty_prior_amazon.csv}"
STAGE="${STAGE:-test}"
CANS_NUM="${CANS_NUM:-20}"
LR="${LR:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
L2_GRID="${L2_GRID:-1e-3,1e-4,1e-5,1e-6,1e-7}"
EPOCHS="${EPOCHS:-10}"
SEED="${SEED:-42}"

echo "Training SASRec from preprocessed dataset"
echo "DATA_DIR=${DATA_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PRIOR_FILE=${PRIOR_FILE}"

python afl/prepare_amazon_artifacts.py \
  --data_dir "${DATA_DIR}" \
  --output_model_path "${MODEL_PATH}" \
  --output_prior_file "${PRIOR_FILE}" \
  --stage "${STAGE}" \
  --cans_num "${CANS_NUM}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --embedding_dim "${EMBEDDING_DIM}" \
  --l2_grid "${L2_GRID}" \
  --seed "${SEED}"
