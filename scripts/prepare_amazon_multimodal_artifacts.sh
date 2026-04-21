#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Categories aligned with common multimodal Amazon setups (VIP5/ReasonRec style)
CATEGORIES=(${CATEGORIES:-"Clothing_Shoes_and_Jewelry Sports_and_Outdoors Beauty Toys_and_Games"})
BASE_DATA_DIR="${BASE_DATA_DIR:-./data/amazon_mm}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./output/amazon_mm}"
BASE_PRIOR_DIR="${BASE_PRIOR_DIR:-./data/prior_amazon}"
MIN_USER_INTERACTIONS="${MIN_USER_INTERACTIONS:-20}"
SPLIT_MODE="${SPLIT_MODE:-leave_one_out}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
EMBEDDING_DIM="${EMBEDDING_DIM:-64}"
L2_GRID="${L2_GRID:-1e-3,1e-4,1e-5,1e-6,1e-7}"
SEED="${SEED:-42}"
CANS_NUM="${CANS_NUM:-20}"

for CATEGORY in "${CATEGORIES[@]}"; do
  DATA_DIR="${BASE_DATA_DIR}/${CATEGORY}"
  MODEL_PATH="${BASE_OUTPUT_DIR}/${CATEGORY}/SASRec.pt"
  PRIOR_FILE="${BASE_PRIOR_DIR}/${CATEGORY}_prior_amazon.csv"

  echo "=============================="
  echo "Preparing category: ${CATEGORY}"
  echo "DATA_DIR: ${DATA_DIR}"
  echo "MODEL_PATH: ${MODEL_PATH}"
  echo "PRIOR_FILE: ${PRIOR_FILE}"
  echo "=============================="

  python dataset/amazon_preprocess.py \
    --data_dir "${DATA_DIR}" \
    --category "${CATEGORY}" \
    --min_user_interactions "${MIN_USER_INTERACTIONS}" \
    --split_mode "${SPLIT_MODE}"

  python afl/prepare_amazon_artifacts.py \
    --data_dir "${DATA_DIR}" \
    --output_model_path "${MODEL_PATH}" \
    --output_prior_file "${PRIOR_FILE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --embedding_dim "${EMBEDDING_DIM}" \
    --l2_grid "${L2_GRID}" \
    --seed "${SEED}" \
    --cans_num "${CANS_NUM}"
done

echo "All categories completed."
