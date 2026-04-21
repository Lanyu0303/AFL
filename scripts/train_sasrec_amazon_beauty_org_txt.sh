#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

TXT_PATH="${TXT_PATH:-./data/amazon_beauty_ab/Beauty.txt}"
AB_DATA_DIR="${AB_DATA_DIR:-./data/amazon_beauty_ab}"
MODEL_PATH="${MODEL_PATH:-./output/amazon_beauty/SASRec_org_txt.pt}"
PRIOR_FILE="${PRIOR_FILE:-./data/prior_amazon/Beauty_prior_amazon_org_txt.csv}"
PRIOR_FILE_AB="${PRIOR_FILE_AB:-./data/prior_amazon/SASRec_AB_20_org_txt.csv}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-0.001}"
HIDDEN_UNITS="${HIDDEN_UNITS:-64}"
MAXLEN="${MAXLEN:-50}"
SEED="${SEED:-42}"

python afl/prepare_amazon_artifacts_org.py \
  --txt_path "${TXT_PATH}" \
  --ab_data_dir "${AB_DATA_DIR}" \
  --output_model_path "${MODEL_PATH}" \
  --output_prior_file "${PRIOR_FILE}" \
  --output_prior_file_ab "${PRIOR_FILE_AB}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --hidden_units "${HIDDEN_UNITS}" \
  --maxlen "${MAXLEN}" \
  --seed "${SEED}"
