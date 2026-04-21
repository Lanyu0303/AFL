#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CATEGORY="${CATEGORY:-Beauty}"
DATA_DIR="${DATA_DIR:-./data/amazon_beauty}"
MODEL_PATH="${MODEL_PATH:-./output/amazon_beauty/SASRec.pt}"
PRIOR_FILE="${PRIOR_FILE:-./data/prior_amazon/Beauty_prior_amazon.csv}"
AB_DATA_DIR="${AB_DATA_DIR:-}"
PRIOR_FILE_AB="${PRIOR_FILE_AB:-./data/prior_amazon/SASRec_AB_20.csv}"
GENERATE_AB_PRIOR="${GENERATE_AB_PRIOR:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-1}"

MIN_USER_INTERACTIONS="${MIN_USER_INTERACTIONS:-20}"
SPLIT_MODE="${SPLIT_MODE:-leave_one_out}"

LR="${LR:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
L2_GRID="${L2_GRID:-1e-4}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
CANS_NUM="${CANS_NUM:-20}"

USE_TEXT_FEAT="${USE_TEXT_FEAT:-0}"
USE_IMAGE_FEAT="${USE_IMAGE_FEAT:-0}"
USE_META_FEAT="${USE_META_FEAT:-0}"
FUSION_TYPE="${FUSION_TYPE:-concat}"
TEXT_FEAT_PATH="${TEXT_FEAT_PATH:-${DATA_DIR}/item_text_emb.npy}"
IMAGE_FEAT_PATH="${IMAGE_FEAT_PATH:-${DATA_DIR}/item_image_emb.npy}"
META_FEAT_PATH="${META_FEAT_PATH:-}"

EXTRA_ARGS=(--fusion_type "${FUSION_TYPE}")
if [ "${USE_TEXT_FEAT}" = "1" ]; then
  EXTRA_ARGS+=(--use_text_feat --text_feat_path "${TEXT_FEAT_PATH}")
fi
if [ "${USE_IMAGE_FEAT}" = "1" ]; then
  EXTRA_ARGS+=(--use_image_feat --image_feat_path "${IMAGE_FEAT_PATH}")
fi
if [ "${USE_META_FEAT}" = "1" ]; then
  EXTRA_ARGS+=(--use_meta_feat --meta_feat_path "${META_FEAT_PATH}")
fi

if [ "${SKIP_PREPROCESS}" != "1" ]; then
  python dataset/amazon_preprocess.py \
    --data_dir "${DATA_DIR}" \
    --category "${CATEGORY}" \
    --min_user_interactions "${MIN_USER_INTERACTIONS}" \
    --split_mode "${SPLIT_MODE}"
fi

if [ "${GENERATE_AB_PRIOR}" = "1" ]; then
  if [ -z "${AB_DATA_DIR}" ]; then
    echo "ERROR: GENERATE_AB_PRIOR=1 requires AB_DATA_DIR to be set." >&2
    exit 1
  fi
  python afl/prepare_amazon_artifacts.py \
    --data_dir "${DATA_DIR}" \
    --ab_data_dir "${AB_DATA_DIR}" \
    --ab_train_cans_num "${CANS_NUM}" \
    --output_model_path "${MODEL_PATH}" \
    --output_prior_file "${PRIOR_FILE}" \
    --output_prior_file_ab "${PRIOR_FILE_AB}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --embedding_dim "${EMBEDDING_DIM}" \
    --l2_grid "${L2_GRID}" \
    --seed "${SEED}" \
    --cans_num "${CANS_NUM}" \
    "${EXTRA_ARGS[@]}"
else
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
    --cans_num "${CANS_NUM}" \
    "${EXTRA_ARGS[@]}"
fi
