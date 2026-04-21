#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-./data/amazon_beauty}"
OUTPUT_AB_DIR="${OUTPUT_AB_DIR:-./data/amazon_beauty_ab}"
COPY_FROM_DIR="${COPY_FROM_DIR:-}"
MIN_USER_INTERACTIONS="${MIN_USER_INTERACTIONS:-20}"
WINDOW_SIZE="${WINDOW_SIZE:-20}"

python dataset/prepare_amazon_ab.py \
  --source_data_dir "${SOURCE_DATA_DIR}" \
  --output_dir "${OUTPUT_AB_DIR}" \
  --copy_from_dir "${COPY_FROM_DIR}" \
  --min_user_interactions "${MIN_USER_INTERACTIONS}" \
  --window_size "${WINDOW_SIZE}"

echo "Amazon AB content ready at ${OUTPUT_AB_DIR}"
echo "Expected files:"
echo "  - ${OUTPUT_AB_DIR}/id2name_AB_llara.txt (from ${SOURCE_DATA_DIR}, fallback=${COPY_FROM_DIR:-none})"
echo "  - ${OUTPUT_AB_DIR}/data_statis.df (from ${SOURCE_DATA_DIR}, fallback=${COPY_FROM_DIR:-none})"
echo "  - ${OUTPUT_AB_DIR}/test_llara.pkl (generated)"
