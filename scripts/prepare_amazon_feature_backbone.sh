#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/amazon_beauty}"
META_GZ="${META_GZ:-${DATA_DIR}/meta_Beauty.json.gz}"
ITEM_META="${ITEM_META:-${DATA_DIR}/item_meta.df}"
TEXT_RAW="${TEXT_RAW:-${DATA_DIR}/item_text_emb_raw.npy}"
TEXT_ALIGNED="${TEXT_ALIGNED:-${DATA_DIR}/item_text_emb.npy}"
IMG_DICT="${IMG_DICT:-${DATA_DIR}/image_embeddings.npy}"
IMG_ALIGNED="${IMG_ALIGNED:-${DATA_DIR}/item_image_emb.npy}"
STATIS="${STATIS:-${DATA_DIR}/data_statis.df}"
REPORT_JSON="${REPORT_JSON:-${DATA_DIR}/multimodal_alignment_report.json}"

python dataset/build_item_meta.py \
  --data_dir "${DATA_DIR}" \
  --meta_gz "${META_GZ}" \
  --output_path "${ITEM_META}"

python dataset/encode_text_embeddings.py \
  --item_meta "${ITEM_META}" \
  --output_path "${TEXT_RAW}"

python dataset/align_multimodal_features.py \
  --data_statis "${STATIS}" \
  --text_embeddings "${TEXT_RAW}" \
  --image_embeddings "${IMG_DICT}" \
  --out_text "${TEXT_ALIGNED}" \
  --out_image "${IMG_ALIGNED}" \
  --report_json "${REPORT_JSON}"

echo "Done. Generated:"
echo "  ${ITEM_META}"
echo "  ${TEXT_ALIGNED}"
echo "  ${IMG_ALIGNED}"
echo "  ${REPORT_JSON}"
