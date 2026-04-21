#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CATEGORY="${CATEGORY:-All_Beauty}"
DATA_DIR="${DATA_DIR:-./data/amazon_all_beauty}"
MIN_USER_INTERACTIONS="${MIN_USER_INTERACTIONS:-20}"

python dataset/amazon_preprocess.py \
  --data_dir "${DATA_DIR}" \
  --category "${CATEGORY}" \
  --min_user_interactions "${MIN_USER_INTERACTIONS}"

echo "Amazon category ${CATEGORY} prepared at ${DATA_DIR}"
