#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-sasrec}"          # sasrec | caser | gru4
DROS_DIR="${DROS_DIR:-./afl/dros}" # folder where DROS *.py files are integrated
DATA="${DATA:-amazon_beauty}"      # expected to map to ./data/amazon_beauty
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EMB_DIM="${EMB_DIM:-64}"
LR="${LR:-0.001}"
L2_GRID="${L2_GRID:-1e-3,1e-4,1e-5,1e-6,1e-7}"
CUDA_ID="${CUDA_ID:-0}"
ALPHA="${ALPHA:-0}"
BETA="${BETA:-1.0}"

case "${MODEL}" in
  sasrec) PY_SCRIPT="SASRec_bce.py" ;;
  caser)  PY_SCRIPT="Caser_bce.py" ;;
  gru4)   PY_SCRIPT="Gru_bce.py" ;;
  *)
    echo "Unsupported MODEL=${MODEL}. Use one of: sasrec, caser, gru4" >&2
    exit 1
    ;;
esac

if [ ! -f "${DROS_DIR}/${PY_SCRIPT}" ]; then
  echo "Could not find ${DROS_DIR}/${PY_SCRIPT}" >&2
  echo "Please integrate DROS scripts first (SASRec_bce.py, Caser_bce.py, Gru_bce.py)." >&2
  exit 1
fi

echo "MODEL=${MODEL}"
echo "DROS_DIR=${DROS_DIR}"
echo "DATA=${DATA}"
echo "EPOCHS=${EPOCHS}, BATCH_SIZE=${BATCH_SIZE}, EMB_DIM=${EMB_DIM}, LR=${LR}, L2_GRID=${L2_GRID}"

IFS=',' read -r -a L2S <<< "${L2_GRID}"
for L2 in "${L2S[@]}"; do
  echo "===================================================="
  echo "Running ${PY_SCRIPT} with l2_decay=${L2}"
  echo "===================================================="
  python "${DROS_DIR}/${PY_SCRIPT}" \
    --data "${DATA}" \
    --epoch "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --hidden_factor "${EMB_DIM}" \
    --lr "${LR}" \
    --l2_decay "${L2}" \
    --cuda "${CUDA_ID}" \
    --alpha "${ALPHA}" \
    --beta "${BETA}"
done
