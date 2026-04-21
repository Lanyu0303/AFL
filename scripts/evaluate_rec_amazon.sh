#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Data/model settings
DATA_DIR="${DATA_DIR:-./data/amazon_all_beauty}"
STAGE="${STAGE:-test}"
CANS_NUM="${CANS_NUM:-20}"
MAX_EPOCH="${MAX_EPOCH:-4}"
MODEL="${MODEL:-Qwen2-1.5B-Instruct}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-Qwen/Qwen2-1.5B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-303}"
MP="${MP:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:---load_in_4bit}"

# Required input artifact:
# 1) SASRec checkpoint trained on Amazon data
# Optional: prior predictions csv with columns id,generate
MODEL_PATH="${MODEL_PATH:-./output/amazon_all_beauty/SASRec.pt}"
PRIOR_FILE="${PRIOR_FILE:-./data/prior_amazon/SASRec.csv}"
PRIOR_MODE="${PRIOR_MODE:-auto}"

LABEL="${LABEL:-amazon_recommendation_experiment}"
P_MODEL="${P_MODEL:-SASRec}"
OUTPUT_FILE="${OUTPUT_FILE:-./data/amazon_rec_eval/${P_MODEL}_${LABEL}_${MODEL}_${SEED}_${TEMPERATURE}_${MP}_${MAX_EPOCH}.jsonl}"
SAVE_REC_DIR="${SAVE_REC_DIR:-./data/amazon_rec_save/rec_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}}"
SAVE_USER_DIR="${SAVE_USER_DIR:-./data/amazon_rec_save/user_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}}"

echo "DATA_DIR = ${DATA_DIR}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "PRIOR_FILE = ${PRIOR_FILE}"
echo "LOCAL_MODEL_NAME = ${LOCAL_MODEL_NAME}"
echo "OUTPUT_FILE = ${OUTPUT_FILE}"
echo "PRIOR_MODE = ${PRIOR_MODE}"

CUDA_VISIBLE_DEVICES=0 python ./afl/evaluate_rec.py \
    --data_dir="${DATA_DIR}" \
    --model_path="${MODEL_PATH}" \
    --prior_file="${PRIOR_FILE}" \
    --prior_mode="${PRIOR_MODE}" \
    --stage="${STAGE}" \
    --cans_num="${CANS_NUM}" \
    --max_epoch="${MAX_EPOCH}" \
    --output_file="${OUTPUT_FILE}" \
    --model="${MODEL}" \
    --seed="${SEED}" \
    --mp="${MP}" \
    --temperature="${TEMPERATURE}" \
    --local_model_name="${LOCAL_MODEL_NAME}" \
    --top_p="${TOP_P}" \
    ${LOAD_IN_4BIT} \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --save_info \
    --save_rec_dir="${SAVE_REC_DIR}" \
    --save_user_dir="${SAVE_USER_DIR}"
