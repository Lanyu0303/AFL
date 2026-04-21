#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-./data/amazon_ab}"
INIT_NUM="${INIT_NUM:-10}"
TRAIN_CANS_NUM="${TRAIN_CANS_NUM:-20}"
EVAL_CANS_NUM="${EVAL_CANS_NUM:-20}"
A_RATIO="${A_RATIO:-1}"
B_RATIO="${B_RATIO:-1}"
STAGE="${STAGE:-test}"
MODEL_PATH="${MODEL_PATH:-./output/amazon_beauty_text/SASRec.pt}"
INPUT_FILE="${INPUT_FILE:-./data/amazon_ab_step1/amazon_user_step1.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-./data/amazon_ab_step2/amazon_user_step2.jsonl}"
SEED="${SEED:-303}"
MP="${MP:-4}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-Qwen/Qwen2-1.5B-Instruct}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TOP_P="${TOP_P:-0.9}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:---load_in_4bit}"

CUDA_VISIBLE_DEVICES=0 python ./afl/evaluate_user_step2.py \
    --data_dir="${DATA_DIR}" \
    --init_num="${INIT_NUM}" \
    --input_file="${INPUT_FILE}" \
    --train_cans_num="${TRAIN_CANS_NUM}" \
    --eval_cans_num="${EVAL_CANS_NUM}" \
    --a_ratio="${A_RATIO}" \
    --b_ratio="${B_RATIO}" \
    --stage="${STAGE}" \
    --output_file="${OUTPUT_FILE}" \
    --seed="${SEED}" \
    --mp="${MP}" \
    --local_model_name="${LOCAL_MODEL_NAME}" \
    --top_p="${TOP_P}" \
    ${LOAD_IN_4BIT} \
    --max_new_tokens="${MAX_NEW_TOKENS}" \
    --model_path="${MODEL_PATH}"
