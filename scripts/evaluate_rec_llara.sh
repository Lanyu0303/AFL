#!/bin/bash
set -euo pipefail

DATA_DIR="./data/lastfm/"
STAGE="test"
CANS_NUM=20
MAX_EPOCH=1
MODEL_PATH="./output/lastfm/SASRec.pt"
PRIOR_FILE="./output/lastfm/prior.csv"
OUTPUT_FILE="./output/lastfm/rec_llara.jsonl"
LLM_PATH=""
LLARA_CKPT_PATH=""
MAX_NEW_TOKENS=256
TOP_P=0.9
PROMPT_STYLE="llama2_chat"
SEED=303
MP=1
TEMPERATURE=0.0

mkdir -p ./output/lastfm

CUDA_VISIBLE_DEVICES=0 python ./afl/evaluate_rec_llara.py \
  --data_dir="$DATA_DIR" \
  --model_path="$MODEL_PATH" \
  --prior_file="$PRIOR_FILE" \
  --stage="$STAGE" \
  --cans_num="$CANS_NUM" \
  --max_epoch="$MAX_EPOCH" \
  --output_file="$OUTPUT_FILE" \
  --seed="$SEED" \
  --mp="$MP" \
  --temperature="$TEMPERATURE" \
  --llm_path="$LLM_PATH" \
  --llara_ckpt_path="$LLARA_CKPT_PATH" \
  --prompt_style="$PROMPT_STYLE" \
  --max_new_tokens="$MAX_NEW_TOKENS" \
  --top_p="$TOP_P"
