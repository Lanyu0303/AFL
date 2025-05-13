work_dir=''
cd $work_dir

DATA_DIR="./data/lastfm/"
STAGE="test"
CANS_NUM=20
MAX_EPOCH=5
MODEL="gpt-4o-mini-2024-07-18" 
API_KEY="" 
MODEL_PATH="./output/lastfm/SASRec.pt"
MAX_RETRY_NUM=5
SEED=303
MP=16
TEMPERATURE=0.0
LABEL="recommendation_experiment"
P_MODEL='SASRec'
PRIOR_FILE="./data/prior_lastfm/SASRec.csv"
OUTPUT_FILE="./data/lastfm_rec_eval/${P_MODEL}_${LABEL}_${MODEL}_${SEED}_${TEMPERATURE}_${MP}_${MAX_EPOCH}.jsonl"

SAVE_REC_DIR="./data/lastfm_rec_save/rec_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"
SAVE_USER_DIR="./data/lastfm_rec_save/user_${P_MODEL}_${LABEL}_${MODEL}_${MAX_EPOCH}"

echo "DATA_DIR = ${DATA_DIR}"
echo "MODEL_PATH = ${MODEL_PATH}"
echo "PRIOR_FILE = ${PRIOR_FILE}"
echo "STAGE = ${STAGE}"
echo "CANS_NUM = ${CANS_NUM}"
echo "MAX_EPOCH = ${MAX_EPOCH}"
echo "MODEL = ${MODEL}"
echo "API_KEY = ${API_KEY}"
echo "MAX_RETRY_NUM = ${MAX_RETRY_NUM}"
echo "SEED = ${SEED}"
echo "TEMPERATURE = ${TEMPERATURE}"
echo "MP = ${MP}"
echo "OUTPUT_FILE = ${OUTPUT_FILE}"
echo "SAVE_REC_DIR = ${SAVE_REC_DIR}"
echo "SAVE_USER_DIR = ${SAVE_USER_DIR}"

CUDA_VISIBLE_DEVICES=0 python ./afl/evaluate_rec.py \
    --data_dir=$DATA_DIR \
    --model_path=$MODEL_PATH \
    --prior_file=$PRIOR_FILE \
    --stage=$STAGE \
    --cans_num=$CANS_NUM \
    --max_epoch=$MAX_EPOCH \
    --output_file=$OUTPUT_FILE \
    --model=$MODEL \
    --api_key=$API_KEY \
    --max_retry_num=$MAX_RETRY_NUM \
    --seed=$SEED \
    --mp=$MP \
    --temperature=$TEMPERATURE \
    --save_info \
    --save_rec_dir=$SAVE_REC_DIR \
    --save_user_dir=$SAVE_USER_DIR



    

