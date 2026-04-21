# AFL

This is the code repository for the paper titled **"Agentic Feedback Loop Modeling Improves Recommendation and User Simulation"**.

## What has been implemented so far

### 1) Local LLM support (offline-friendly)
- Added a local LLM abstraction layer:
  - `afl/llm_interface.py`
  - `afl/local_qwen.py`
  - `utils/local_model.py`
- Updated API routing to local inference in:
  - `utils/api_request.py`

### 2) Amazon dataset pipeline
- Added Amazon preprocessing and dataset support:
  - `dataset/amazon_preprocess.py`
  - `dataset/amazon_dataset.py`
  - `dataset/prepare_amazon_ab.py`
- Added SASRec/prep tooling:
  - `afl/prepare_amazon_artifacts.py`
  - `afl/prepare_amazon_artifacts_org.py`

### 3) Evaluation workflow extensions
- Added/updated recommendation and user-simulation evaluation for Amazon and local inference:
  - `afl/evaluate_rec.py`
  - `afl/evaluate_user_step1.py`
  - `afl/evaluate_user_step2.py`
  - `afl/evaluate_rec_llara.py`
- Added local agent variant:
  - `utils/agent_llara.py`

### 4) Prompting and parser robustness
- Added Amazon prompt constants:
  - `constant/amazon_ab_model_prompt.py`
  - `constant/amazon_prior_model_prompt.py`
- Improved tolerant parsing in:
  - `utils/regular_function.py`

### 5) Image download + encoding utilities
- Added image utilities:
  - `dataset/download_images.py`
  - `dataset/encode_images.py`
- `dataset/amazon_preprocess.py` now writes:
  - `id2name.txt`
  - `id2imgurl.txt`

## Amazon data flow (recommended order)

Run these steps in sequence:

1. **Preprocess Amazon data**
   ```bash
   python dataset/amazon_preprocess.py \
     --data_dir ./data/amazon_beauty \
     --category Beauty \
     --min_user_interactions 20
   ```

2. **Download item images**
   ```bash
   python dataset/download_images.py \
     --id2imgurl ./data/amazon_beauty/id2imgurl.txt \
     --img_dir ./data/amazon_beauty/images \
     --workers 32 \
     --timeout 15
   ```

3. **Encode images with CLIP**
   ```bash
   python dataset/encode_images.py \
     --img_dir ./data/amazon_beauty/images \
     --output_path ./data/amazon_beauty/image_embeddings.npy \
     --batch_size 64
   ```

## Notes
- If `image_embeddings.npy` contains 0 items, check that:
  1. `id2imgurl.txt` has non-empty URLs.
  2. `images/` actually contains downloaded image files.
  3. Filenames are numeric item IDs (e.g., `123.jpg`) so they map to item indices.

## SASRec PyTorch port (`_pytorch` naming)
- Added a separate port under `afl/sasrec_pytorch/`:
  - `main_pytorch.py`
  - `model_pytorch.py`
  - `utils_pytorch.py`
  - `README.md` (detailed usage and behavior)
- Added training wrappers:
  - `scripts/train_sasrec_amazon_beauty_pytorch.sh`
  - `scripts/slurm_train_sasrec_amazon_beauty_pytorch.sh`

## DROS baseline model wrappers (Amazon Beauty)
- Added shell wrappers for DROS-style baselines (after integrating DROS `.py` files under `afl/dros/`):
  - `scripts/train_dros_sasrec_amazon_beauty.sh`
  - `scripts/train_dros_caser_amazon_beauty.sh`
  - `scripts/train_dros_gru4_amazon_beauty.sh`
- Added a shared grid-search launcher:
  - `scripts/train_dros_model_amazon_beauty.sh`
- Added SLURM launchers:
  - `scripts/train_dros_sasrec_amazon_beauty.slurm`
  - `scripts/train_dros_caser_amazon_beauty.slurm`
  - `scripts/train_dros_gru4_amazon_beauty.slurm`
- Defaults used:
  - `lr=0.001`
  - `batch_size=256`
  - `hidden_factor/emb_dim=64`
  - `l2_decay` grid: `1e-3,1e-4,1e-5,1e-6,1e-7`

## Feature-centric / multimodal backbone (Amazon)

Keep interaction `.df` files unchanged, and build parallel item feature artifacts:

1. Build item metadata table + text embeddings + aligned image/text matrices:
   ```bash
   bash scripts/prepare_amazon_feature_backbone.sh \
     DATA_DIR=./data/amazon_beauty \
     META_GZ=./data/amazon_beauty/meta_Beauty.json.gz
   ```

   This produces (under `DATA_DIR` by default):
   - `item_meta.df`
   - `item_text_emb.npy`
   - `item_image_emb.npy`
   - `multimodal_alignment_report.json`

2. Train SASRec with multimodal fusion (optional flags):
   ```bash
   USE_TEXT_FEAT=1 USE_IMAGE_FEAT=1 FUSION_TYPE=concat \
   TEXT_FEAT_PATH=./data/amazon_beauty/item_text_emb.npy \
   IMAGE_FEAT_PATH=./data/amazon_beauty/item_image_emb.npy \
   bash scripts/train_sasrec_amazon_beauty.sh
   ```

3. Prompt-side semantic enrichment for AFL rec/user simulation:
   pass `--use_item_meta_prompt` (and optional `--item_meta_path`) in `afl/evaluate_rec.py`.

Ablation-ready modes:
- ID-only: default (no multimodal flags)
- ID+image: `USE_IMAGE_FEAT=1`
- ID+text: `USE_TEXT_FEAT=1`
- ID+text+image: `USE_TEXT_FEAT=1 USE_IMAGE_FEAT=1`
