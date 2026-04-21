# SASRec PyTorch (ported files with `_pytorch` naming)

This folder contains a directly ported SASRec PyTorch implementation with naming adapted to avoid confusion with existing AFL modules.

## Files

### `main_pytorch.py`
Training/inference entrypoint.

What it does:
1. Parses CLI args (`--dataset`, `--train_dir`, etc.).
2. Creates output directory `<dataset>_<train_dir>/` and writes `args.txt`.
3. Builds user/item indexes and partitions train/valid/test.
4. Creates `WarpSampler` mini-batch generator.
5. Initializes `SASRec` from `model_pytorch.py`.
6. Trains with BCE loss + optional embedding L2 regularization.
7. Every 20 epochs, evaluates metrics and writes `log.txt`.
8. Saves checkpoints named:
   `SASRec.epoch=...lr=...layer=...head=...hidden=...maxlen=....pth`.

Metrics reported:
- `NDCG@10`
- `HR@10`

Notes:
- `--norm_first` enables Pre-LN.

### `model_pytorch.py`
SASRec architecture (PyTorch) with selectable normalization design.

What it does:
- Defines `PointWiseFeedForward`.
- Defines `SASRec` with:
  - item + positional embeddings
  - multi-block self-attention stack
  - pointwise FFN blocks
  - `norm_first` switch:
    - `True`: Pre-LN
    - `False`: Post-LN
- Exposes:
  - `forward(...)` for training (`pos_logits`, `neg_logits`)
  - `predict(...)` for ranking/evaluation

### `utils_pytorch.py`
Data loading, sampling, splitting, and evaluation.

What it does:
- Loads interactions from `data/<dataset>.txt` in `user_id item_id` format.
- Partitions data per SASRec protocol (last two interactions per user for valid/test when possible).
- Provides multi-process `WarpSampler`.
- Implements:
  - `evaluate` (test)
  - `evaluate_valid` (validation)
  returning `(NDCG@10, HR@10)`.

## Reproduction command (Amazon Beauty hyperparameters requested)

Use the wrapper script:

```bash
bash scripts/train_sasrec_amazon_beauty_pytorch.sh
```

By default, this wrapper trains from `data/Beauty.txt` (`DATASET=Beauty`).

Defaults used there:
- `hidden_units=50`
- `lr=0.001`
- `batch_size=128`
- `num_blocks=2`
- `dropout_rate=0.5`
- `maxlen=50`
- `num_epochs=1000`
- `norm_first=True` (Pre-LN)

## SLURM

Run on cluster with:

```bash
sbatch scripts/slurm_train_sasrec_amazon_beauty_pytorch.sh
```

Override variables at submit time, for example:

```bash
sbatch --export=ALL,DATASET=Beauty,TRAIN_DIR=ab_pt_run1,NUM_EPOCHS=400 scripts/slurm_train_sasrec_amazon_beauty_pytorch.sh
```
