#!/bin/bash
#SBATCH --job-name=sasrec_pytorch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute
#SBATCH --account=recsys-y8xnv0gztug-default-cpu
#SBATCH --output=logs/sasrec_pytorch_%j.out
#SBATCH --error=logs/sasrec_pytorch.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Optional env activation
# source ~/.bashrc
# conda activate afl

# Force CPU execution for this profile.
export DEVICE=cpu
export DATASET="${DATASET:-Beauty}"

bash scripts/train_sasrec_amazon_beauty_pytorch.sh
