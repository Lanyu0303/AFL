#!/bin/bash
#SBATCH --job-name=dros_caser_ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute
#SBATCH --account=recsys-y8xnv0gztug-default-cpu
#SBATCH --output=logs/dros_caser_ab_%j.out
#SBATCH --error=logs/dros_caser_ab_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

bash scripts/train_dros_caser_amazon_beauty.sh
