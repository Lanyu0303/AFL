#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

python dataset/check_df_sizes.py --data_dir ./data/amazon_beauty
