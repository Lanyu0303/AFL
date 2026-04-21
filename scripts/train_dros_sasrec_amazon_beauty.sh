#!/bin/bash
set -euo pipefail
MODEL=sasrec bash "$(dirname "$0")/train_dros_model_amazon_beauty.sh"
