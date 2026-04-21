#!/bin/bash
set -euo pipefail
MODEL=caser bash "$(dirname "$0")/train_dros_model_amazon_beauty.sh"
