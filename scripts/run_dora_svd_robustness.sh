#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

"${PYTHON_BIN}" testing_compression_with_dora_svd/run_robustness_experiments.py \
  --output_root testing_compression_with_dora_svd/robustness_results \
  --bf16 \
  "$@"
