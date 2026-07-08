#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-testing/venv/bin/python}"

"${PYTHON_BIN}" testing_dora_svd/run_robustness_experiments.py \
  --output_root testing_dora_svd/robustness_results \
  --bf16 \
  "$@"
