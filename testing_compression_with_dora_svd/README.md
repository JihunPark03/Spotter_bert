# DoRA Post-Training Direction SVD Compression

This directory contains the code and results for **DoRA with post-training
direction SVD compression**.

The method compresses only the LoRA-style direction update inside DoRA:

```text
W_prime = m * normalize(W_base + Delta_W)
Delta_W = B @ A
```

The proposed compression applies SVD to `Delta_W = B @ A` after training. It
does not prune the DoRA magnitude parameter `m`, does not prune `W_base`, and
does not apply SVD to the full `m * normalize(...)` weight.

## Method

For each adapted layer `l`, compute:

```text
Delta_W_l = B_l @ A_l
Delta_W_l = U_l diag(sigma_l) V_l^T
```

Then choose the smallest layer-specific rank `k_l` that preserves the target
singular-value energy:

```text
(sigma_l,1^2 + ... + sigma_l,k^2)
/
(sigma_l,1^2 + ... + sigma_l,16^2)
>= tau
```

The main experiments use:

```text
tau = 0.90
tau = 0.95
```

The pruned direction update is reconstructed as:

```text
Delta_W_hat_l = U_l,k diag(sigma_l,k) V_l,k^T
```

and converted back into LoRA factors. The DoRA magnitude branch is kept
unchanged.

## Kept Files

Core implementation:

```text
train_dora_svd.py
dora_svd_pruning.py
run_robustness_experiments.py
export_compact_adapter.py
validate_compact_adapter.py
```

Main result directory:

```text
paper_main_energy_results/
```

Important result summaries:

```text
paper_main_energy_results/robustness_results.csv
paper_main_energy_results/compact_export_summary.csv
paper_main_energy_results/compact_validation_results.csv
```

## Main Experiment

The paper-oriented experiment compares native DoRA ranks against rank-16 DoRA
compressed after training with layer-wise SVD energy thresholds.

```bash
nohup scripts/run_dora_svd_robustness.sh \
  --preset paper_main_energy \
  --output_root testing_dora_svd/paper_main_energy_results \
  --skip_existing \
  > testing_dora_svd/paper_main_energy.log 2>&1 &
```

Grid:

```text
Datasets: SST-2, IMDb
Noise: clean, 20% label noise, 40% label noise
Data sizes: 1000, 5000
Methods:
  - DoRA rank12
  - DoRA rank16
  - DoRA rank16 -> post-training direction SVD energy0.90
  - DoRA rank16 -> post-training direction SVD energy0.95
Seeds: 13, 21, 42
Total: 144 runs
```

Collect existing results without rerunning:

```bash
testing/venv/bin/python testing_dora_svd/run_robustness_experiments.py \
  --preset paper_main_energy \
  --output_root testing_dora_svd/paper_main_energy_results \
  --collect_only
```

## Compact Adapter Export

Post-training SVD lowers the effective rank inside the rank-16 DoRA adapter.
To claim real checkpoint/storage reduction, export compact adapter artifacts
whose LoRA A/B tensors are physically sliced to each layer's kept rank.

```bash
testing/venv/bin/python testing_dora_svd/export_compact_adapter.py \
  testing_dora_svd/paper_main_energy_results \
  --overwrite
```

Each SVD experiment writes:

```text
compact_svd_adapter/adapter_model.safetensors
compact_svd_adapter/manifest.json
compact_svd_adapter/layers.csv
```

The root summary is:

```text
testing_dora_svd/paper_main_energy_results/compact_export_summary.csv
```

## Compact Adapter Validation

Validate compact artifacts by loading compact A/B tensors, zero-padding them
back to the original PEFT rank, and comparing them against the full-shape
pruned adapter.

```bash
nohup testing/venv/bin/python testing_dora_svd/validate_compact_adapter.py \
  testing_dora_svd/paper_main_energy_results \
  --evaluate_full_adapter \
  --output_csv testing_dora_svd/paper_main_energy_results/compact_validation_results.csv \
  > testing_dora_svd/compact_validation.log 2>&1 &
```

Important columns:

```text
compact_vs_reloaded_full_f1_diff
compact_vs_reloaded_full_accuracy_diff
max_reconstruction_abs_diff
```

In the current completed validation, all three are `0.0`, meaning the compact
artifact reconstructs the same pruned adapter behavior.

## Current Main Results

Mean over SST-2, IMDb, noise levels, data sizes, and 3 seeds:

```text
DoRA rank12:             F1 0.7373, avg rank 12.00
DoRA rank16:             F1 0.7516, avg rank 16.00
Post SVD energy0.90:     F1 0.7504, avg rank  6.15
Post SVD energy0.95:     F1 0.7515, avg rank  8.68
```

Compact export:

```text
Energy0.90: average file size reduction 55.6%
Energy0.95: average file size reduction 39.9%
```

## Nohup Pattern

Use this pattern for background runs:

```bash
nohup COMMAND > log_file 2>&1 &
```

`nohup` keeps the process alive after logout, `> log_file` saves stdout,
`2>&1` saves stderr to the same log, and `&` runs the process in the
background.
