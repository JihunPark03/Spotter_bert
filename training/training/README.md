# Training Instructions

This directory trains and compares ModernBERT fake-review detectors with:

- `full`
- `lora`
- `adalora`
- `star_lora`

The base dataset is `testing/data/deepseek_synthetic_reviews.jsonl`. Each JSONL row produces two classification examples: one `source_review` as `REAL` and one `synthetic_review` as `FAKE`. Therefore 20,000 JSONL rows become 40,000 classification samples.

## Single Run

Edit `config.yaml`:

```yaml
fine_tuning:
  method: lora

training:
  output_dir: testing/training/modernbert-large-fake-review-detector-lora
```

Then run:

```bash
testing/venv/bin/python testing/training/main.py
```

To use a different config file:

```bash
TRAIN_CONFIG=testing/training/config.yaml testing/venv/bin/python testing/training/main.py
```

## Data Size Sweep

To compare `full`, `lora`, `adalora`, and `star_lora` across multiple training sample sizes:

Run this from `/home/jihun/Spotter_bert`

```bash
nohup testing/venv/bin/python testing/training/run_datasize_experiments.py \
  --sample-sizes 200 300 400 500 600 700 800 900 \
  --methods full lora adalora star_lora \
  --skip-existing \
  > testing/training/datasize_experiments.log 2>&1 &
```

Outputs are written under:

```text
testing/training/datasize_runs/{method}_n{sample_size}
```

Generated configs are written under:

```text
testing/training/datasize_configs/
```

Sampling is balanced by label. For example, `--sample-sizes 100` means `50 REAL` and `50 FAKE` before the train/test split.

## Summarize Results

After the sweep finishes, build the comparison table from the generated `datasize_runs` folders:

```bash
testing/venv/bin/python testing/training/summarize_results.py \
  --root testing/training/datasize_runs \
  --output testing/training/datasize_comparison
```

This writes:

```text
testing/training/datasize_comparison.csv
testing/training/datasize_comparison.md
```

To print the generated evaluation metrics in the terminal:

```bash
cat testing/training/datasize_comparison.md
```

For a CSV view:

```bash
column -s, -t < testing/training/datasize_comparison.csv | less -S
```

The most important columns are:

- `final_eval_f1`: final evaluation weighted F1 from `experiment_info.json`
- `final_eval_loss`: final evaluation loss from `experiment_info.json`
- `best_eval_f1`: best checkpoint F1 from `trainer_state.json`
- `best_eval_loss`: loss at the best checkpoint

To plot `final_eval_f1` on the y-axis and `num_samples` on the x-axis for each method:

```bash
testing/venv/bin/python testing/training/plot_datasize_f1.py \
  --input testing/training/datasize_comparison.csv \
  --output testing/training/datasize_final_eval_f1.png
```

For a log-scaled x-axis:

```bash
testing/venv/bin/python testing/training/plot_datasize_f1.py \
  --input testing/training/datasize_comparison.csv \
  --output testing/training/datasize_final_eval_f1_logx.png \
  --log-x
```

For the existing 40,000-sample result folders, run:

```bash
testing/venv/bin/python testing/training/summarize_results.py \
  --root testing/training \
  --output testing/training/final_comparison
```

## Notes

- Use `--skip-existing` to resume a sweep without rerunning completed experiments.
- W&B logging comes from `training.report_to: wandb` and `training.wandb_project` in `config.yaml`.
- The main metric currently saved is weighted F1 as `eval_f1`.
- `best_eval_f1` in the summary comes from the best checkpoint recorded in `trainer_state.json`.
