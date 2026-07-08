# ModernBERT-large DoRA Training

This directory fine-tunes Spotter BERT with DoRA adapters on `training/data/deepseek_synthetic_reviews.jsonl`.

This training pipeline is for **English product reviews only**. The dataset labels English `source_review` text as original and English `synthetic_review` text as synthetic.

Each JSONL row is converted into two binary-classification examples:

- `source_review` -> `original` / label `0`
- `synthetic_review` -> `synthetic` / label `1`

## Setup

```bash
cd /home/jihun/Spotter_bert
python3 -m venv .venv
.venv/bin/pip install -r training/requirements.txt
```

If the existing `.venv` shows invalid package metadata warnings, rebuild it:

```bash
rm -rf .venv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r training/requirements.txt
```

## Train

```bash
.venv/bin/python training/train_modernbert.py --config training/config.yaml
```

For a quick smoke test:

```bash
.venv/bin/python training/train_modernbert.py \
  --config training/config.yaml \
  --max_samples 1000 \
  --epochs 1 \
  --output_dir training/outputs/smoke-modernbert-large
```

Run in the background:

```bash
nohup .venv/bin/python training/train_modernbert.py \
  --config training/config.yaml \
  > training/modernbert-large.log 2>&1 &
```

Check progress:

```bash
tail -f training/modernbert-large.log
```

## Outputs

Default output directory:

```text
training/outputs/modernbert-large-dora-fake-review-detector
```

Important files:

- `final_metrics.json`
- `dataset_stats.json`
- `resolved_config.json`
- saved DoRA adapter, classifier head, and tokenizer

The saved adapter can be used with the DoRA-SVD compression scripts because the base ModernBERT weights stay frozen and the learned update is stored in DoRA/LoRA adapter factors.

## Notes

The default config freezes `answerdotai/ModernBERT-large` and trains DoRA adapters with rank 16, alpha 32, bf16, and gradient checkpointing. If CUDA memory is tight, lower `per_device_train_batch_size` in `training/config.yaml`.
