#!/usr/bin/env python3
"""Train ModernBERT-large to classify original vs synthetic reviews.

The dataset JSONL contains paired reviews. Each row becomes two training
examples:
  - source_review -> label 0 (original)
  - synthetic_review -> label 1 (synthetic)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
LABEL_NAMES = ["original", "synthetic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ModernBERT-large for Spotter BERT.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--report_to", default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    config.setdefault("model", {})
    config.setdefault("data", {})
    config.setdefault("dora", {})
    config.setdefault("training", {})

    if args.data_path is not None:
        config["data"]["path"] = str(args.data_path)
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["training"]["per_device_eval_batch_size"] = args.eval_batch_size
    if args.seed is not None:
        config["data"]["seed"] = args.seed
    if args.no_bf16:
        config["training"]["bf16"] = False
    if args.report_to is not None:
        config["training"]["report_to"] = args.report_to

    return config


def load_review_pairs(path: Path) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            source_review = (item.get("source_review") or "").strip()
            synthetic_review = (item.get("synthetic_review") or "").strip()
            meta = {
                "source_id": str(item.get("source_id", line_no)),
                "category": str(item.get("category", "")),
                "product_name": str(item.get("product_name", "")),
            }
            if source_review:
                rows.append({"text": source_review, "labels": 0, **meta})
            if synthetic_review:
                rows.append({"text": synthetic_review, "labels": 1, **meta})

    if not rows:
        raise ValueError(f"No usable reviews found in {path}")

    dataset = Dataset.from_list(rows)
    return dataset.cast_column("labels", ClassLabel(names=LABEL_NAMES))


def maybe_subsample(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    if max_samples < len(LABEL_NAMES):
        raise ValueError(f"max_samples must be at least {len(LABEL_NAMES)}")

    rng = random.Random(seed)
    per_label = max_samples // len(LABEL_NAMES)
    remainder = max_samples % len(LABEL_NAMES)
    indices: list[int] = []
    labels = dataset["labels"]
    for label_id in range(len(LABEL_NAMES)):
        label_indices = [idx for idx, label in enumerate(labels) if label == label_id]
        rng.shuffle(label_indices)
        take = per_label + (1 if label_id < remainder else 0)
        indices.extend(label_indices[:take])
    rng.shuffle(indices)
    return dataset.select(indices)


def split_dataset(dataset: Dataset, validation_size: float, test_size: float, seed: int):
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive fractions.")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1.")

    temp_size = validation_size + test_size
    first = dataset.train_test_split(
        test_size=temp_size,
        seed=seed,
        stratify_by_column="labels",
    )
    relative_test_size = test_size / temp_size
    second = first["test"].train_test_split(
        test_size=relative_test_size,
        seed=seed,
        stratify_by_column="labels",
    )
    return first["train"], second["train"], second["test"]


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["text", "source_id", "category", "product_name"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def infer_modernbert_target_modules(model, leaf_names: list[str]) -> list[str]:
    target_modules = []
    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in leaf_names:
            continue
        if leaf_name == "Wo" and ".attn." not in module_name:
            continue
        if module.__class__.__name__ != "Linear":
            continue
        target_modules.append(module_name)

    if not target_modules:
        raise ValueError(
            "Could not infer ModernBERT DoRA target modules. "
            "Set dora.target_modules explicitly in the config."
        )
    return target_modules


def apply_dora_adapter(model, dora_cfg: dict[str, Any]):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "DoRA fine-tuning requires PEFT. Install it with "
            "`pip install -r training/requirements.txt`."
        ) from exc

    target_modules = dora_cfg.get("target_modules")
    if not target_modules:
        leaf_names = dora_cfg.get("target_leaf_modules", ["Wqkv", "Wo"])
        target_modules = infer_modernbert_target_modules(model, list(leaf_names))

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(dora_cfg.get("rank", 16)),
        lora_alpha=int(dora_cfg.get("alpha", 32)),
        lora_dropout=float(dora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        bias=dora_cfg.get("bias", "none"),
        use_dora=True,
        modules_to_save=dora_cfg.get("modules_to_save", ["classifier"]),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, target_modules


def build_training_args(training_cfg: dict[str, Any], output_dir: Path, seed: int) -> TrainingArguments:
    report_to = training_cfg.get("report_to", "tensorboard")
    if isinstance(report_to, str):
        report_to = [] if report_to.lower() in {"none", "false", "off"} else [report_to]

    optim = training_cfg.get("optim")
    if not optim:
        optim = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    bf16 = bool(training_cfg.get("bf16", False)) and torch.cuda.is_available()
    fp16 = bool(training_cfg.get("fp16", False)) and torch.cuda.is_available() and not bf16

    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(training_cfg.get("num_train_epochs", 3)),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.06)),
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 8)),
        "gradient_checkpointing": bool(training_cfg.get("gradient_checkpointing", True)),
        "bf16": bf16,
        "fp16": fp16,
        "optim": optim,
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "save_strategy": training_cfg.get("save_strategy", "epoch"),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(training_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": training_cfg.get("metric_for_best_model", "f1"),
        "greater_is_better": bool(training_cfg.get("greater_is_better", True)),
        "report_to": report_to,
        "run_name": training_cfg.get("run_name", "modernbert-large-spotter"),
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": True,
    }

    eval_key = "eval_strategy"
    if eval_key not in inspect.signature(TrainingArguments.__init__).parameters:
        eval_key = "evaluation_strategy"
    kwargs[eval_key] = training_cfg.get("eval_strategy", "epoch")

    return TrainingArguments(**kwargs)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    dora_cfg = config.get("dora", {})
    training_cfg = config.get("training", {})

    seed = int(data_cfg.get("seed", 42))
    set_seed(seed)

    data_path = resolve_path(data_cfg.get("path", "training/data/deepseek_synthetic_reviews.jsonl"))
    output_dir = resolve_path(training_cfg.get("output_dir", "training/outputs/modernbert-large-fake-review-detector"))
    model_id = model_cfg.get("id", "answerdotai/ModernBERT-large")
    max_length = int(model_cfg.get("max_length", 512))
    max_samples = data_cfg.get("max_samples")
    max_samples = None if max_samples in {None, "", "null"} else int(max_samples)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_review_pairs(data_path)
    dataset = maybe_subsample(dataset, max_samples=max_samples, seed=seed)
    train_ds, val_ds, test_ds = split_dataset(
        dataset,
        validation_size=float(data_cfg.get("validation_size", 0.1)),
        test_size=float(data_cfg.get("test_size", 0.1)),
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_train = tokenize_dataset(train_ds, tokenizer, max_length)
    tokenized_val = tokenize_dataset(val_ds, tokenizer, max_length)
    tokenized_test = tokenize_dataset(test_ds, tokenizer, max_length)

    label2id = {name: idx for idx, name in enumerate(LABEL_NAMES)}
    id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LABEL_NAMES),
        label2id=label2id,
        id2label=id2label,
    )
    model, dora_target_modules = apply_dora_adapter(model, dora_cfg)

    training_args = build_training_args(training_cfg, output_dir, seed)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "dataset_stats.json",
        {
            "data_path": str(data_path),
            "num_rows_jsonl": len(dataset) // 2,
            "num_examples": len(dataset),
            "train_examples": len(train_ds),
            "validation_examples": len(val_ds),
            "test_examples": len(test_ds),
            "labels": LABEL_NAMES,
            "fine_tuning_method": "dora",
            "dora_rank": int(dora_cfg.get("rank", 16)),
            "dora_alpha": int(dora_cfg.get("alpha", 32)),
            "dora_target_modules": dora_target_modules,
        },
    )
    save_json(output_dir / "resolved_config.json", config)

    trainer.train()

    val_metrics = trainer.evaluate(tokenized_val, metric_key_prefix="validation")
    test_metrics = trainer.evaluate(tokenized_test, metric_key_prefix="test")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = {**val_metrics, **test_metrics}
    save_json(output_dir / "final_metrics.json", metrics)

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_checkpoint:
        save_json(output_dir / "best_checkpoint.json", {"path": best_checkpoint})

    config_copy = output_dir / "training_config.yaml"
    shutil.copyfile(args.config, config_copy)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model and metrics to: {output_dir}")


if __name__ == "__main__":
    main()
