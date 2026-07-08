import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dora_svd_pruning import (
    DoraSVDPruningCallback,
    SVDPruningConfig,
    estimate_effective_direction_params,
    prune_lora_direction_with_svd,
    write_pruning_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / "training" / "data" / "deepseek_synthetic_reviews.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LoRA, DoRA, or DoRA with periodic SVD pruning."
    )
    parser.add_argument(
        "--method",
        choices=["lora", "dora", "dora_svd_pruning"],
        default="dora",
    )
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-large")
    parser.add_argument(
        "--dataset_key",
        choices=["local_review", "sst2", "imdb"],
        default="local_review",
        help="Dataset preset. local_review keeps the original repo-local fake-review setup.",
    )
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument(
        "--output_dir",
        default="testing_compression_with_dora_svd/results/debug",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--label_noise_rate",
        type=float,
        default=0.0,
        help="Fraction of training labels to flip. Eval labels are always kept clean.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=None)
    parser.add_argument(
        "--modernbert_target_leaf_modules",
        nargs="+",
        default=["Wqkv", "Wo"],
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", default="epoch")
    parser.add_argument("--save_strategy", default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--svd_pruning_start_step", type=int, default=100)
    parser.add_argument(
        "--svd_pruning_start_ratio",
        type=float,
        default=None,
        help="Optional fraction of total training steps where periodic pruning starts.",
    )
    parser.add_argument("--svd_pruning_interval", type=int, default=100)
    parser.add_argument("--svd_pruning_threshold", type=float, default=0.0)
    parser.add_argument("--svd_pruning_keep_ratio", type=float, default=1.0)
    parser.add_argument(
        "--svd_pruning_energy_ratio",
        type=float,
        default=None,
        help="Keep the smallest rank whose squared singular-value energy reaches this ratio.",
    )
    parser.add_argument(
        "--svd_pruning_mode",
        choices=["periodic", "post_training", "none"],
        default="periodic",
        help="periodic prunes during training; post_training prunes once after DoRA training.",
    )
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--max_rank", type=int, default=None)
    return parser.parse_args()


def load_review_dataset(path):
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            source_review = row.get("source_review")
            synthetic_review = row.get("synthetic_review")
            if source_review:
                rows.append({"text": source_review, "labels": 0})
            if synthetic_review:
                rows.append({"text": synthetic_review, "labels": 1})

    if not rows:
        raise ValueError(f"No usable review rows found in {path}")

    return Dataset.from_list(rows)


def load_public_dataset(dataset_key):
    if dataset_key == "sst2":
        dataset = load_dataset("glue", "sst2")
        return {
            "train": dataset["train"].rename_column("sentence", "text"),
            "eval": dataset["validation"].rename_column("sentence", "text"),
            "label_names": ["negative", "positive"],
            "dataset_name": "glue",
            "dataset_config": "sst2",
        }

    if dataset_key == "imdb":
        dataset = load_dataset("imdb")
        return {
            "train": dataset["train"],
            "eval": dataset["test"],
            "label_names": ["negative", "positive"],
            "dataset_name": "imdb",
            "dataset_config": None,
        }

    raise ValueError(f"Unsupported public dataset: {dataset_key}")


def load_experiment_dataset(args):
    if args.dataset_key == "local_review":
        label_names = ["REAL", "FAKE"]
        raw_dataset = load_review_dataset(args.data_path)
        raw_dataset = raw_dataset.cast_column("labels", ClassLabel(names=label_names))
        raw_dataset = balanced_subsample(raw_dataset, args.max_samples, args.seed)
        split_dataset = raw_dataset.train_test_split(
            test_size=args.test_size,
            seed=args.seed,
            stratify_by_column="labels",
        )
        return {
            "train": split_dataset["train"],
            "eval": split_dataset["test"],
            "label_names": label_names,
            "dataset_name": "local_review",
            "dataset_config": None,
        }

    dataset = load_public_dataset(args.dataset_key)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("label", "labels")
    if "label" in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column("label", "labels")

    train_dataset = balanced_subsample(train_dataset, args.max_samples, args.seed)
    train_dataset = apply_label_noise(
        train_dataset,
        label_column="labels",
        noise_rate=args.label_noise_rate,
        num_labels=len(dataset["label_names"]),
        seed=args.seed,
    )

    return {
        "train": train_dataset,
        "eval": eval_dataset,
        "label_names": dataset["label_names"],
        "dataset_name": dataset["dataset_name"],
        "dataset_config": dataset["dataset_config"],
    }


def balanced_subsample(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset.shuffle(seed=seed)

    labels = sorted(set(int(label) for label in dataset["labels"]))
    if max_samples < len(labels):
        raise ValueError(f"max_samples must be at least {len(labels)}")

    base_per_label = max_samples // len(labels)
    remainder = max_samples % len(labels)
    splits = []

    for offset, label_id in enumerate(labels):
        label_dataset = dataset.filter(lambda row, label_id=label_id: int(row["labels"]) == label_id)
        label_budget = base_per_label + (1 if offset < remainder else 0)
        if label_budget > len(label_dataset):
            raise ValueError(
                f"Requested {label_budget} samples for label {label_id}, "
                f"but only {len(label_dataset)} are available"
            )
        splits.append(label_dataset.shuffle(seed=seed).select(range(label_budget)))

    return concatenate_datasets(splits).shuffle(seed=seed)


def apply_label_noise(dataset, label_column, noise_rate, num_labels, seed):
    if noise_rate <= 0:
        return dataset

    if not 0 <= noise_rate < 1:
        raise ValueError("--label_noise_rate must be in [0, 1).")

    rng = np.random.default_rng(seed)
    noisy_indices = set(
        rng.choice(
            len(dataset),
            size=int(round(len(dataset) * noise_rate)),
            replace=False,
        ).tolist()
    )

    def flip_label(row, index):
        if index not in noisy_indices:
            row["is_noisy_label"] = False
            return row

        original_label = int(row[label_column])
        if num_labels == 2:
            row[label_column] = 1 - original_label
        else:
            choices = [label for label in range(num_labels) if label != original_label]
            row[label_column] = int(rng.choice(choices))
        row["is_noisy_label"] = True
        return row

    return dataset.map(flip_label, with_indices=True)


def infer_modernbert_target_modules(model, leaf_names):
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
        return list(leaf_names)

    return target_modules


def build_lora_config(args, target_modules, use_dora):
    try:
        return LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_dora=use_dora,
        )
    except TypeError as exc:
        raise TypeError(
            "This experiment needs a PEFT version whose LoraConfig supports "
            "`use_dora=True`. Upgrade peft in .venv if this fails."
        ) from exc


def trainable_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def total_parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return None

    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted")),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_experiment_dataset(args)
    label_names = dataset_bundle["label_names"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_remove_columns = [
        column for column in dataset_bundle["train"].column_names if column != "labels"
    ]
    eval_remove_columns = [
        column for column in dataset_bundle["eval"].column_names if column != "labels"
    ]
    tokenized_train = dataset_bundle["train"].map(
        tokenize,
        batched=True,
        remove_columns=train_remove_columns,
    )
    tokenized_eval = dataset_bundle["eval"].map(
        tokenize,
        batched=True,
        remove_columns=eval_remove_columns,
    )

    label2id = {label: index for index, label in enumerate(label_names)}
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )

    target_modules = args.target_modules or infer_modernbert_target_modules(
        model,
        args.modernbert_target_leaf_modules,
    )
    peft_config = build_lora_config(
        args,
        target_modules,
        use_dora=args.method in {"dora", "dora_svd_pruning"},
    )
    model = get_peft_model(model, peft_config)

    pruning_config = SVDPruningConfig(
        start_step=args.svd_pruning_start_step,
        start_ratio=args.svd_pruning_start_ratio,
        interval=args.svd_pruning_interval,
        threshold=args.svd_pruning_threshold,
        keep_ratio=args.svd_pruning_keep_ratio,
        energy_ratio=args.svd_pruning_energy_ratio,
        mode=args.svd_pruning_mode,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
    )
    callbacks = []
    if args.method == "dora_svd_pruning" and args.svd_pruning_mode == "periodic":
        callbacks.append(DoraSVDPruningCallback(pruning_config, output_dir))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        optim=args.optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.save_strategy == args.eval_strategy,
        metric_for_best_model="f1",
        report_to=args.report_to,
        run_name=args.run_name,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    start_time = time.perf_counter()
    train_result = trainer.train()
    training_time_seconds = time.perf_counter() - start_time
    eval_metrics = trainer.evaluate()

    pre_pruning_eval_metrics = None
    if args.method == "dora_svd_pruning" and args.svd_pruning_mode == "post_training":
        # Post-training SVD pruning leaves DoRA training untouched, then compresses
        # only the LoRA direction update Delta_W = B @ A before final evaluation.
        pre_pruning_eval_metrics = dict(eval_metrics)
        records = prune_lora_direction_with_svd(model, pruning_config)
        for record in records:
            record.step = int(trainer.state.global_step)
        write_pruning_records(output_dir, records)
        eval_metrics = trainer.evaluate()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    effective_direction_params, effective_ranks = estimate_effective_direction_params(model)
    trainable_params = trainable_parameter_count(model)
    total_params = total_parameter_count(model)

    experiment_info = {
        "method": args.method,
        "dataset_key": args.dataset_key,
        "dataset_name": dataset_bundle["dataset_name"],
        "dataset_config": dataset_bundle["dataset_config"],
        "model_name": args.model_name,
        "data_path": str(Path(args.data_path).resolve()),
        "max_samples": args.max_samples,
        "label_noise_rate": args.label_noise_rate,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "svd_pruning": pruning_config.__dict__,
        "pre_pruning_eval_metrics": pre_pruning_eval_metrics,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "effective_direction_params": effective_direction_params,
        "effective_ranks": effective_ranks,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "training_time_seconds": round(training_time_seconds, 3),
        "gpu_max_memory_mb": gpu_memory_mb(),
        "train_samples": len(tokenized_train),
        "eval_samples": len(tokenized_eval),
        "seed": args.seed,
        "output_dir": str(output_dir),
    }
    save_json(output_dir / "experiment_info.json", experiment_info)

    print(json.dumps(experiment_info, indent=2))


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent))
    main()
