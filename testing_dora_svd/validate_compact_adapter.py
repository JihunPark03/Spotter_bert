import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from train_dora_svd import compute_metrics, load_experiment_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate compact SVD adapters by zero-padding them back to the "
            "PEFT rank used during training and re-running evaluation."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Experiment directories or a results root containing experiment directories.",
    )
    parser.add_argument("--compact_name", default="compact_svd_adapter")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional quick-check limit. Omit for full validation.",
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--evaluate_full_adapter",
        action="store_true",
        help="Also re-evaluate the original full-shape adapter in the same process.",
    )
    return parser.parse_args()


def find_experiment_dirs(paths, compact_name):
    experiment_dirs = []
    for raw_path in paths:
        path = Path(raw_path)
        if (path / compact_name / "adapter_model.safetensors").exists():
            experiment_dirs.append(path)
            continue

        if not path.exists():
            raise FileNotFoundError(path)

        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / compact_name / "adapter_model.safetensors").exists():
                experiment_dirs.append(child)

    return experiment_dirs


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def make_dataset_args(info):
    class DatasetArgs:
        pass

    args = DatasetArgs()
    args.dataset_key = info["dataset_key"]
    args.data_path = info.get("data_path")
    args.max_samples = info.get("max_samples")
    args.test_size = 0.1
    args.label_noise_rate = info.get("label_noise_rate", 0.0)
    args.seed = info.get("seed", 42)
    return args


def zero_pad_compact_state(compact_state, full_state):
    padded_state = {}

    for key, compact_tensor in compact_state.items():
        if key not in full_state:
            raise KeyError(f"{key} is missing from the original full-rank adapter.")

        full_tensor = full_state[key]
        if compact_tensor.shape == full_tensor.shape:
            padded_state[key] = compact_tensor
            continue

        if key.endswith(".lora_A.weight"):
            padded = torch.zeros_like(full_tensor)
            kept_rank = compact_tensor.shape[0]
            padded[:kept_rank, :] = compact_tensor
            padded_state[key] = padded
            continue

        if key.endswith(".lora_B.weight"):
            padded = torch.zeros_like(full_tensor)
            kept_rank = compact_tensor.shape[1]
            padded[:, :kept_rank] = compact_tensor
            padded_state[key] = padded
            continue

        raise ValueError(
            f"Only LoRA A/B tensors may have compact shapes, but {key} has "
            f"compact shape {tuple(compact_tensor.shape)} and full shape "
            f"{tuple(full_tensor.shape)}."
        )

    return padded_state


def max_state_abs_diff(left_state, right_state):
    max_diff = 0.0
    worst_key = None
    for key, left_tensor in left_state.items():
        if key not in right_state:
            raise KeyError(f"{key} is missing from the right-hand state dict.")
        diff = (left_tensor - right_state[key]).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst_key = key
    return max_diff, worst_key


def build_eval_model(experiment_dir, info, label_names, padded_state):
    label2id = {label: index for index, label in enumerate(label_names)}
    id2label = {index: label for label, index in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        info["model_name"],
        num_labels=len(label_names),
        label2id=label2id,
        id2label=id2label,
    )
    peft_config = LoraConfig.from_pretrained(str(experiment_dir))
    model = get_peft_model(model, peft_config)
    load_result = set_peft_model_state_dict(
        model,
        padded_state,
        adapter_name="default",
    )
    return model, load_result


def evaluate_state(experiment_dir, info, label_names, tokenized_eval, state, args):
    model, load_result = build_eval_model(experiment_dir, info, label_names, state)

    with tempfile.TemporaryDirectory(prefix="compact_adapter_eval_") as tmpdir:
        training_args = TrainingArguments(
            output_dir=tmpdir,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            bf16=args.bf16,
            fp16=args.fp16,
            report_to="none",
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_eval,
            compute_metrics=compute_metrics,
        )
        eval_metrics = trainer.evaluate()
    return eval_metrics, load_result

def evaluate_experiment(experiment_dir, args):
    info_path = experiment_dir / "experiment_info.json"
    full_adapter_path = experiment_dir / "adapter_model.safetensors"
    compact_dir = experiment_dir / args.compact_name
    compact_adapter_path = compact_dir / "adapter_model.safetensors"

    info = load_json(info_path)
    set_seed(info.get("seed", 42))

    dataset_args = make_dataset_args(info)
    dataset_bundle = load_experiment_dataset(dataset_args)
    eval_dataset = dataset_bundle["eval"]
    if args.max_eval_samples is not None and args.max_eval_samples < len(eval_dataset):
        eval_dataset = eval_dataset.shuffle(seed=info.get("seed", 42)).select(
            range(args.max_eval_samples)
        )

    tokenizer = AutoTokenizer.from_pretrained(str(experiment_dir))
    tokenizer.model_max_length = args.max_length

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    eval_remove_columns = [column for column in eval_dataset.column_names if column != "labels"]
    tokenized_eval = eval_dataset.map(
        tokenize,
        batched=True,
        remove_columns=eval_remove_columns,
    )

    full_state = load_file(str(full_adapter_path))
    compact_state = load_file(str(compact_adapter_path))
    padded_state = zero_pad_compact_state(compact_state, full_state)
    max_reconstruction_abs_diff, worst_reconstruction_key = max_state_abs_diff(
        padded_state,
        full_state,
    )
    compact_eval_metrics, load_result = evaluate_state(
        experiment_dir,
        info,
        dataset_bundle["label_names"],
        tokenized_eval,
        padded_state,
        args,
    )

    full_eval_metrics = None
    if args.evaluate_full_adapter:
        full_eval_metrics, _ = evaluate_state(
            experiment_dir,
            info,
            dataset_bundle["label_names"],
            tokenized_eval,
            full_state,
            args,
        )

    original_eval_metrics = info.get("eval_metrics", {})

    result = {
        "experiment": experiment_dir.name,
        "dataset_key": info.get("dataset_key"),
        "max_samples": info.get("max_samples"),
        "label_noise_rate": info.get("label_noise_rate"),
        "seed": info.get("seed"),
        "method": info.get("method"),
        "rank": info.get("rank"),
        "energy_ratio": (info.get("svd_pruning") or {}).get("energy_ratio"),
        "eval_samples": len(tokenized_eval),
        "original_eval_loss": original_eval_metrics.get("eval_loss"),
        "original_accuracy": original_eval_metrics.get("eval_accuracy"),
        "original_f1": original_eval_metrics.get("eval_f1"),
        "compact_eval_loss": compact_eval_metrics.get("eval_loss"),
        "compact_accuracy": compact_eval_metrics.get("eval_accuracy"),
        "compact_f1": compact_eval_metrics.get("eval_f1"),
        "reloaded_full_eval_loss": (
            full_eval_metrics.get("eval_loss") if full_eval_metrics else None
        ),
        "reloaded_full_accuracy": (
            full_eval_metrics.get("eval_accuracy") if full_eval_metrics else None
        ),
        "reloaded_full_f1": (
            full_eval_metrics.get("eval_f1") if full_eval_metrics else None
        ),
        "loss_diff": (
            compact_eval_metrics.get("eval_loss") - original_eval_metrics.get("eval_loss")
            if original_eval_metrics.get("eval_loss") is not None
            else None
        ),
        "accuracy_diff": (
            compact_eval_metrics.get("eval_accuracy")
            - original_eval_metrics.get("eval_accuracy")
            if original_eval_metrics.get("eval_accuracy") is not None
            else None
        ),
        "f1_diff": (
            compact_eval_metrics.get("eval_f1") - original_eval_metrics.get("eval_f1")
            if original_eval_metrics.get("eval_f1") is not None
            else None
        ),
        "compact_vs_reloaded_full_loss_diff": (
            compact_eval_metrics.get("eval_loss") - full_eval_metrics.get("eval_loss")
            if full_eval_metrics
            else None
        ),
        "compact_vs_reloaded_full_accuracy_diff": (
            compact_eval_metrics.get("eval_accuracy")
            - full_eval_metrics.get("eval_accuracy")
            if full_eval_metrics
            else None
        ),
        "compact_vs_reloaded_full_f1_diff": (
            compact_eval_metrics.get("eval_f1") - full_eval_metrics.get("eval_f1")
            if full_eval_metrics
            else None
        ),
        "max_reconstruction_abs_diff": max_reconstruction_abs_diff,
        "worst_reconstruction_key": worst_reconstruction_key,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "compact_adapter": str(compact_adapter_path),
    }
    save_json(compact_dir / "compact_validation.json", result)
    return result


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "experiment",
        "dataset_key",
        "max_samples",
        "label_noise_rate",
        "seed",
        "method",
        "rank",
        "energy_ratio",
        "eval_samples",
        "original_eval_loss",
        "original_accuracy",
        "original_f1",
        "compact_eval_loss",
        "compact_accuracy",
        "compact_f1",
        "reloaded_full_eval_loss",
        "reloaded_full_accuracy",
        "reloaded_full_f1",
        "loss_diff",
        "accuracy_diff",
        "f1_diff",
        "compact_vs_reloaded_full_loss_diff",
        "compact_vs_reloaded_full_accuracy_diff",
        "compact_vs_reloaded_full_f1_diff",
        "max_reconstruction_abs_diff",
        "worst_reconstruction_key",
        "compact_adapter",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"Wrote {path}")


def main():
    args = parse_args()
    experiment_dirs = find_experiment_dirs(args.paths, args.compact_name)
    if args.limit is not None:
        experiment_dirs = experiment_dirs[: args.limit]

    if not experiment_dirs:
        raise ValueError("No compact adapters found.")

    rows = []
    for experiment_dir in experiment_dirs:
        print(f"Validating {experiment_dir}")
        row = evaluate_experiment(experiment_dir, args)
        rows.append(row)
        print(
            f"{row['experiment']}: original_f1={row['original_f1']}, "
            f"compact_f1={row['compact_f1']}, f1_diff={row['f1_diff']}"
        )

    output_csv = (
        Path(args.output_csv)
        if args.output_csv
        else Path(args.paths[0]) / "compact_validation_results.csv"
    )
    write_csv(output_csv, rows)


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent))
    main()
