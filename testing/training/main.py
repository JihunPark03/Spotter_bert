import json
import math
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import ClassLabel, Dataset, concatenate_datasets
import numpy as np
import os
from sklearn.metrics import f1_score
import yaml

STAR_LORA_DIR = Path(__file__).resolve().parents[1] / "star_lora"
if str(STAR_LORA_DIR) not in sys.path:
    sys.path.append(str(STAR_LORA_DIR))

from dataset_analyzer import analyze_dataset
from rank_policy import compute_dataset_aware_rank
from stability_callback import AdaLoraAllocationCallback, StabilityAwareCallback

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "deepseek_synthetic_reviews.jsonl"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config():
    config_path = Path(os.getenv("TRAIN_CONFIG", DEFAULT_CONFIG_PATH))
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")

    return config, config_path


config, config_path = load_config()
model_config = config.get("model", {})
data_config = config.get("data", {})
fine_tuning_config = config.get("fine_tuning", {})
training_config = config.get("training", {})
model_id = model_config.get("id", "answerdotai/ModernBERT-large")

if training_config.get("wandb_project") and "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = str(training_config["wandb_project"])

DATA_PATH = Path(data_config.get("path", DATA_PATH))
if not DATA_PATH.is_absolute():
    DATA_PATH = Path(__file__).resolve().parents[2] / DATA_PATH

OUTPUT_DIR = Path(
    training_config.get(
        "output_dir",
        Path(__file__).resolve().parent / "modernbert-large-fake-review-detector",
    )
)
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = Path(__file__).resolve().parents[2] / OUTPUT_DIR


def load_review_dataset(path):
    rows = []
    with path.open(encoding="utf-8") as file:
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


def require_peft():
    try:
        from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA/AdaLoRA training requires peft. Install it with "
            "`testing/venv/bin/pip install peft` or install testing/requirements.txt."
        ) from exc

    return AdaLoraConfig, LoraConfig, TaskType, get_peft_model


def get_config_int(section, key, default):
    value = section.get(key, default)
    if value is None:
        return default
    return int(value)


def get_config_float(section, key, default):
    value = section.get(key, default)
    if value is None:
        return default
    return float(value)


def get_config_optional_int(section, key):
    value = section.get(key)
    if value is None:
        return None
    return int(value)


def balanced_subsample(dataset, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset

    label_names = dataset.features["labels"].names
    if max_samples < len(label_names):
        raise ValueError(
            f"data.max_samples must be at least {len(label_names)} for balanced sampling"
        )

    base_per_label = max_samples // len(label_names)
    remainder = max_samples % len(label_names)
    splits = []

    for label_id, _label_name in enumerate(label_names):
        label_dataset = dataset.filter(lambda row, label_id=label_id: row["labels"] == label_id)
        label_budget = base_per_label + (1 if label_id < remainder else 0)
        if label_budget > len(label_dataset):
            raise ValueError(
                f"Requested {label_budget} samples for label {label_id}, "
                f"but only {len(label_dataset)} are available"
            )

        splits.append(label_dataset.shuffle(seed=seed).select(range(label_budget)))

    return concatenate_datasets(splits).shuffle(seed=seed)


def estimate_total_steps(train_dataset, batch_size, gradient_accumulation_steps, epochs, max_steps):
    if max_steps and max_steps > 0:
        return max_steps

    effective_batch_size = max(1, batch_size * gradient_accumulation_steps)
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

    return max(1, int(steps_per_epoch * epochs))


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


def resolve_target_modules(model):
    target_modules = fine_tuning_config.get("target_modules")
    if not target_modules:
        leaf_names = fine_tuning_config.get("modernbert_target_leaf_modules", ["Wqkv", "Wo"])
        target_modules = infer_modernbert_target_modules(model, leaf_names)

    return target_modules


def build_lora_config(
    rank,
    target_modules,
    lora_alpha,
    lora_dropout,
):
    _AdaLoraConfig, LoraConfig, TaskType, _get_peft_model = require_peft()

    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )


def build_adalora_config(
    init_rank,
    target_rank,
    total_step,
    target_modules,
    lora_alpha,
    lora_dropout,
    tinit_fraction,
    tfinal_fraction,
    delta_fraction,
):
    AdaLoraConfig, _LoraConfig, TaskType, _get_peft_model = require_peft()
    tinit = max(0, int(tinit_fraction * total_step))
    tfinal = max(0, int(tfinal_fraction * total_step))
    delta_t = max(1, int(delta_fraction * total_step))

    if tinit + tfinal >= total_step:
        tinit = 0
        tfinal = max(0, total_step - 1)

    return AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=init_rank,
        target_r=target_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=delta_t,
        beta1=0.85,
        beta2=0.85,
        orth_reg_weight=0.5,
        total_step=total_step,
    )


def build_model(num_labels, label2id, id2label, train_dataset, tokenizer, total_step):
    method = fine_tuning_config.get("method", "full")
    valid_methods = {"full", "lora", "adalora", "star_lora"}
    if method not in valid_methods:
        raise ValueError(
            "fine_tuning.method must be one of: "
            + ", ".join(sorted(valid_methods))
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    experiment_info = {
        "config_path": str(config_path),
        "method": method,
        "model_id": model_id,
    }

    if method == "full":
        experiment_info["trainable_type"] = "full_finetuning"
        return model, experiment_info

    _AdaLoraConfig, _LoraConfig, _TaskType, get_peft_model = require_peft()
    target_modules = resolve_target_modules(model)
    base_rank = get_config_int(fine_tuning_config, "base_rank", 8)

    if method == "lora":
        rank = get_config_int(fine_tuning_config, "rank", base_rank)
        lora_alpha = get_config_int(fine_tuning_config, "lora_alpha", rank * 2)
        lora_dropout = get_config_float(fine_tuning_config, "lora_dropout", 0.1)
        lora_config = build_lora_config(
            rank=rank,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        model = get_peft_model(model, lora_config)

        experiment_info.update(
            {
                "trainable_type": "lora",
                "rank": rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules,
            }
        )

        return model, experiment_info

    if method == "adalora":
        rank = get_config_int(fine_tuning_config, "rank", base_rank)
        init_rank = get_config_int(fine_tuning_config, "init_rank", max(rank + 4, 4))
        lora_alpha = get_config_int(fine_tuning_config, "lora_alpha", rank * 2)
        lora_dropout = get_config_float(fine_tuning_config, "lora_dropout", 0.1)
        adalora_config = build_adalora_config(
            init_rank=init_rank,
            target_rank=rank,
            total_step=total_step,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            tinit_fraction=get_config_float(fine_tuning_config, "tinit_fraction", 0.1),
            tfinal_fraction=get_config_float(fine_tuning_config, "tfinal_fraction", 0.5),
            delta_fraction=get_config_float(fine_tuning_config, "delta_fraction", 0.02),
        )
        model = get_peft_model(model, adalora_config)

        experiment_info.update(
            {
                "trainable_type": "adalora",
                "rank": rank,
                "init_rank": init_rank,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "target_modules": target_modules,
                "estimated_total_steps": total_step,
            }
        )

        return model, experiment_info

    stats = analyze_dataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        model=model,
        text_column="text",
        label_column="labels",
    )

    min_rank = get_config_int(fine_tuning_config, "min_rank", 4)
    max_rank = get_config_int(fine_tuning_config, "max_rank", 48)
    rank = compute_dataset_aware_rank(
        stats=stats,
        base_rank=base_rank,
        min_rank=min_rank,
        max_rank=max_rank,
    )
    init_rank = get_config_int(fine_tuning_config, "init_rank", max(rank + 8, rank * 2))
    lora_alpha = get_config_int(fine_tuning_config, "lora_alpha", max(rank * 4, base_rank * 2))

    lora_dropout = fine_tuning_config.get("lora_dropout")
    if lora_dropout is None:
        lora_dropout = 0.05
        if stats.num_samples < 1000:
            lora_dropout += 0.05
        if stats.class_imbalance > 0.2 or stats.token_sparsity > 0.6:
            lora_dropout += 0.05
        lora_dropout = min(lora_dropout, 0.2)
    else:
        lora_dropout = float(lora_dropout)

    adalora_config = build_adalora_config(
        init_rank=init_rank,
        target_rank=rank,
        total_step=total_step,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        tinit_fraction=get_config_float(fine_tuning_config, "tinit_fraction", 0.2),
        tfinal_fraction=get_config_float(fine_tuning_config, "tfinal_fraction", 0.85),
        delta_fraction=get_config_float(fine_tuning_config, "delta_fraction", 0.05),
    )
    model = get_peft_model(model, adalora_config)

    experiment_info.update(
        {
            "trainable_type": "star_lora_dataset_aware_adalora",
            "base_rank": base_rank,
            "rank": rank,
            "init_rank": init_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "dataset_stats": stats.to_dict(),
            "estimated_total_steps": total_step,
        }
    )

    return model, experiment_info


def save_experiment_info(output_dir, experiment_info):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "experiment_info.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(experiment_info, file, indent=2)


def print_trainable_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    ratio = 100 * trainable / total
    print(f"trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(labels, predictions, average="weighted")
    return {"f1": float(score)}


def main():
    raw_dataset = load_review_dataset(DATA_PATH)
    label_names = ["REAL", "FAKE"]
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}
    raw_dataset = raw_dataset.cast_column("labels", ClassLabel(names=label_names))
    max_samples = get_config_optional_int(data_config, "max_samples")
    data_seed = get_config_int(data_config, "seed", 42)
    raw_dataset = balanced_subsample(
        dataset=raw_dataset,
        max_samples=max_samples,
        seed=data_seed,
    )
    raw_dataset = raw_dataset.train_test_split(
        test_size=get_config_float(data_config, "test_size", 0.1),
        seed=data_seed,
        stratify_by_column="labels",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = get_config_int(model_config, "max_length", 512)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

    num_labels = len(label_names)
    max_steps = int(os.getenv("MAX_STEPS", str(get_config_int(training_config, "max_steps", -1))))
    total_step = estimate_total_steps(
        train_dataset=raw_dataset["train"],
        batch_size=get_config_int(training_config, "per_device_train_batch_size", 8),
        gradient_accumulation_steps=get_config_int(training_config, "gradient_accumulation_steps", 4),
        epochs=get_config_int(training_config, "num_train_epochs", 5),
        max_steps=max_steps,
    )
    model, experiment_info = build_model(
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        train_dataset=raw_dataset["train"],
        tokenizer=tokenizer,
        total_step=total_step,
    )
    print_trainable_parameters(model)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=get_config_int(training_config, "per_device_train_batch_size", 8),
        per_device_eval_batch_size=get_config_int(training_config, "per_device_eval_batch_size", 8),
        gradient_accumulation_steps=get_config_int(training_config, "gradient_accumulation_steps", 4),
        learning_rate=get_config_float(training_config, "learning_rate", 5e-5),
        num_train_epochs=get_config_int(training_config, "num_train_epochs", 5),
        max_steps=max_steps,
        bf16=bool(training_config.get("bf16", True)),
        optim=training_config.get("optim", "adamw_torch_fused"),
        logging_strategy="steps",
        logging_steps=get_config_int(training_config, "logging_steps", 100),
        eval_strategy=training_config.get("eval_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_total_limit=get_config_int(training_config, "save_total_limit", 2),
        load_best_model_at_end=bool(training_config.get("load_best_model_at_end", True)),
        metric_for_best_model="f1",
        report_to=training_config.get("report_to", "tensorboard"),
        run_name=training_config.get("run_name"),
        push_to_hub=False,
        hub_strategy="every_save",
    )

    callbacks = []
    method = fine_tuning_config.get("method", "full")
    if method == "star_lora":
        callbacks.append(StabilityAwareCallback(output_dir=str(OUTPUT_DIR)))
    if method in {"adalora", "star_lora"}:
        callbacks.append(AdaLoraAllocationCallback())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    experiment_info["train_metrics"] = train_result.metrics
    experiment_info["eval_metrics"] = eval_metrics
    experiment_info["output_dir"] = str(OUTPUT_DIR)
    experiment_info["num_samples"] = len(tokenized_dataset["train"]) + len(tokenized_dataset["test"])
    experiment_info["train_samples"] = len(tokenized_dataset["train"])
    experiment_info["eval_samples"] = len(tokenized_dataset["test"])
    save_experiment_info(OUTPUT_DIR, experiment_info)


if __name__ == "__main__":
    main()
