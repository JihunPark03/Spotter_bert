import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import ClassLabel, Dataset
import numpy as np
import os
from sklearn.metrics import f1_score

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "deepseek_synthetic_reviews.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "modernbert-large-fake-review-detector"


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


raw_dataset = load_review_dataset(DATA_PATH)
label_names = ["REAL", "FAKE"]
label2id = {label: i for i, label in enumerate(label_names)}
id2label = {i: label for label, i in label2id.items()}
raw_dataset = raw_dataset.cast_column("labels", ClassLabel(names=label_names))
raw_dataset = raw_dataset.train_test_split(
    test_size=0.1,
    seed=42,
    stratify_by_column="labels",
)

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
 
# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# Tokenize dataset
tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])
# 기본으로 batch_size = 1000개씩 묶어서 처리
 
# Prepare model labels - useful for inference
num_labels = len(label_names)
 
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=num_labels, label2id=label2id, id2label=id2label,
)

# Metric helper method
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(labels, predictions, average="weighted")
    return {"f1": float(score)}
 
# Define training args
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=5,
    max_steps=int(os.getenv("MAX_STEPS", "-1")),
    bf16=True, # bfloat16 training 
    optim="adamw_torch_fused", # improved optimizer 
    # logging & evaluation strategies
    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    # push to hub parameters
    report_to="tensorboard",
    push_to_hub=False,
    hub_strategy="every_save",
 
)
 
# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)
