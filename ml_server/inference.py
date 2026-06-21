import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.getenv("AD_DETECTOR_MODEL", "answerdotai/ModernBERT-large")
MAX_LENGTH = int(os.getenv("AD_DETECTOR_MAX_LENGTH", "512"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
tokenizer = None
current_model_id = None


def load_model():
    global model, tokenizer, current_model_id

    if model is not None and tokenizer is not None and current_model_id == MODEL_ID:
        print(f"[ML] Using current model: {MODEL_ID}")
        return

    print(f"[ML] Loading model: {MODEL_ID}")

    new_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    new_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    new_model.eval()

    tokenizer = new_tokenizer
    model = new_model
    current_model_id = MODEL_ID


def _ad_label_index() -> int:
    id2label = getattr(model.config, "id2label", {}) or {}
    for idx, label in id2label.items():
        if "ad" in str(label).lower() or "spam" in str(label).lower():
            return int(idx)
    return 1 if model.config.num_labels > 1 else 0


@torch.no_grad()
def predict_prob(text: str) -> float:
    if model is None or tokenizer is None:
        load_model()

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    logits = model(**inputs).logits.squeeze(0)

    if model.config.num_labels == 1:
        prob = torch.sigmoid(logits).item()
    else:
        prob = torch.softmax(logits, dim=-1)[_ad_label_index()].item()

    return prob
