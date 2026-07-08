import torch
from transformers import pipeline
from pprint import pprint
from pathlib import Path

# pipe = pipeline(
#     "sentiment-analysis",
#     model="answerdotai/ModernBERT-large",
#     torch_dtype=torch.bfloat16,
# )

# input_text = "He likes a flower"
# results = pipe(input_text)
# pprint(results)

def resolve_model_dir(path):
    model_dir = Path(path)
    if (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists():
        return model_dir

    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda checkpoint: int(checkpoint.name.rsplit("-", 1)[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(f"No model weights or checkpoints found in {model_dir}")


model_dir = resolve_model_dir("/home/jihun/Spotter_bert/testing/training/modernbert-large-fake-review-detector")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model=str(model_dir),
    tokenizer=str(model_dir),
    device=device,
)
 
sample = "These nylon jaw pads fit my bench vise well and grip parts securely without marring them. I like that they're reversible and work for both flat and round stock. The magnets make installation quick, but they're stronger than expected, and now my drill bits and small tools pick up metal debris. It's not a dealbreaker, but it's a bit annoying."
 
 
pred = classifier(sample)
print(pred)
