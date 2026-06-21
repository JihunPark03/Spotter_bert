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
 
sample = "A bit harder to print than PLA.  Is somewhat stringy unless a high amount of retraction is used.  This adds print time and stress to the extruder motor and filament.\n\nDoes look nice though once the temps and settings get dialed in.\n\nIs a little more pliable than PLA but not as picky as ABS as to how it's printed.  Doesn't have the poisonous smelly odor of ABS either.\n\nI can print this on 3 of my 4 machines.  The 3 that work with high retraction are all not all metal hot ends.  The one that won't print with it is an all metal hot end that clogs with the high retraction rates necessary to stop stringing.  I have tried running this filament on the all metal hot end printer with slow speed (25 mm/s) and small retraction settings.  This does reduce the jams and stringing.  But this printer is capable of running at 150mm/s with PLA.  It does about 100 mm/s with certain high temp plastics, so I"
 
 
pred = classifier(sample)
print(pred)
