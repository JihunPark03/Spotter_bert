import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "sentiment-analysis",
    model="answerdotai/ModernBERT-large",
    torch_dtype=torch.bfloat16,
)

input_text = "He likes a flower"
results = pipe(input_text)
pprint(results)
