from datasets import load_dataset
from random import randrange

dataset_id = "legacy-datasets/banking77"

raw_dataset = load_dataset(dataset_id)

random_id = randrange(len(raw_dataset['train']))

# print(f"Train dataset size: {len(raw_dataset['train'])}")
# print(f"Test dataset size: {len(raw_dataset['test'])}")
print(raw_dataset['train'][random_id])
