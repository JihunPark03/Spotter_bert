from transformers import AutoTokenizer
from datasets import load_dataset
from random import randrange

"""
예시) batch_size = 3의 동작:
batch = {
    "text": [
        "I love coding", 
        "ModernBERT is awesome", 
        "This is a very long sentence for testing"
    ],
    "labels": [2, 5, 12]
}
batch['text'] = ["I want to open a bank account.", "How do I change my PIN code?", "My card is lost, what should I do?"]

토크나이저 출력 결과
토크나이저에서는 embedding하지 않고 우선 token id로 변환한다
{
    'input_ids': tensor([
        [  101,  1045,  2293, 17527,   102,     0,     0,     0],  # 문장 1 (패딩 0 추가)
        [  101,  2711, 23114,  2003, 12476,   102,     0,     0],  # 문장 2 (패딩 0 추가)
        [  101,  2023,  2003,  1037,  2200,  2146,  6251,   102]   # 문장 3 (잘림 현상 발생)
    ]),
    
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 0, 0, 0],  # 문장 1 (진짜 단어 5개, 패딩 3개)
        [1, 1, 1, 1, 1, 1, 0, 0],  # 문장 2 (진짜 단어 6개, 패딩 2개)
        [1, 1, 1, 1, 1, 1, 1, 1]   # 문장 3 (8개 모두 진짜 단어)
    ])
}

"""
dataset_id = "legacy-datasets/banking77"

raw_dataset = load_dataset(dataset_id)

# Model id to load the tokenizer
model_id = "answerdotai/ModernBERT-base"# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.model_max_length = 512 # set model_max_length to 512 as prompts are not longer than 1024 tokens
 
# Tokenize helper function
def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors="pt")

# Tokenize dataset
raw_dataset =  raw_dataset.rename_column("label", "labels") # to match Trainer
tokenized_dataset = raw_dataset.map(tokenize, batched=True,remove_columns=["text"])
# 기본으로 batch_size = 1000개씩 묶어서 처리

print(tokenized_dataset["train"].features.keys())
# dict_keys(['labels', 'input_ids', 'attention_mask'])
