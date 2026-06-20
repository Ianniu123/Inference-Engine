from transformers import AutoTokenizer
from typing import List
class Tokenizer:
    def __init__(self, tokenizer: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def encode(self, prompts: List[str]) -> List[List[int]]:
        return self.tokenizer(prompts, return_tensors='pt', padding=True)