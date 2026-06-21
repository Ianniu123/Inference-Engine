from typing import List

from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, tokenizer: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def encode(self, prompts: List[str]) -> List[List[int]]:
        # Ragged token ids, no padding/tensors — tensorizing is the runner's per-step job.
        return self.tokenizer(prompts)["input_ids"]

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
