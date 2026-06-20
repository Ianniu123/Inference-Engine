from typing import List

import torch
from transformers import AutoModelForCausalLM

from ..core import Sequence


class ModelRunner:
    def __init__(self, model: str):
        # Use registry and route to a class from the models directory, for now use hf
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.device = next(self.model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def run(self, seqs: List[Sequence], is_prefill: bool) -> torch.Tensor:
        # adhering to huggingface implementation requirements.
        last_logits = []
        for seq in seqs:
            if is_prefill:
                input_ids = torch.tensor([seq.input_ids], dtype=torch.long, device=self.device)
                output = self.model(input_ids=input_ids, use_cache=True)
            else:
                input_ids = torch.tensor([[seq.input_ids[-1]]], dtype=torch.long, device=self.device)
                output = self.model(input_ids=input_ids, past_key_values=seq.kv, use_cache=True)
            seq.kv = output.past_key_values
            last_logits.append(output.logits[0, -1, :])

        # [num_seqs, vocab] — one row of next-token logits per sequence.
        return torch.stack(last_logits, dim=0)
