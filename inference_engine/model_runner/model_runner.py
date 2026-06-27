# HuggingFace baseline runner, one sequence per forward step. Not used for serving (the
# engine serves the from-scratch model via PagedModelRunner); this is the benchmark baseline.

from typing import List

import torch
from transformers import AutoModelForCausalLM

from ..core import Sequence


class ModelRunner:
    def __init__(self, model: str):
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.device = next(self.model.parameters()).device
        self.vocab_size = self.model.config.vocab_size
        self.model.eval()

    @torch.no_grad()
    def run(self, seqs: List[Sequence], is_prefill: bool) -> torch.Tensor:
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
        return torch.stack(last_logits, dim=0)

    def free(self, seq: Sequence) -> None:
        seq.kv = None
