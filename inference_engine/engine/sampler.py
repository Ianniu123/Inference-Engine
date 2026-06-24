from dataclasses import dataclass
from typing import List

import torch

from ..core import Sequence

MIN_TEMP = 1e-6
MIN_TOP_P = 1e-6


@dataclass
class BatchSamplingArgs:
    all_greedy: bool
    greedy: List[bool]
    temperatures: List[float]
    top_ks: List[int]
    top_ps: List[float]


class Sampler:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def prepare(self, seqs: List[Sequence]) -> BatchSamplingArgs:
        params = [seq.sampling_params for seq in seqs]
        greedy = [p.is_greedy for p in params]
        temperatures = [max(p.temperature, MIN_TEMP) for p in params]
        top_ks = [min(p.top_k, self.vocab_size) if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_TOP_P), 1.0) for p in params]
        return BatchSamplingArgs(all(greedy), greedy, temperatures, top_ks, top_ps)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> List[int]:
        if args.all_greedy:
            return torch.argmax(logits, dim=-1).tolist()

        device = logits.device
        temps = torch.tensor(args.temperatures, dtype=torch.float32, device=device)
        ks = torch.tensor(args.top_ks, dtype=torch.long, device=device)
        ps = torch.tensor(args.top_ps, dtype=torch.float32, device=device)
        greedy = torch.tensor(args.greedy, dtype=torch.bool, device=device)

        logits = logits.float() / temps.unsqueeze(1)

        max_k = int(ks.max())
        vals, idx = torch.topk(logits, max_k, dim=-1)
        col = torch.arange(max_k, device=device)
        vals = vals.masked_fill(col.unsqueeze(0) >= ks.unsqueeze(1), float("-inf"))
        probs = torch.softmax(vals, dim=-1)

        # nucleus: drop tokens whose preceding cumulative mass already exceeds top_p
        cumsum = torch.cumsum(probs, dim=-1)
        probs = probs.masked_fill((cumsum - probs) > ps.unsqueeze(1), 0.0)

        choice = torch.multinomial(probs, num_samples=1)
        sampled = idx.gather(1, choice).squeeze(1)
        tokens = torch.where(greedy, torch.argmax(logits, dim=-1), sampled)
        return tokens.tolist()
