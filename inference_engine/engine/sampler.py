from dataclasses import dataclass
from typing import List

import torch

from ..core import Sequence

# Floors so temp=0 rows don't divide-by-zero and top_p=0 rows keep at least the top token.
MIN_TEMP = 1e-6
MIN_TOP_P = 1e-6


@dataclass
class BatchSamplingArgs:
    # Per-row sampling config, index-aligned with the logits rows.
    all_greedy: bool
    greedy: List[bool]
    temperatures: List[float]
    top_ks: List[int]
    top_ps: List[float]


class Sampler:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def prepare(self, seqs: List[Sequence]) -> BatchSamplingArgs:
        # Unpack each sequence's params into index-aligned lists and clamp to safe ranges.
        params = [seq.sampling_params for seq in seqs]
        greedy = [p.is_greedy for p in params]
        temperatures = [max(p.temperature, MIN_TEMP) for p in params]
        # top_k < 1 means "disabled" -> full vocab; otherwise cap at vocab so topk is valid.
        top_ks = [min(p.top_k, self.vocab_size) if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_TOP_P), 1.0) for p in params]
        return BatchSamplingArgs(all(greedy), greedy, temperatures, top_ks, top_ps)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> List[int]:
        # logits: [num_seqs, vocab]
        # Fast path: whole batch greedy -> one argmax over the batch, no masking at all.
        if args.all_greedy:
            return torch.argmax(logits, dim=-1).tolist()

        device = logits.device
        temps = torch.tensor(args.temperatures, dtype=torch.float32, device=device)
        ks = torch.tensor(args.top_ks, dtype=torch.long, device=device)
        ps = torch.tensor(args.top_ps, dtype=torch.float32, device=device)
        greedy = torch.tensor(args.greedy, dtype=torch.bool, device=device)

        # Temperature scale (fp32 for stable softmax/multinomial under bf16 serving later).
        logits = logits.float() / temps.unsqueeze(1)  # [N, vocab]

        # top-k: take the largest max_k once, then mask each row down to its own k.
        # topk returns sorted descending -> exactly the order nucleus needs, and we
        # never sort the full 256k vocab.
        max_k = int(ks.max())
        vals, idx = torch.topk(logits, max_k, dim=-1)            # [N, max_k]
        col = torch.arange(max_k, device=device)
        vals = vals.masked_fill(col.unsqueeze(0) >= ks.unsqueeze(1), float("-inf"))

        probs = torch.softmax(vals, dim=-1)                      # [N, max_k]

        # top-p (nucleus): zero tokens whose mass *strictly before* them already exceeds p.
        # The top token of each row always survives (its preceding mass is 0).
        cumsum = torch.cumsum(probs, dim=-1)
        probs = probs.masked_fill((cumsum - probs) > ps.unsqueeze(1), 0.0)

        # One batched sample (probs are unnormalized weights), then map back to vocab ids.
        choice = torch.multinomial(probs, num_samples=1)         # [N, 1]
        sampled = idx.gather(1, choice).squeeze(1)               # [N]

        # Greedy rows: override with argmax of the (scale-invariant) logits.
        greedy_tokens = torch.argmax(logits, dim=-1)
        tokens = torch.where(greedy, greedy_tokens, sampled)
        return tokens.tolist()
