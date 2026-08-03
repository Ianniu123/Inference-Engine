# Serving runner: packs the batch into one flat forward over the from-scratch model
# (prefill concatenates prompts, decode takes one token per sequence) over the paged pool.
# Pure executor: block tables are allocated by the scheduler, which owns the BlockManager.

from __future__ import annotations

from typing import List

import torch

from ..core import Sequence
from ..kv_cache import BlockManager, CacheBatch, KVCache
from ..models.loader import load_gemma_from_hf


class ModelRunner:
    def __init__(
        self,
        model,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        num_blocks: int = 2048,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.blocks = BlockManager(num_blocks, block_size)
        self.cache = KVCache(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, self.device)

    @classmethod
    def from_hf(cls, model_name: str, *, device=None, dtype=None, **kwargs) -> "ModelRunner":
        from transformers import AutoModelForCausalLM

        device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float16

        hf = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16)
        cfg = hf.config
        return cls(
            load_gemma_from_hf(hf),
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            vocab_size=cfg.vocab_size,
            device=device,
            dtype=dtype,
            **kwargs,
        )

    @torch.no_grad()
    def run(self, seqs: List[Sequence], is_prefill: bool) -> torch.Tensor:
        return self._prefill(seqs) if is_prefill else self._decode(seqs)

    def _prefill(self, seqs: List[Sequence]) -> torch.Tensor:
        input_ids, positions, slots, seq_lens = [], [], [], []
        for seq in seqs:
            n = seq.num_tokens
            input_ids += seq.input_ids
            positions += range(n)
            slots += self._slots(seq, 0, n)
            seq_lens.append(n)
            seq.num_cached_tokens = n

        ctx = CacheBatch(is_prefill=True, cache=self.cache, slot_mapping=self._t(slots), seq_lens=seq_lens)
        logits = self.model(self._t(input_ids), self._t(positions), ctx)
        last = torch.tensor(seq_lens, device=self.device).cumsum(0) - 1
        return logits[last]

    def _decode(self, seqs: List[Sequence]) -> torch.Tensor:
        input_ids, positions, slots, context_lens, tables = [], [], [], [], []
        for seq in seqs:
            p = seq.num_cached_tokens
            input_ids.append(seq.last_token)
            positions.append(p)
            slots += self._slots(seq, p, p + 1)
            context_lens.append(p + 1)
            tables.append(seq.block_table)
            seq.num_cached_tokens = p + 1

        ctx = CacheBatch(
            is_prefill=False,
            cache=self.cache,
            slot_mapping=self._t(slots),
            block_tables=self._pad(tables),
            context_lens=self._t(context_lens),
        )
        return self.model(self._t(input_ids), self._t(positions), ctx)

    def _slots(self, seq: Sequence, start: int, end: int) -> list[int]:
        bs = self.block_size
        return [seq.block_table[p // bs] * bs + p % bs for p in range(start, end)]

    def _pad(self, tables: list[list[int]]) -> torch.Tensor:
        width = max(len(t) for t in tables)
        return self._t([t + [0] * (width - len(t)) for t in tables])

    def _t(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.long, device=self.device)
