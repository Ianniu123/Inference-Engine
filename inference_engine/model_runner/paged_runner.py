# Serving runner: packs the batch into one flat forward over the from-scratch model
# (prefill concatenates prompts, decode takes one token per sequence) over the paged pool.

from __future__ import annotations

from typing import List

import torch

from ..core import Sequence
from ..kv_cache import BlockManager, CacheBatch, KVCache, OutOfBlocks
from ..models.loader import load_gemma_from_hf


class PagedModelRunner:
    def __init__(
        self,
        model,
        *,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        num_blocks: int = 2048,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        self.device = torch.device(device) if device else next(model.parameters()).device
        self.model = model.eval()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.blocks = BlockManager(num_blocks, block_size)
        self.cache = KVCache(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, self.device)

    @classmethod
    def from_hf(cls, model_name: str, **kwargs) -> "PagedModelRunner":
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(model_name)
        cfg = hf.config
        return cls(
            load_gemma_from_hf(hf),
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=cfg.hidden_size // cfg.num_attention_heads,
            vocab_size=cfg.vocab_size,
            **kwargs,
        )

    @torch.no_grad()
    def run(self, seqs: List[Sequence], is_prefill: bool) -> torch.Tensor:
        self._check_capacity(seqs, is_prefill)
        return self._prefill(seqs) if is_prefill else self._decode(seqs)

    def blocks_needed(self, seqs: List[Sequence], is_prefill: bool) -> int:
        need = 0
        for seq in seqs:
            total = seq.num_tokens if is_prefill else seq.num_cached_tokens + 1
            need += max(0, self.blocks.blocks_for(total) - len(seq.block_table))
        return need

    def has_capacity(self, seqs: List[Sequence], is_prefill: bool) -> bool:
        return self.blocks_needed(seqs, is_prefill) <= len(self.blocks.free)

    def _check_capacity(self, seqs: List[Sequence], is_prefill: bool) -> None:
        # Before any allocation/KV write, so the engine can preempt and retry from clean state.
        if not self.has_capacity(seqs, is_prefill):
            raise OutOfBlocks(f"batch needs {self.blocks_needed(seqs, is_prefill)} blocks, "
                              f"{len(self.blocks.free)} free")

    def _prefill(self, seqs: List[Sequence]) -> torch.Tensor:
        input_ids, positions, slots, seq_lens = [], [], [], []
        for seq in seqs:
            n = seq.num_tokens
            self._ensure_blocks(seq, n)
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
            self._ensure_blocks(seq, p + 1)
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

    def _ensure_blocks(self, seq: Sequence, total_tokens: int) -> None:
        need = self.blocks.blocks_for(total_tokens) - len(seq.block_table)
        if need > 0:
            seq.block_table.extend(self.blocks.allocate(need))

    def _pad(self, tables: list[list[int]]) -> torch.Tensor:
        width = max(len(t) for t in tables)
        return self._t([t + [0] * (width - len(t)) for t in tables])

    def _t(self, data) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.long, device=self.device)

    def free(self, seq: Sequence) -> None:
        self.blocks.free_blocks(seq.block_table)
        seq.block_table = []
        seq.num_cached_tokens = 0
