from __future__ import annotations

from dataclasses import dataclass

import torch


class OutOfBlocks(RuntimeError):
    pass


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.free: list[int] = list(range(num_blocks))

    def blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks > len(self.free):
            raise OutOfBlocks(f"need {num_blocks} blocks, {len(self.free)} free")
        blocks, self.free = self.free[:num_blocks], self.free[num_blocks:]
        return blocks

    def free_blocks(self, blocks: list[int]) -> None:
        self.free.extend(blocks)


class KVCache:
    # Per-layer K/V pools flattened over (blocks, block_size) into one slot dimension,
    # so slot_mapping / block_tables index straight in.
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device):
        self.block_size = block_size
        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)


@dataclass
class CacheBatch:
    is_prefill: bool
    cache: KVCache | None = None
    slot_mapping: torch.Tensor | None = None
    seq_lens: list[int] | None = None             # prefill
    cu_seqlens: torch.Tensor | None = None        # prefill
    block_tables: torch.Tensor | None = None      # decode
    context_lens: torch.Tensor | None = None      # decode
