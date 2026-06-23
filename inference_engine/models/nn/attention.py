from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

from .linear import Linear
from .ops import scaled_dot_product_attention


class RotaryPositionalEmbedding(nn.Module):
    # Interleaved (GPT-NeoX) RoPE over [T, num_heads, head_dim]; the loader permutes HF's
    # rotate_half layout to match this.
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions].unsqueeze(1)
        sin = self.sin_cached[positions].unsqueeze(1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rearrange(rotated, "... d two -> ... (d two)")


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=dim)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int | None = None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.head_dim, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, self.num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, self.num_kv_heads * self.head_dim, device=device, dtype=dtype)
        self.output_proj = Linear(num_heads * self.head_dim, d_model, device=device, dtype=dtype)

    def forward(self, x, rope, positions, ctx, layer_idx: int) -> torch.Tensor:
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        q, k = rope(q, positions), rope(k, positions)

        if ctx.cache is not None:
            ctx.cache.k[layer_idx][ctx.slot_mapping] = k
            ctx.cache.v[layer_idx][ctx.slot_mapping] = v

        out = self._prefill(q, k, v, ctx.seq_lens) if ctx.is_prefill else self._decode(q, ctx, layer_idx)
        return self.output_proj(rearrange(out, "... h d -> ... (h d)"))

    def _prefill(self, q, k, v, seq_lens) -> torch.Tensor:
        device = q.device
        mask = torch.block_diag(*[torch.tril(torch.ones(n, n, dtype=torch.bool, device=device)) for n in seq_lens])
        q = rearrange(q, "t h d -> h t d")
        k = rearrange(repeat_kv(k, self.n_rep, 1), "t h d -> h t d")
        v = rearrange(repeat_kv(v, self.n_rep, 1), "t h d -> h t d")
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        return rearrange(out, "h t d -> t h d")

    def _decode(self, q, ctx, layer_idx) -> torch.Tensor:
        device = q.device
        bs = ctx.cache.block_size
        lens = ctx.context_lens
        S, max_len = q.shape[0], int(lens.max())

        pos = torch.arange(max_len, device=device)
        slots = ctx.block_tables[:, pos // bs] * bs + (pos % bs)
        valid = pos.unsqueeze(0) < lens.unsqueeze(1)

        k = rearrange(repeat_kv(ctx.cache.k[layer_idx][slots], self.n_rep, 2), "s l h d -> s h l d")
        v = rearrange(repeat_kv(ctx.cache.v[layer_idx][slots], self.n_rep, 2), "s l h d -> s h l d")
        q = q.unsqueeze(2)
        out = scaled_dot_product_attention(q, k, v, mask=valid.view(S, 1, 1, max_len))
        return out.squeeze(2)
