from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import einsum

from .nn.attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from .nn.ffn import GeGLU
from .nn.linear import Embedding
from .nn.norm import GemmaRMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        layer_idx: int = 0,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = GemmaRMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, num_kv_heads, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        self.ln2 = GemmaRMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = GeGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), self.rope, positions, ctx, self.layer_idx)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, num_kv_heads, d_ff, context_length, rope_theta,
                    layer_idx=i, eps=eps, device=device, dtype=dtype,
                )
                for i in range(num_layers)
            ]
        )
        self.ln_final = GemmaRMSNorm(d_model, eps=eps, device=device, dtype=dtype)  # lm head is tied to token_embeddings

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ctx) -> torch.Tensor:
        x = self.token_embeddings(input_ids) * math.sqrt(self.d_model)
        for layer in self.layers:
            x = layer(x, positions, ctx)
        x = self.ln_final(x)
        return einsum(x, self.token_embeddings.weight, "... d_model, vocab_size d_model -> ... vocab_size")
