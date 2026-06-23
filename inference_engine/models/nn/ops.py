from __future__ import annotations

import math

import torch
from einops import einsum


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x, approximate="tanh")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x - x.amax(dim=dim, keepdim=True)
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return einsum(softmax(scores, dim=-1), V, "... query key, ... key d_v -> ... query d_v")
