"""Variable-length FlashAttention-2 prefill kernel in Triton. Forward only.

Prompts arrive packed with no padding and `cu_seqlens` marking the boundaries, the layout
flash_attn_varlen_func uses. Importing this module fails without Triton, which is what
selects the PyTorch fallback in attention.py.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_varlen_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    cu_seqlens_ptr,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_ot, stride_oh, stride_od,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    tile_index = tl.program_id(0)
    seq_index = tl.program_id(1)
    head_index = tl.program_id(2)

    seq_start = tl.load(cu_seqlens_ptr + seq_index)
    seq_len = tl.load(cu_seqlens_ptr + seq_index + 1) - seq_start

    if tile_index * Q_TILE_SIZE >= seq_len:  # grid is sized for the longest sequence
        return

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + seq_start * stride_qt + head_index * stride_qh,
        shape=(seq_len, D),
        strides=(stride_qt, stride_qd),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + seq_start * stride_kt + head_index * stride_kh,
        shape=(seq_len, D),
        strides=(stride_kt, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + seq_start * stride_vt + head_index * stride_vh,
        shape=(seq_len, D),
        strides=(stride_vt, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + seq_start * stride_ot + head_index * stride_oh,
        shape=(seq_len, D),
        strides=(stride_ot, stride_od),
        offsets=(tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    q_idx = tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(seq_len, K_TILE_SIZE)):
        k = tl.load(K_block_ptr, boundary_check=(0,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        S = tl.dot(q, tl.trans(k)) * scale

        # Also masks the zero-padded tail of the last key tile: those columns sit at
        # k_idx >= seq_len, past every real query.
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        S = S + tl.where(q_idx[:, None] < k_idx[None, :], -1e6, 0.0)

        m_new = tl.maximum(m, tl.max(S, axis=-1))
        P = tl.exp(S - m_new[:, None])
        m_scaled = tl.exp(m - m_new)

        l = m_scaled * l + tl.sum(P, axis=-1)
        o = o * m_scaled[:, None]
        o = tl.dot(P.to(v.dtype), v, acc=o)
        m = m_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    o = o * (1.0 / l)[:, None]
    tl.store(O_block_ptr, o.to(O_block_ptr.type.element_ty), boundary_check=(0,))


def flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen, q_tile=16, k_tile=16) -> torch.Tensor:
    """Causal attention over packed sequences.

    q, k, v:    (total_tokens, heads, head_dim)
    cu_seqlens: (num_sequences + 1,) int32 on device, cumulative token offsets
    """
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    _, heads, d = q.shape
    num_seqs = cu_seqlens.numel() - 1
    o = torch.empty_like(q)

    flash_fwd_varlen_kernel[(triton.cdiv(max_seqlen, q_tile), num_seqs, heads)](
        q, k, v, o,
        cu_seqlens,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        1.0 / math.sqrt(d),
        D=d,
        Q_TILE_SIZE=q_tile,
        K_TILE_SIZE=k_tile,
    )
    return o
