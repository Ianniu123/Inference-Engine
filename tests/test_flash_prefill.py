"""The Triton varlen prefill kernel must match the masked PyTorch path (GPU only)."""

import itertools

import pytest
import torch

from inference_engine.kv_cache import CacheBatch
from inference_engine.models.nn import attention as attention_module
from inference_engine.models.nn.attention import MultiHeadSelfAttention

SEQ_LENS = [17, 64, 5, 33]  # ragged on purpose: none are multiples of the tile size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")
@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
def test_flash_prefill_matches_torch(num_kv_heads):
    torch.manual_seed(0)
    d_model, num_heads = 128, 4
    head_dim = d_model // num_heads
    layer = MultiHeadSelfAttention(d_model, num_heads, num_kv_heads, device="cuda")

    total = sum(SEQ_LENS)
    q = torch.randn(total, num_heads, head_dim, device="cuda")
    k = torch.randn(total, num_kv_heads, head_dim, device="cuda")
    v = torch.randn(total, num_kv_heads, head_dim, device="cuda")
    ctx = CacheBatch(
        is_prefill=True,
        seq_lens=SEQ_LENS,
        cu_seqlens=torch.tensor([0, *itertools.accumulate(SEQ_LENS)], dtype=torch.int32, device="cuda"),
    )

    flash = layer._prefill(q, k, v, ctx)
    attention_module.USE_FLASH = False
    try:
        reference = layer._prefill(q, k, v, ctx)
    finally:
        attention_module.USE_FLASH = True

    torch.testing.assert_close(flash, reference, rtol=1e-2, atol=1e-2)
