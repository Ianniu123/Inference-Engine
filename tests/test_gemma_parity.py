"""From-scratch Gemma prefill logits must match HF after load_gemma_from_hf (MHA/GQA/MQA)."""

import pytest
import torch
from transformers import GemmaConfig, GemmaForCausalLM

from inference_engine.kv_cache import CacheBatch
from inference_engine.models.loader import load_gemma_from_hf


@pytest.mark.parametrize("num_kv_heads", [4, 2, 1])  # MHA, GQA, MQA
def test_gemma_logit_parity(num_kv_heads):
    torch.manual_seed(0)
    hidden, heads = 64, 4
    cfg = GemmaConfig(
        vocab_size=256,
        hidden_size=hidden,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=heads,
        num_key_value_heads=num_kv_heads,
        head_dim=hidden // heads,
        max_position_embeddings=128,
        hidden_act="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
    )
    hf = GemmaForCausalLM(cfg).eval()
    mine = load_gemma_from_hf(hf).eval()

    prompt = [3, 14, 15, 92, 65, 35, 89, 79]
    input_ids = torch.tensor(prompt, dtype=torch.long)
    positions = torch.arange(len(prompt), dtype=torch.long)

    ctx = CacheBatch(is_prefill=True, seq_lens=[len(prompt)])
    with torch.no_grad():
        my_logits = mine(input_ids, positions, ctx)       # [T, V]
        hf_logits = hf(input_ids.unsqueeze(0)).logits[0]   # [T, V]

    assert torch.equal(hf_logits.argmax(-1), my_logits.argmax(-1))
    assert (hf_logits - my_logits).abs().max().item() < 1e-3
