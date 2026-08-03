"""Streamed per-token deltas must concatenate to the full completion text."""

import torch
from transformers import AutoTokenizer, GemmaConfig, GemmaForCausalLM

from inference_engine.core import Request, SamplingParams
from inference_engine.engine.engine import Engine
from inference_engine.model_runner import ModelRunner
from inference_engine.models.loader import load_gemma_from_hf
from inference_engine.tokenizer import Tokenizer


def test_stream_deltas_reconstruct_text():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    cfg = GemmaConfig(
        vocab_size=tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, max_position_embeddings=128,
        hidden_act="gelu_pytorch_tanh", rms_norm_eps=1e-6,
    )
    hf = GemmaForCausalLM(cfg).eval()
    runner = ModelRunner(
        load_gemma_from_hf(hf).eval(), num_layers=2, num_kv_heads=4,
        head_dim=16, vocab_size=cfg.vocab_size, num_blocks=64, block_size=8,
    )
    engine = Engine(tokenizer=Tokenizer("gpt2"), model_runner=runner)

    params = SamplingParams(temperature=0.0, max_tokens=20, ignore_eos=True)
    engine.add_request(Request(prompt="The capital of France is", uid=0, sampling_params=params))

    deltas: list[str] = []
    completion = None
    while engine.has_work():
        for c in engine.step(on_token=lambda uid, d: deltas.append(d)):
            completion = c

    assert completion is not None
    assert "".join(deltas) == completion.text
    assert len(completion.text) > 0
