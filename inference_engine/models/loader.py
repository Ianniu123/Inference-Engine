from __future__ import annotations

import torch

from .gemma import TransformerLM


def _hf_to_interleaved(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    # Reorder q/k output rows from HF's rotate_half layout [a0..,b0..] to our interleaved
    # RoPE layout [a0,b0,a1,b1,...]. v/o proj carry no RoPE and are copied as-is.
    d_out, d_in = w.shape
    head_dim = d_out // num_heads
    m = head_dim // 2
    return w.view(num_heads, 2, m, d_in).transpose(1, 2).reshape(d_out, d_in)


def load_gemma_from_hf(hf_model) -> TransformerLM:
    cfg = hf_model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.max_position_embeddings,
        d_model=cfg.hidden_size,
        num_layers=cfg.num_hidden_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        d_ff=cfg.intermediate_size,
        rope_theta=getattr(cfg, "rope_theta", 10000.0),
        eps=cfg.rms_norm_eps,
    )

    hf = hf_model.state_dict()
    sd = {"token_embeddings.weight": hf["model.embed_tokens.weight"]}
    for i in range(cfg.num_hidden_layers):
        p, q = f"model.layers.{i}.", f"layers.{i}."
        sd[q + "attn.q_proj.weight"] = _hf_to_interleaved(hf[p + "self_attn.q_proj.weight"], num_heads)
        sd[q + "attn.k_proj.weight"] = _hf_to_interleaved(hf[p + "self_attn.k_proj.weight"], num_kv_heads)
        sd[q + "attn.v_proj.weight"] = hf[p + "self_attn.v_proj.weight"]
        sd[q + "attn.output_proj.weight"] = hf[p + "self_attn.o_proj.weight"]
        sd[q + "ffn.w1.weight"] = hf[p + "mlp.gate_proj.weight"]
        sd[q + "ffn.w3.weight"] = hf[p + "mlp.up_proj.weight"]
        sd[q + "ffn.w2.weight"] = hf[p + "mlp.down_proj.weight"]
        sd[q + "ln1.weight"] = hf[p + "input_layernorm.weight"]
        sd[q + "ln2.weight"] = hf[p + "post_attention_layernorm.weight"]
    sd["ln_final.weight"] = hf["model.norm.weight"]

    model.load_state_dict(sd)
    model.eval()
    return model
