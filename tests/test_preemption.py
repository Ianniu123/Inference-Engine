"""Under a too-small pool the engine must preempt/recompute yet still match HF greedy decode."""

import torch
from transformers import GemmaConfig, GemmaForCausalLM

from inference_engine.core import Request, SamplingParams
from inference_engine.engine.engine import Engine
from inference_engine.model_runner import ModelRunner
from inference_engine.models.loader import load_gemma_from_hf

NEW_TOKENS = 15
BLOCK_SIZE = 4
NUM_BLOCKS = 10  # fits ~1-2 sequences; 3 concurrent -> forces preemption
PROMPTS = [
    [3, 14, 15, 92, 65, 35],
    [7, 8, 9, 10, 11, 12],
    [21, 22, 23, 24, 25, 26],
]


class StubTokenizer:
    def __init__(self, prompts):
        self._prompts, self._i, self.eos_token_id = list(prompts), 0, -1

    def encode(self, prompts):
        out = [list(self._prompts[self._i])]
        self._i += 1
        return out

    def decode(self, token_ids):
        return ""


@torch.no_grad()
def hf_greedy(hf, prompt_ids):
    out = hf(torch.tensor([prompt_ids]), use_cache=True)
    past, generated = out.past_key_values, []
    for _ in range(NEW_TOKENS):
        nxt = int(out.logits[0, -1].argmax())
        generated.append(nxt)
        out = hf(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
    return generated


def test_preemption_preserves_output():
    torch.manual_seed(0)
    hidden, heads = 64, 4
    cfg = GemmaConfig(
        vocab_size=256, hidden_size=hidden, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=heads, num_key_value_heads=heads, head_dim=hidden // heads,
        max_position_embeddings=128, hidden_act="gelu_pytorch_tanh", rms_norm_eps=1e-6,
    )
    hf = GemmaForCausalLM(cfg).eval()
    runner = ModelRunner(
        load_gemma_from_hf(hf).eval(),
        num_layers=cfg.num_hidden_layers, num_kv_heads=heads,
        head_dim=hidden // heads, vocab_size=cfg.vocab_size,
        num_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE,
    )
    engine = Engine(tokenizer=StubTokenizer(PROMPTS), model_runner=runner)

    params = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)
    seqs = []
    for i in range(len(PROMPTS)):
        engine.add_request(Request(prompt="x", uid=i, sampling_params=params))
        seqs.append(engine.scheduler.pending[-1])

    while engine.has_work():
        engine.step()

    assert engine.num_preemptions > 0                    # the pool pressure actually triggered it
    for seq, prompt in zip(seqs, PROMPTS):
        assert seq.completion_ids == hf_greedy(hf, prompt)  # recomputation kept output identical
    assert len(runner.blocks.free) == NUM_BLOCKS         # all blocks reclaimed at the end
