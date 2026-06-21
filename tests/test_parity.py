"""M0 parity test: the engine's prefill/decode/sampler plumbing must reproduce a
plain greedy HF decode token-for-token.

Greedy (temperature=0) decoding is deterministic, so this is a real equality test, not
a fuzzy one. Both sides share the *same* loaded HF model (we reuse the engine's own
model object as the reference oracle), so any divergence is purely the engine's
plumbing -- KV handling, last-token logit selection, the decode step, the sampler's
greedy path -- not a difference in weights, device, or tokenization.
"""

import os

import pytest
import torch

from inference_engine.core import Request, SamplingParams
from inference_engine.engine.engine import Engine

MODEL = os.environ.get("PARITY_MODEL", "gpt2")
PROMPT = "The capital of France is"
MAX_TOKENS = 20


@torch.no_grad()
def reference_greedy(model, tokenizer, device, prompt: str, max_tokens: int):
    """Canonical single-sequence greedy decode with a KV cache. Mirrors the engine's
    stopping rule: emit up to max_tokens tokens, stopping after EOS (EOS included)."""
    input_ids = torch.tensor([tokenizer(prompt)["input_ids"]], dtype=torch.long, device=device)
    out = model(input_ids=input_ids, use_cache=True)  # prefill
    past = out.past_key_values

    generated = []
    for _ in range(max_tokens):
        next_id = int(out.logits[0, -1].argmax())
        generated.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
        cur = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=cur, past_key_values=past, use_cache=True)  # decode
        past = out.past_key_values
    return generated


@pytest.fixture(scope="module")
def engine():
    try:
        return Engine(MODEL, MODEL)
    except Exception as exc:  # model not cached / offline
        pytest.skip(f"could not load {MODEL!r}: {exc}")


def test_engine_matches_greedy_reference(engine):
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    request = Request(prompt=PROMPT, uid=0, sampling_params=params)

    # Hold a reference to the Sequence before stepping; the engine appends sampled
    # tokens to seq.input_ids in place, so after the run it holds the full completion.
    engine.add_request(request)
    seq = engine.scheduler.pending[-1]
    while engine.has_work():
        engine.step()
    engine_ids = seq.completion_ids

    reference_ids = reference_greedy(
        engine.model_runner.model,
        engine.tokenizer.tokenizer,
        engine.model_runner.device,
        PROMPT,
        MAX_TOKENS,
    )

    assert engine_ids == reference_ids
    assert len(engine_ids) > 0
