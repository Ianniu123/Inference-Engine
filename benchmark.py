# HF-per-sequence vs static-batching vs continuous-batching throughput on identical (random)
# weights and a variable-length workload; useful tokens/sec. Random weights are fine -- timing
# depends on shapes, not values. Auto-uses GPU/fp16 when present.
#   python benchmark.py        # small model
#   BIG=1 python benchmark.py  # ~1.2B model, for a GPU run

import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaConfig, GemmaForCausalLM

from inference_engine.core import Request, SamplingParams, Sequence
from inference_engine.engine.engine import Engine
from inference_engine.model_runner import ModelRunner, PagedModelRunner
from inference_engine.models.loader import load_gemma_from_hf
from inference_engine.tokenizer import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
BIG = os.environ.get("BIG") == "1"

_rng = random.Random(0)
NUM_REQUESTS = 32
BATCH = 8
PROMPTS = [
    "The capital of France is", "Once upon a time there was", "In a distant galaxy",
    "The stock market today", "Scientists have discovered that", "My favorite food is",
    "The weather forecast says", "Deep learning models can",
]
# Realistic skew: most responses short, a minority long -- where static batching stalls.
LENGTHS = [_rng.randint(96, 128) if _rng.random() < 0.25 else _rng.randint(8, 24) for _ in range(NUM_REQUESTS)]
PROMPT_OF = [PROMPTS[i % len(PROMPTS)] for i in range(NUM_REQUESTS)]
USEFUL = sum(LENGTHS)


def build_model_dir() -> str:
    import tempfile

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    if BIG:  # ~1.2B, GQA -- representative of a small production model
        cfg = dict(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
                   num_attention_heads=16, num_key_value_heads=8, head_dim=128)
    else:
        cfg = dict(hidden_size=512, intermediate_size=2048, num_hidden_layers=6,
                   num_attention_heads=8, num_key_value_heads=8, head_dim=64)
    config = GemmaConfig(vocab_size=tok.vocab_size, max_position_embeddings=512,
                         hidden_act="gelu_pytorch_tanh", rms_norm_eps=1e-6, **cfg)
    d = tempfile.mkdtemp()
    GemmaForCausalLM(config).eval().save_pretrained(d)
    tok.save_pretrained(d)
    return d


def new_paged_runner(model, cfg) -> PagedModelRunner:
    return PagedModelRunner(
        model, num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads, vocab_size=cfg.vocab_size,
        num_blocks=1024, block_size=16, device=DEVICE, dtype=DTYPE,
    )


def make_requests():
    return [
        Request(prompt=PROMPT_OF[i], uid=i,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=LENGTHS[i], ignore_eos=True))
        for i in range(NUM_REQUESTS)
    ]


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def run_continuous(engine: Engine) -> float:
    engine.scheduler.max_num_seqs = BATCH
    engine.generate(make_requests()[:2])  # warmup
    _sync()
    t0 = time.perf_counter()
    engine.generate(make_requests())
    _sync()
    return USEFUL / (time.perf_counter() - t0)


@torch.no_grad()
def run_static(runner: PagedModelRunner, tokenizer: Tokenizer) -> float:
    ids = tokenizer.encode(PROMPT_OF)
    _sync()
    t0 = time.perf_counter()
    for i in range(0, NUM_REQUESTS, BATCH):
        wave = [Sequence(list(ids[j]), j, SamplingParams(), len(ids[j])) for j in range(i, min(i + BATCH, NUM_REQUESTS))]
        steps = max(LENGTHS[i:i + BATCH])
        prefill = True
        for _ in range(steps):  # keep every sequence in the batch for the wave's full length
            logits = runner.run(wave, is_prefill=prefill)
            for seq, tok in zip(wave, logits.argmax(-1).tolist()):
                seq.input_ids.append(tok)
            prefill = False
        for seq in wave:
            runner.free(seq)
    _sync()
    return USEFUL / (time.perf_counter() - t0)


def hf_baseline_engine(model_dir: str) -> Engine:
    runner = ModelRunner(model_dir)
    runner.model = runner.model.to(device=DEVICE, dtype=DTYPE)
    runner.device = DEVICE
    return Engine(tokenizer=Tokenizer(model_dir), model_runner=runner)


def main() -> None:
    model = os.environ.get("MODEL") or build_model_dir()
    hf = AutoModelForCausalLM.from_pretrained(model).eval()
    cfg = hf.config
    scratch = load_gemma_from_hf(hf).eval()  # stateless: shared across paged runners

    baseline = run_continuous(hf_baseline_engine(model))
    static = run_static(new_paged_runner(scratch, cfg), Tokenizer(model))
    cont = run_continuous(Engine(tokenizer=Tokenizer(model), model_runner=new_paged_runner(scratch, cfg)))

    size = "1.2B" if BIG else "6L/512d"
    print(f"device={DEVICE.type} dtype={DTYPE} model={size}  "
          f"workload: {NUM_REQUESTS} reqs, batch {BATCH}, {min(LENGTHS)}-{max(LENGTHS)} tokens\n")
    print(f"  HF baseline (per-seq)     {baseline:8.1f} tok/s")
    print(f"  static batching           {static:8.1f} tok/s")
    print(f"  continuous batching (new) {cont:8.1f} tok/s")
    print(f"\n  continuous vs static: {cont / static:.2f}x    vs HF per-seq: {cont / baseline:.2f}x")


if __name__ == "__main__":
    main()
