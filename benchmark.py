# HF-per-sequence vs static-batching vs continuous-batching throughput on identical (random)
# weights and a variable-length workload; useful tokens/sec. Random weights are fine -- timing
# depends on shapes, not values. Auto-uses GPU/fp16 when present.
#   python benchmark.py        # small model
#   BIG=1 python benchmark.py  # ~1.2B model, for a GPU run

import gc
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GemmaConfig, GemmaForCausalLM

from inference_engine.core import Request, SamplingParams, Sequence
from inference_engine.engine.engine import Engine
from inference_engine.model_runner import HFBaselineRunner, ModelRunner
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


def new_paged_runner(model, cfg) -> ModelRunner:
    return ModelRunner(
        model, num_layers=cfg.num_hidden_layers,
        num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        head_dim=cfg.hidden_size // cfg.num_attention_heads, vocab_size=cfg.vocab_size,
        num_blocks=512, block_size=16, device=DEVICE, dtype=DTYPE,
    )


def _free():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def make_requests():
    return [
        Request(prompt=PROMPT_OF[i], uid=i,
                sampling_params=SamplingParams(temperature=0.0, max_tokens=LENGTHS[i], ignore_eos=True))
        for i in range(NUM_REQUESTS)
    ]


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# The static/latency paths drive the runner directly (no scheduler), so they reserve
# blocks themselves: enough for the prompt plus every token they will generate.
def reserve(runner: ModelRunner, seqs, new_tokens: int) -> None:
    for seq in seqs:
        seq.block_table = runner.blocks.allocate(runner.blocks.blocks_for(seq.num_tokens + new_tokens))


def release(runner: ModelRunner, seqs) -> None:
    for seq in seqs:
        runner.blocks.free_blocks(seq.block_table)
        seq.block_table = []


def run_continuous(engine: Engine) -> float:
    engine.scheduler.max_num_seqs = BATCH
    engine.generate(make_requests()[:2])  # warmup
    _sync()
    t0 = time.perf_counter()
    engine.generate(make_requests())
    _sync()
    return USEFUL / (time.perf_counter() - t0)


@torch.no_grad()
def run_static(runner: ModelRunner, tokenizer: Tokenizer) -> float:
    ids = tokenizer.encode(PROMPT_OF)
    _sync()
    t0 = time.perf_counter()
    for i in range(0, NUM_REQUESTS, BATCH):
        wave = [Sequence(list(ids[j]), j, SamplingParams(), len(ids[j])) for j in range(i, min(i + BATCH, NUM_REQUESTS))]
        steps = max(LENGTHS[i:i + BATCH])
        reserve(runner, wave, steps)
        prefill = True
        for _ in range(steps):  # keep every sequence in the batch for the wave's full length
            logits = runner.run(wave, is_prefill=prefill)
            for seq, tok in zip(wave, logits.argmax(-1).tolist()):
                seq.input_ids.append(tok)
            prefill = False
        release(runner, wave)
    _sync()
    return USEFUL / (time.perf_counter() - t0)


def hf_baseline_engine(model_dir: str) -> Engine:
    runner = HFBaselineRunner(model_dir)
    runner.model = runner.model.to(device=DEVICE, dtype=DTYPE)
    runner.device = DEVICE
    return Engine(tokenizer=Tokenizer(model_dir), model_runner=runner)


@torch.no_grad()
def decode_latency_ms(runner: ModelRunner, tokenizer: Tokenizer):
    # Inter-token latency: time each decode step over a full batch (one token per sequence).
    ids = tokenizer.encode(PROMPT_OF[:BATCH])
    seqs = [Sequence(list(ids[i]), i, SamplingParams(max_tokens=99, ignore_eos=True), len(ids[i])) for i in range(BATCH)]
    reserve(runner, seqs, 41)  # 1 prefill token + 40 timed decode steps
    logits = runner.run(seqs, is_prefill=True)
    for seq, t in zip(seqs, logits.argmax(-1).tolist()):
        seq.input_ids.append(t)
    times = []
    for _ in range(40):
        _sync(); t0 = time.perf_counter()
        logits = runner.run(seqs, is_prefill=False)
        _sync(); times.append((time.perf_counter() - t0) * 1000)
        for seq, t in zip(seqs, logits.argmax(-1).tolist()):
            seq.input_ids.append(t)
    release(runner, seqs)
    times.sort()
    return times[len(times) // 2], times[int(0.99 * len(times))]


@torch.no_grad()
def prefill_ms(runner: ModelRunner, use_flash: bool, prompt_len: int = 512):
    # Time to first token is entirely prefill, which is the only path the Triton kernel
    # touches. Returns None when the kernel is unavailable (no CUDA / no Triton).
    from inference_engine.models.nn import attention

    if use_flash and attention.flash_attention_varlen is None:
        return None

    previous, attention.USE_FLASH = attention.USE_FLASH, use_flash
    try:
        seqs = [
            Sequence([_rng.randrange(100) for _ in range(prompt_len)], i, SamplingParams(), prompt_len)
            for i in range(BATCH)
        ]
        reserve(runner, seqs, 1)
        runner.run(seqs, is_prefill=True)  # warmup
        _sync()
        t0 = time.perf_counter()
        for _ in range(10):
            runner.run(seqs, is_prefill=True)
        _sync()
        release(runner, seqs)
        return (time.perf_counter() - t0) / 10 * 1000
    finally:
        attention.USE_FLASH = previous


def main() -> None:
    model = os.environ.get("MODEL") or build_model_dir()
    hf = AutoModelForCausalLM.from_pretrained(model).eval()
    cfg = hf.config
    scratch = load_gemma_from_hf(hf).eval()  # stateless: shared across paged runners
    del hf
    _free()

    baseline = run_continuous(hf_baseline_engine(model))
    _free()  # drop the HF baseline model before allocating paged KV pools
    static = run_static(new_paged_runner(scratch, cfg), Tokenizer(model))
    _free()
    cont = run_continuous(Engine(tokenizer=Tokenizer(model), model_runner=new_paged_runner(scratch, cfg)))
    _free()

    size = "1.2B" if BIG else "6L/512d"
    print(f"device={DEVICE.type} dtype={DTYPE} model={size}  "
          f"workload: {NUM_REQUESTS} reqs, batch {BATCH}, {min(LENGTHS)}-{max(LENGTHS)} tokens\n")
    print(f"  HF baseline (per-seq)     {baseline:8.1f} tok/s")
    print(f"  static batching           {static:8.1f} tok/s")
    print(f"  continuous batching (new) {cont:8.1f} tok/s")
    print(f"\n  continuous vs static: {cont / static:.2f}x    vs HF per-seq: {cont / baseline:.2f}x")

    p50, p99 = decode_latency_ms(new_paged_runner(scratch, cfg), Tokenizer(model))
    print(f"\n  inter-token latency @ batch {BATCH}: p50 {p50:.1f} ms   p99 {p99:.1f} ms")
    _free()

    flash = prefill_ms(new_paged_runner(scratch, cfg), use_flash=True)
    _free()
    if flash is not None:
        torch_attn = prefill_ms(new_paged_runner(scratch, cfg), use_flash=False)
        print(f"  prefill @ batch {BATCH} x 512 tokens: triton {flash:.1f} ms   "
              f"pytorch {torch_attn:.1f} ms   {torch_attn / flash:.2f}x")


if __name__ == "__main__":
    main()
