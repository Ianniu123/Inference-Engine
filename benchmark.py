import time
import torch
import matplotlib.pyplot as plt
from generate import load_model, load_tokenizer, generate

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
TOKENS_TO_GENERATE = 20   # Keep small for CPU — increase if on GPU
TEMPERATURE = 0.0         # Greedy: deterministic, so both runs produce same tokens
TOP_K = 1                 # Greedy

# Sequence lengths to benchmark (prompt token counts, approximate)
# ⚠️ On CPU this will be slow — each data point takes ~1-3 min.
# Reduce to [16, 32, 64] if you want faster results.
SEQUENCE_LENGTHS = [16, 32, 64, 128]


def make_prompt(approx_tokens: int) -> str:
    """Generate a dummy prompt of approximately `approx_tokens` tokens."""
    return "hello " * approx_tokens  # each 'hello ' ≈ 1-2 tokens


def time_generate(model, tokenizer, prompt, use_cache: bool) -> float:
    """Run generate() once and return elapsed wall-clock time in seconds."""
    start = time.perf_counter()
    generate(
        model, tokenizer, prompt,
        temperature=TEMPERATURE,
        k=TOP_K,
        max_tokens=TOKENS_TO_GENERATE,
        use_cache=use_cache,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # wait for all CUDA kernels to finish
    return time.perf_counter() - start


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}\n")

    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, device)

    latency_cached    = []
    latency_no_cache  = []

    print("Warming up...")
    warmup_prompt = make_prompt(8)
    time_generate(model, tokenizer, warmup_prompt, use_cache=True)
    time_generate(model, tokenizer, warmup_prompt, use_cache=False)
    print("Done.\n")

    for seq_len in SEQUENCE_LENGTHS:
        prompt = make_prompt(seq_len)
        actual_tokens = tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]

        print(f"Prompt length: ~{actual_tokens} tokens")

        t_cached   = time_generate(model, tokenizer, prompt, use_cache=True)
        t_no_cache = time_generate(model, tokenizer, prompt, use_cache=False)

        latency_cached.append(t_cached)
        latency_no_cache.append(t_no_cache)

        speedup = t_no_cache / t_cached if t_cached > 0 else float('inf')
        print(f"  cached={t_cached:.2f}s  |  no_cache={t_no_cache:.2f}s  |  speedup={speedup:.2f}x\n")

    # --- Plot ---
    plt.figure(figsize=(9, 5))
    plt.plot(SEQUENCE_LENGTHS, latency_no_cache, 'r-o', label='No KV Cache', linewidth=2)
    plt.plot(SEQUENCE_LENGTHS, latency_cached,   'g-o', label='With KV Cache', linewidth=2)
    plt.xlabel('Approximate Prompt Length (tokens)')
    plt.ylabel(f'Time to generate {TOKENS_TO_GENERATE} tokens (s)')
    plt.title('KV Cache: Latency vs. Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kv_cache_benchmark.png', dpi=150)
    plt.show()
    print("Plot saved → kv_cache_benchmark.png")