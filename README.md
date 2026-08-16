# Inference Engine

An LLM inference engine with continuous batching and a paged KV-cache, serving a Gemma model
written from scratch (weights loaded from a HuggingFace checkpoint and checked against it).
Prefill attention runs on a Triton FlashAttention-2 kernel; the rest is readable PyTorch, meant
to work through how vLLM-style serving fits together end to end.

## Features

- Continuous batching: the scheduler admits and retires sequences every iteration, and the
  active set runs in one batched forward.
- Paged KV-cache: a shared block pool with per-sequence block tables. Blocks are allocated on
  demand and reclaimed on completion, and sequences are preempted and recomputed under memory
  pressure.
- Triton FlashAttention-2 kernel for prefill: variable-length sequences packed with `cu_seqlens`
  and tiled so the attention matrix is never written to memory, making prefill O(N) in memory
  rather than O(N^2). Falls back to the PyTorch path when Triton or CUDA is unavailable.
- From-scratch Gemma transformer (RoPE, GQA/MQA, GeGLU, tied embeddings) with a weight loader
  that reconciles the interleaved vs. rotate_half RoPE layouts, checked token-for-token against
  HuggingFace.
- FastAPI serving with SSE streaming, Prometheus metrics, Grafana dashboards, and Docker /
  Kubernetes manifests.

## Architecture

```
client ──POST /generate──▶ FastAPI (async edge)
                              │  enqueue, await result
                              ▼
                        Engine loop (single thread)
                          scheduler     → prefill / decode batch
                          paged runner  → one batched forward
                          sampler       → next tokens
                          → stream tokens back
```

The engine loop (`add_request` / `step`) is synchronous and single-threaded. Concurrency lives
at the FastAPI edge: requests are queued and their results awaited, so many clients are batched
by the one loop.

## Benchmark

`python benchmark.py` compares continuous batching against static-batching and per-sequence
baselines on the same weights, and times prefill with and without the Triton kernel. On an
RTX 4090 with a 1.2B model (`BIG=1`):

- continuous batching runs about 1.7x the throughput of static batching on a variable-length
  workload, the win being the padding static batching wastes on finished sequences
- the Triton kernel makes prefill about 3x faster than the masked PyTorch path it replaced
- inter-token latency at batch 8 is 21 ms p50, 22 ms p99

It auto-uses the GPU (fp16) when present.

## Tests

`pytest` checks the from-scratch model token-for-token against HuggingFace (MHA/GQA/MQA), plus
batched-decode parity, preemption, and streaming. On a GPU it also checks the Triton prefill
kernel against the PyTorch path over ragged sequence lengths.

## Running it

```bash
pip install -r requirements.txt

MODEL=google/gemma-2b uvicorn server:app --port 8000
curl -s localhost:8000/generate        -d '{"prompt":"The capital of France is","max_tokens":32}'
curl -N localhost:8000/generate/stream -d '{"prompt":"Once upon a time","max_tokens":32}'
curl -s localhost:8000/metrics

docker compose -f deploy/docker-compose.yml up   # engine + Prometheus + Grafana
kubectl apply -f deploy/k8s/                      # deployment, service, HPA
```

## Notes

- Decode attention is still PyTorch, gathering K/V from the pool through the block tables.
  FlashAttention would not help there. Decode attends one query token against the whole cache,
  so its cost is reading the KV cache rather than building and storing a large score matrix.
  That path wants a paged-attention kernel instead, which is a different design.
- Preemption is recomputation-based: drop the victim's blocks and re-prefill later, as in vLLM.
- Next: chunked prefill and a paged decode kernel.
