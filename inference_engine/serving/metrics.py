from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("inference_requests_total", "Completed generation requests")
TOKENS = Counter("inference_tokens_generated_total", "Completion tokens generated")
LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end request latency (submit to completion)",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)
RUNNING = Gauge("inference_running_sequences", "Sequences in the decode set")
PENDING = Gauge("inference_pending_sequences", "Sequences awaiting prefill")
