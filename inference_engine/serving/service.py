# Thread + queue transport around the single-threaded Engine: one background thread owns the
# engine and runs the step loop; FastAPI coroutines submit over a queue and await a Future
# (or drain an asyncio.Queue for streaming) that the engine thread resolves.

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from queue import SimpleQueue
from typing import Optional

from ..core import Request, SamplingParams
from ..engine.engine import Completion, Engine
from . import metrics


class EngineService:
    def __init__(self, tokenizer: str, model: str):
        self.engine = Engine(tokenizer, model)
        self._incoming: SimpleQueue = SimpleQueue()
        self._futures: dict[int, asyncio.Future] = {}
        self._streams: dict[int, asyncio.Queue] = {}
        self._start_times: dict[int, float] = {}
        self._uid = itertools.count()
        self._wake = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, prompt: str, params: SamplingParams) -> asyncio.Future:
        sink = self._loop.create_future()
        self._enqueue(prompt, params, sink)
        return sink

    def submit_stream(self, prompt: str, params: SamplingParams) -> asyncio.Queue:
        sink: asyncio.Queue = asyncio.Queue()
        self._enqueue(prompt, params, sink)
        return sink

    def _enqueue(self, prompt: str, params: SamplingParams, sink) -> None:
        uid = next(self._uid)
        self._start_times[uid] = time.monotonic()
        self._incoming.put((Request(prompt=prompt, uid=uid, sampling_params=params), sink))
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._drain_incoming()
            if self.engine.has_work():
                for completion in self.engine.step(on_token=self._on_token):
                    self._finish(completion)
                self._update_gauges()
            else:
                self._wake.wait()
                self._wake.clear()

    def _drain_incoming(self) -> None:
        while not self._incoming.empty():
            req, sink = self._incoming.get()
            (self._streams if isinstance(sink, asyncio.Queue) else self._futures)[req.uid] = sink
            self.engine.add_request(req)

    def _on_token(self, uid: int, delta: str) -> None:
        q = self._streams.get(uid)
        if q is not None:
            self._loop.call_soon_threadsafe(q.put_nowait, delta)

    def _finish(self, completion: Completion) -> None:
        uid = completion.uid
        start = self._start_times.pop(uid, None)
        if start is not None:
            metrics.REQUESTS.inc()
            metrics.TOKENS.inc(completion.num_tokens)
            metrics.LATENCY.observe(time.monotonic() - start)

        q = self._streams.pop(uid, None)
        if q is not None:
            self._loop.call_soon_threadsafe(q.put_nowait, None)  # end-of-stream sentinel
            return
        fut = self._futures.pop(uid, None)
        if fut is not None and not fut.done():
            self._loop.call_soon_threadsafe(fut.set_result, completion)

    def _update_gauges(self) -> None:
        metrics.RUNNING.set(len(self.engine.scheduler.processing))
        metrics.PENDING.set(len(self.engine.scheduler.pending))
