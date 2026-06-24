# Thread + queue transport around the single-threaded Engine: one background thread owns the
# engine and runs the step loop; FastAPI coroutines submit over a queue and await a Future the
# engine thread resolves. Requests submitted mid-decode are admitted at the next scheduler pass.

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
        fut = self._loop.create_future()
        uid = next(self._uid)
        self._start_times[uid] = time.monotonic()
        self._incoming.put((Request(prompt=prompt, uid=uid, sampling_params=params), fut))
        self._wake.set()
        return fut

    def _run(self) -> None:
        while True:
            self._drain_incoming()
            if self.engine.has_work():
                for completion in self.engine.step():
                    self._resolve(completion)
                self._update_gauges()
            else:
                self._wake.wait()
                self._wake.clear()

    def _drain_incoming(self) -> None:
        while not self._incoming.empty():
            req, fut = self._incoming.get()
            self._futures[req.uid] = fut
            self.engine.add_request(req)

    def _resolve(self, completion: Completion) -> None:
        fut = self._futures.pop(completion.uid, None)
        start = self._start_times.pop(completion.uid, None)
        if start is not None:
            metrics.REQUESTS.inc()
            metrics.TOKENS.inc(completion.num_tokens)
            metrics.LATENCY.observe(time.monotonic() - start)
        if fut is not None and not fut.done():
            self._loop.call_soon_threadsafe(fut.set_result, completion)

    def _update_gauges(self) -> None:
        metrics.RUNNING.set(len(self.engine.scheduler.processing))
        metrics.PENDING.set(len(self.engine.scheduler.pending))
