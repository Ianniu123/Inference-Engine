from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, NamedTuple, Optional

from ..core import Request, Sequence, SequenceStatus
from ..kv_cache import OutOfBlocks
from ..model_runner import PagedModelRunner
from ..scheduler import Scheduler
from ..tokenizer import Tokenizer
from .sampler import Sampler


class Completion(NamedTuple):
    uid: int
    text: str
    num_tokens: int


class Engine:
    def __init__(self, tokenizer, model: str = None, model_runner=None):
        # Defaults to the from-scratch model over the paged cache; tokenizer/model_runner
        # can be injected (a test stub, or the HF baseline runner in the benchmark).
        self.tokenizer = tokenizer if not isinstance(tokenizer, str) else Tokenizer(tokenizer)
        self.model_runner = model_runner if model_runner is not None else PagedModelRunner.from_hf(model)
        self.sampler = Sampler(self.model_runner.vocab_size)
        self.scheduler = Scheduler()
        self.eos_token_id = self.tokenizer.eos_token_id
        self.num_preemptions = 0

    def add_request(self, request: Request) -> None:
        [input_ids] = self.tokenizer.encode([request.prompt])
        seq = Sequence(input_ids, request.uid, request.sampling_params, len(input_ids))
        self.scheduler.add(seq)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def generate(self, requests: List[Request]) -> Dict[int, str]:
        for request in requests:
            self.add_request(request)
        outputs: Dict[int, str] = {}
        while self.has_work():
            for c in self.step():
                outputs[c.uid] = c.text
        return outputs

    def generate_overlapped(self, requests: List[Request]) -> Dict[int, str]:
        # Overlap scheduler (SGLang style): the next batch's forward runs on a worker thread
        # while this thread post-processes the previous batch. On a GPU the async launch hides
        # the CPU work; on CPU the forward is synchronous, so this only mirrors the structure
        # (no speedup). Output is identical to generate().
        for request in requests:
            self.add_request(request)
        outputs: Dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=1) as pool:
            inflight = self._launch(pool)
            while inflight is not None:
                batch, _, future = inflight
                logits = future.result()
                tokens = self.sampler.sample(logits, self.sampler.prepare(batch))

                finished: List[Sequence] = []
                for seq, token in zip(batch, tokens):
                    seq.input_ids.append(token)
                    seq.status = SequenceStatus.DECODING
                    if self._is_finished(seq, token):
                        self.scheduler.remove(seq)
                        self.model_runner.free(seq)
                        finished.append(seq)

                inflight = self._launch(pool)
                for seq in finished:
                    outputs[seq.uid] = self.tokenizer.decode(seq.completion_ids)
        return outputs

    def _launch(self, pool: ThreadPoolExecutor):
        while self.scheduler.has_work():
            batch, is_prefill = self.scheduler.schedule()
            if not batch:
                return None
            batch = self._make_room(batch, is_prefill)
            if batch:
                return (batch, is_prefill, pool.submit(self.model_runner.run, batch, is_prefill))
        return None

    def _make_room(self, batch: List[Sequence], is_prefill: bool) -> List[Sequence]:
        has_capacity = getattr(self.model_runner, "has_capacity", None)
        while batch and has_capacity and not has_capacity(batch, is_prefill):
            victim = self._preemption_victim(batch)
            if victim is None:
                break
            self.scheduler.preempt(victim)
            self.model_runner.free(victim)
            self.num_preemptions += 1
            if victim in batch:
                batch.remove(victim)
        return batch

    def step(self, on_token: Optional[Callable[[int, str], None]] = None) -> List[Completion]:
        # on_token(uid, delta) streams each new piece of text as it is decoded (server path);
        # offline callers pass nothing and pay no per-token detokenization cost.
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return []

        logits = self._run_with_preemption(batch, is_prefill)
        if not batch:
            return []
        tokens = self.sampler.sample(logits, self.sampler.prepare(batch))

        finished: List[Completion] = []
        for seq, token in zip(batch, tokens):
            seq.input_ids.append(token)
            seq.status = SequenceStatus.DECODING
            if on_token is not None:
                text = self.tokenizer.decode(seq.completion_ids)
                delta = text[seq.num_streamed_chars:]
                seq.num_streamed_chars = len(text)
                if delta:
                    on_token(seq.uid, delta)
            if self._is_finished(seq, token):
                self.scheduler.remove(seq)
                self.model_runner.free(seq)
                text = self.tokenizer.decode(seq.completion_ids)
                finished.append(Completion(seq.uid, text, seq.num_completion_tokens))
        return finished

    def _run_with_preemption(self, batch: List[Sequence], is_prefill: bool):
        while batch:
            try:
                return self.model_runner.run(batch, is_prefill)
            except OutOfBlocks:
                victim = self._preemption_victim(batch)
                if victim is None:
                    raise
                self.scheduler.preempt(victim)
                self.model_runner.free(victim)
                self.num_preemptions += 1
                if victim in batch:
                    batch.remove(victim)
        return None

    def _preemption_victim(self, batch: List[Sequence]):
        # Prefer a victim not in the current batch so a re-prefilling sequence never evicts
        # itself; fall back to the newest in-batch sequence under pure decode pressure.
        for seq in reversed(self.scheduler.processing):
            if seq not in batch:
                return seq
        return self.scheduler.processing[-1] if len(batch) > 1 else None

    def _is_finished(self, seq: Sequence, token: int) -> bool:
        if not seq.sampling_params.ignore_eos and token == self.eos_token_id:
            return True
        return seq.num_completion_tokens >= seq.sampling_params.max_tokens
