from typing import Dict, List, Tuple

from ..core import Request, Sequence, SequenceStatus
from ..model_runner import ModelRunner
from ..scheduler import Scheduler
from ..tokenizer import Tokenizer
from .sampler import Sampler


class Engine:
    def __init__(self, tokenizer: str, model: str):
        self.tokenizer = Tokenizer(tokenizer)
        self.model_runner = ModelRunner(model)
        self.sampler = Sampler(self.model_runner.vocab_size)
        self.scheduler = Scheduler()
        self.eos_token_id = self.tokenizer.eos_token_id

    def add_request(self, request: Request) -> None:
        # Receive = enqueue into the scheduler's waiting queue. (The shell hands us
        # requests off the inbox; tokenization is the request->sequence seam.)
        [input_ids] = self.tokenizer.encode([request.prompt])
        seq = Sequence(input_ids, request.uid, request.sampling_params, len(input_ids))
        self.scheduler.add(seq)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def generate(self, requests: List[Request]) -> Dict[int, str]:
        # Offline driver (nano-vllm LLM.generate style): no inbox, no transport — add a
        # fixed batch up front, then step to completion. This is the M0 loop the parity
        # test calls; the online thread/ZMQ shell is the same step() under a real inbox.
        for request in requests:
            self.add_request(request)
        outputs: Dict[int, str] = {}
        while self.has_work():
            for uid, text in self.step():
                outputs[uid] = text
        return outputs

    def step(self) -> List[Tuple[int, str]]:
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return []

        logits = self.model_runner.run(batch, is_prefill)
        args = self.sampler.prepare(batch)
        tokens = self.sampler.sample(logits, args)

        finished: List[Tuple[int, str]] = []
        for seq, token in zip(batch, tokens):
            seq.input_ids.append(token)
            seq.status = SequenceStatus.DECODING
            if self._is_finished(seq, token):
                self.scheduler.remove(seq)
                finished.append((seq.uid, self.tokenizer.decode(seq.completion_ids)))
        return finished

    def _is_finished(self, seq: Sequence, token: int) -> bool:
        if not seq.sampling_params.ignore_eos and token == self.eos_token_id:
            return True
        return seq.num_completion_tokens >= seq.sampling_params.max_tokens
