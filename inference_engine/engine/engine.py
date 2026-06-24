from typing import Dict, List, NamedTuple

from ..core import Request, Sequence, SequenceStatus
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

    def step(self) -> List[Completion]:
        batch, is_prefill = self.scheduler.schedule()
        if not batch:
            return []
        logits = self.model_runner.run(batch, is_prefill)
        tokens = self.sampler.sample(logits, self.sampler.prepare(batch))

        finished: List[Completion] = []
        for seq, token in zip(batch, tokens):
            seq.input_ids.append(token)
            seq.status = SequenceStatus.DECODING
            if self._is_finished(seq, token):
                self.scheduler.remove(seq)
                self.model_runner.free(seq)
                text = self.tokenizer.decode(seq.completion_ids)
                finished.append(Completion(seq.uid, text, seq.num_completion_tokens))
        return finished

    def _is_finished(self, seq: Sequence, token: int) -> bool:
        if not seq.sampling_params.ignore_eos and token == self.eos_token_id:
            return True
        return seq.num_completion_tokens >= seq.sampling_params.max_tokens
