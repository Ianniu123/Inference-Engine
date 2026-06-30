from collections import deque
from typing import List, Tuple

from ..core import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, max_num_seqs: int = 256):
        self.max_num_seqs = max_num_seqs
        self.pending: deque[Sequence] = deque()
        self.processing: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self.pending.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        # Prefill-priority: admit waiting sequences (decode pauses that step), else decode.
        if self.pending:
            budget = self.max_num_seqs - len(self.processing)
            batch: List[Sequence] = []
            while self.pending and len(batch) < budget:
                seq = self.pending.popleft()
                seq.status = SequenceStatus.PREFILL
                batch.append(seq)
                self.processing.append(seq)
            if batch:
                return batch, True

        return list(self.processing), False

    def remove(self, seq: Sequence) -> None:
        self.processing.remove(seq)

    def preempt(self, seq: Sequence) -> None:
        # Back to the waiting queue to re-prefill; generated tokens stay on the Sequence,
        # so recomputation leaves output unchanged. Caller frees its blocks.
        self.processing.remove(seq)
        seq.status = SequenceStatus.WAITING
        self.pending.appendleft(seq)

    def has_work(self) -> bool:
        return bool(self.pending or self.processing)
