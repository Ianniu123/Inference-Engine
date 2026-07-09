# Owns admission and KV block accounting: sequences are admitted only when their prompt
# fits the pool (a small watermark reserves room for decode growth), and decode steps that
# outgrow the pool preempt the newest sequence back to pending.

from collections import deque
from typing import List, Optional, Tuple

from ..core import Sequence, SequenceStatus
from ..kv_cache import BlockManager, OutOfBlocks


class Scheduler:
    def __init__(self, max_num_seqs: int = 256, block_manager: Optional[BlockManager] = None):
        self.max_num_seqs = max_num_seqs
        self.blocks = block_manager
        self.watermark = max(1, len(block_manager.free) // 100) if block_manager else 0
        self.pending: deque[Sequence] = deque()
        self.processing: deque[Sequence] = deque()
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.pending.append(seq)

    def schedule(self) -> Tuple[List[Sequence], bool]:
        # Prefill-priority: admit waiting sequences while they fit (decode pauses that step).
        # An idle engine force-admits so an oversized request fails loudly instead of stalling.
        batch: List[Sequence] = []
        while self.pending and len(self.processing) < self.max_num_seqs:
            seq = self.pending[0]
            if self.blocks is not None:
                need = self.blocks.blocks_for(seq.num_tokens)
                if need > len(self.blocks.free) - self.watermark and (batch or self.processing):
                    break
                seq.block_table.extend(self.blocks.allocate(need))
            self.pending.popleft()
            seq.status = SequenceStatus.PREFILL
            batch.append(seq)
            self.processing.append(seq)
        if batch:
            return batch, True
        return self._decode_batch(), False

    def _decode_batch(self) -> List[Sequence]:
        if self.blocks is not None:
            for seq in list(self.processing):
                if seq.status is not SequenceStatus.WAITING:  # skip seqs preempted this pass
                    self._grow(seq)
        return list(self.processing)

    def _grow(self, seq: Sequence) -> None:
        need = self.blocks.blocks_for(seq.num_cached_tokens + 1) - len(seq.block_table)
        while need > len(self.blocks.free):
            self._preempt_newest(seq)
        if need > 0:
            seq.block_table.extend(self.blocks.allocate(need))

    def _preempt_newest(self, seq: Sequence) -> None:
        for victim in reversed(self.processing):
            if victim is not seq:
                self.preempt(victim)
                return
        raise OutOfBlocks("pool cannot hold a single sequence")

    def preempt(self, seq: Sequence) -> None:
        # Recomputation preemption: generated tokens stay on the Sequence, so re-prefilling
        # from token zero reproduces the same output.
        self.processing.remove(seq)
        self._release(seq)
        seq.status = SequenceStatus.WAITING
        self.pending.appendleft(seq)
        self.num_preemptions += 1

    def remove(self, seq: Sequence) -> None:
        self.processing.remove(seq)
        self._release(seq)

    def _release(self, seq: Sequence) -> None:
        if self.blocks is not None:
            self.blocks.free_blocks(seq.block_table)
            seq.block_table = []
        seq.num_cached_tokens = 0

    def has_work(self) -> bool:
        return bool(self.pending or self.processing)
