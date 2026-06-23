from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODING = auto()

@dataclass
class SamplingParams:
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 1024
    ignore_eos: bool = False

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Sequence:
    input_ids: List[int]
    uid: int
    sampling_params: SamplingParams
    num_prompt_tokens: int
    kv: Optional[Any] = None                              # HF baseline path
    block_table: List[int] = field(default_factory=list)  # paged path
    num_cached_tokens: int = 0
    status: SequenceStatus = SequenceStatus.WAITING

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def last_token(self) -> int:
        return self.input_ids[-1]

    @property
    def num_completion_tokens(self) -> int:
        return len(self.input_ids) - self.num_prompt_tokens

    @property
    def completion_ids(self) -> List[int]:
        return self.input_ids[self.num_prompt_tokens:]

@dataclass(eq=False)
class Request:
    prompt: str
    uid: int
    sampling_params: SamplingParams

