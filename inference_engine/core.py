from dataclasses import dataclass
from typing import Any, List, Optional

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
    kv: Optional[Any] = None

@dataclass(eq=False)
class Request:
    prompt: str
    uid: int
    sampling_params: SamplingParams

