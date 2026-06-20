from typing import List

from ..tokenizer import Tokenizer
from ..model_runner import ModelRunner
from ..core import Sequence, Request

class Engine:
    def __init__(self, tokenizer: str, model: str):
        self.tokenizer = Tokenizer(tokenizer)
        self.model_runner = ModelRunner(model)

    def loop(self):
        while True:
            # loop until termination signal
            
            # while less than budget and queue is not empty, keep filling up the batch
            # forward_batch()
            # send results back
            pass
    
    def forward_batch(self):
        #scheduler schedule
        #model runner process
        #sample
        # return processed
        pass

    def process_requests(self, requests: List[Request]) -> List[Sequence]:
        prompts = [request.prompt for request in requests]
        encoded_prompts = self.tokenizer.encode(prompts)
        return [Sequence(input_ids, request.uid, request.sampling_params, len(input_ids)) for request, input_ids in zip(requests, encoded_prompts)]

