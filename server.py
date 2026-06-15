from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import torch
from generate import load_model, load_tokenizer, generate

app = FastAPI()

# Configuration
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 4
MAX_WAIT_MS = 50  # milliseconds to wait to form a batch

# Global state
queue = asyncio.Queue()
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.8
    top_k: int = 50
    max_tokens: int = 50

# Startup event to load the model
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print("Loading model and tokenizer...")
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DEVICE)
    
    # Start the background batch processing task
    asyncio.create_task(batch_processor())
    print("Server ready!")

async def batch_processor():
    """Background loop that collects requests into batches and processes them."""
    while True:
        batch = []
        
        # Wait for the first request
        req, future = await queue.get()
        batch.append((req, future))
        
        # Wait a tiny bit (e.g. 50ms) to see if more requests arrive
        try:
            # Gather up to MAX_BATCH_SIZE requests
            while len(batch) < MAX_BATCH_SIZE:
                # Wait with timeout for the next request
                req, future = await asyncio.wait_for(queue.get(), timeout=MAX_WAIT_MS / 1000.0)
                batch.append((req, future))
        except asyncio.TimeoutError:
            # Time's up, process whatever we have
            pass
        
        # Process the batch
        print(f"Processing batch of size {len(batch)}")
        
        # Extract prompts and parameters
        prompts = [item[0].prompt for item in batch]
        # In a real system, parameters could vary per request, but here we enforce batch uniformity
        temp = batch[0][0].temperature
        top_k = batch[0][0].top_k
        max_tokens = batch[0][0].max_tokens
        
        try:
            # Call your refactored batched generate function!
            # Since generate takes a list of strings, this works perfectly.
            # Run in a thread so it doesn't block the asyncio event loop
            responses = await asyncio.to_thread(
                generate, model, tokenizer, prompts, temp, top_k, max_tokens, True
            )
            
            # Send the responses back to the waiting requests
            for i, (_, future) in enumerate(batch):
                future.set_result(responses[i])
                
        except Exception as e:
            # If the batch fails, fail all requests in the batch
            for _, future in batch:
                if not future.done():
                    future.set_exception(e)

@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    # Create an asyncio Future for this specific request
    future = asyncio.Future()
    
    # Put the request AND its personal Future into the queue
    await queue.put((request, future))
    
    # Wait for the background worker to fill the Future with the result
    response_text = await future
    
    return {"generated_text": response_text}
