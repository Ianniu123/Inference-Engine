import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from inference_engine.core import SamplingParams
from inference_engine.serving.service import EngineService

MODEL = os.environ.get("MODEL", "google/gemma-2b")

service: EngineService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = EngineService(MODEL, MODEL)
    service.start()
    yield


app = FastAPI(lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 64


@app.post("/generate")
async def generate(req: GenerateRequest):
    params = SamplingParams(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    completion = await service.submit(req.prompt, params)
    return {"text": completion.text, "tokens": completion.num_tokens}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
