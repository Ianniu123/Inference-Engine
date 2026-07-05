# CPU image for the inference engine. Uses the PyTorch CPU wheels to keep the
# image lean; swap the extra-index-url for a CUDA build to run on GPU.
FROM python:3.12-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY inference_engine ./inference_engine
COPY server.py ./

ENV MODEL=google/gemma-2b
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
