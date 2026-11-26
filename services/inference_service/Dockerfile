FROM python:3.12-slim-bookworm

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
COPY services/inference_service/requirements.txt /tmp/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --default-timeout=120 --retries 5 --index-url "$TORCH_INDEX_URL" torch==2.7.1 \
    && pip install --no-cache-dir --default-timeout=120 --retries 5 -r /tmp/requirements.txt

COPY config.py project_state.py ./
COPY contracts/ ./contracts/
COPY services/__init__.py ./services/
COPY services/inference_service/ ./services/inference_service/
COPY services/model_service/models/ ./models/
