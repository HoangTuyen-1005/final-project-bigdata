FROM python:3.12-slim-bookworm

ENV PYTHONPATH=/app:/app/services/model_service \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
COPY services/model_service/requirements.txt /tmp/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --default-timeout=120 --retries 5 --index-url "$TORCH_INDEX_URL" torch==2.7.1 \
    && pip install --no-cache-dir --default-timeout=120 --retries 5 -r /tmp/requirements.txt \
    && mkdir -p /tmp/image_cache /data/cache/huggingface \
    && chmod -R 777 /tmp/image_cache /data/cache

COPY config.py project_state.py ./
COPY contracts/ ./contracts/
COPY services/__init__.py ./services/
COPY services/model_service/ ./services/model_service/
