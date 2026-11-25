FROM python:3.12-slim-bookworm

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp
WORKDIR /app

COPY services/orchestrator_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY config.py project_state.py ./
COPY contracts/ ./contracts/
COPY services/__init__.py ./services/
COPY services/orchestrator_service/ ./services/orchestrator_service/
