FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

RUN mkdir -p /app/data /app/cache && chmod -R 777 /app/data /app/cache

COPY src/ /app/src/
COPY backend/ /app/backend/
COPY models/ /app/models/

EXPOSE 8000

ENV PYTHONPATH=/app:/app/backend

CMD ["sh", "-c", "exec python -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
