FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY rag_system/ rag_system/
COPY scripts/ scripts/
COPY static/ static/

EXPOSE 8000

CMD ["sh", "-c", "python -m scripts.warm_models && exec uvicorn rag_system.app:app --host 0.0.0.0 --port 8000"]
