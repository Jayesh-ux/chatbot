# Product Catalog Chatbot
# Run on a normal Linux/ChromeOS-Linux host:  docker compose up --build
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}

WORKDIR /app

# System deps required by chromadb (onnxruntime/numpy wheels)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the chat app on port 80
EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
