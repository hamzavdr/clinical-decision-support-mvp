# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (use your pinned requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + vector store (committed chroma_db/)
COPY . .

# Cloud Run will inject $PORT; default to 8080 for local runs
ENV PORT=8080
EXPOSE 8080

CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8080"]
