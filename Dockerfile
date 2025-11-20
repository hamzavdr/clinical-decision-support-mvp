FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    TOKENIZERS_PARALLELISM=false \
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# (Optional) Ensure CPU-only torch; uncomment if you see CUDA deps
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
# RUN pip install --no-cache-dir -r requirements.txt

# Otherwise, one pass:
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-cache the ST model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

EXPOSE 8080
CMD ["streamlit","run","streamlit_app.py","--server.address=0.0.0.0","--server.port=8080"]
