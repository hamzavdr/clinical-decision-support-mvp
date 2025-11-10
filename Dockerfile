# Dockerfile
FROM python:3.11-slim

# Prevents prompts
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps (if PyMuPDF needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy code first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Streamlit config for headless
ENV STREAMLIT_PORT=7860
ENV PORT=7860
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# For sentence-transformers on CPU
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
