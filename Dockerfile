FROM python:3.12-slim

# System dependencies for docling / image processing / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home appuser
WORKDIR /app

# Install Python deps (cached layer — only re-runs when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run.py .
COPY src/ src/

# Data dirs will be mounted as volumes, but create them so the app
# doesn't fail if the user forgets to mount.
RUN mkdir -p data/pdfs data/output data/vectorstore \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /app /home/appuser/.cache

USER appuser

EXPOSE 7860

CMD ["python", "run.py", "serve", "--port", "7860"]
