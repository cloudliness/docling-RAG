import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root (one level up from src/)
PROJECT_ROOT = Path(__file__).parent.parent

# LM Studio
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl-8b")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3-vl-8b")

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

# Paths
PDF_DIR = PROJECT_ROOT / os.getenv("PDF_DIR", "data/pdfs")
OUTPUT_DIR = PROJECT_ROOT / os.getenv("OUTPUT_DIR", "data/output")
VECTORSTORE_DIR = PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "data/vectorstore")

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
