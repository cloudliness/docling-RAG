# Docling RAG Pipeline

PDF → Docling (+ LM Studio vision) → Markdown → ChromaDB → RAG

A fully local RAG pipeline that:
1. Converts PDFs to markdown using **Docling**, with **LM Studio + Qwen3-VL-8B** generating rich image descriptions
2. Chunks and embeds the markdown into **ChromaDB** using sentence-transformers
3. Answers questions via a **LangChain** retrieval chain powered by LM Studio
4. Provides a **Gradio web UI** for interactive Q&A

All inference runs locally — no cloud APIs required.

## Prerequisites

- **Python 3.10+**
- **LM Studio** running locally with a vision model loaded (e.g., `qwen3-vl-8b`)
  - Download from [lmstudio.ai](https://lmstudio.ai)
  - Load your model and start the server (default: `http://localhost:1234`)

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `.env` to match your setup:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1    # LM Studio API
VISION_MODEL=qwen3-vl-8b                      # Model for image descriptions
CHAT_MODEL=qwen3-vl-8b                        # Model for RAG responses
EMBEDDING_MODEL=Alibaba-NLP/gte-large-en-v1.5
```

## Usage

### 1. Add your PDFs

Drop PDF files into the `data/pdfs/` folder.

### 2. Run the full pipeline

```bash
python run.py pipeline
```

This converts all PDFs to markdown (with image descriptions from LM Studio) and indexes them into ChromaDB.

### 3. Launch the web UI

```bash
python run.py serve
```

Open `http://localhost:7860` in your browser to chat with your documents.

### Individual commands

```bash
# Convert PDFs to markdown (does NOT touch the vector store)
python run.py convert

# Ingest all markdown files from data/output/ into ChromaDB
python run.py ingest

# Ingest only specific files
python run.py ingest "10.1 - Graphs and Graph Models.md" table1.md

# List all indexed documents and their chunk counts
python run.py list

# Remove a specific document from the vector store
python run.py remove table1.md

# Remove multiple documents at once
python run.py remove table1.md "Sex-Secrets - Secrets Of Seduction ross jeffries.md"

# Query from the command line
python run.py query "What are the main findings?"

# Launch Gradio UI
python run.py serve
python run.py serve --port 8080        # Custom port
python run.py serve --share            # Public link

# Enable debug logging
python run.py -v convert
```

### Managing the vector store

Conversion and ingestion are **decoupled** — converting a PDF to markdown does not automatically index it. This gives you full control over what ends up in the vector store.

```bash
# Typical workflow:
# 1. Convert your PDFs
python run.py convert

# 2. See what markdown files are available
ls data/output/

# 3. Ingest only the ones you want
python run.py ingest "10.1 - Graphs and Graph Models.md"

# 4. Check what's indexed
python run.py list

# 5. Remove something you no longer need
python run.py remove "10.1 - Graphs and Graph Models.md"
```

Ingestion is **incremental** — running `ingest` adds to the existing store rather than replacing it. To re-index a file, `remove` it first, then `ingest` it again.

The Gradio web UI also includes a **Vector Store** tab where you can view indexed documents and remove them interactively.

## Project Structure

```
docling-rag-pipeline/
├── data/
│   ├── pdfs/           # Input: drop your PDF files here
│   ├── output/         # Generated markdown + extracted images
│   └── vectorstore/    # ChromaDB persistent storage
├── src/
│   ├── config.py       # Centralized configuration from .env
│   ├── convert.py      # PDF → Markdown via Docling + LM Studio vision
│   ├── ingest.py       # Markdown → chunks → embeddings → ChromaDB
│   ├── rag.py          # LangChain retrieval chain + LM Studio LLM
│   └── app.py          # Gradio web UI
├── run.py              # CLI entrypoint
├── requirements.txt
├── .env                # Configuration
└── README.md
```

## How It Works

1. **Convert** (`src/convert.py`): Docling processes each PDF — extracts text, tables, and images. For each image, it sends the image to LM Studio's Qwen3-VL model via the OpenAI-compatible API to generate a detailed description. The output is a markdown file with image descriptions inlined.

2. **Ingest** (`src/ingest.py`): Markdown files are split into chunks using header-aware splitting (preserves document structure), then further split by size. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in ChromaDB.

3. **Query** (`src/rag.py`): Questions are embedded and matched against the vector store. The top-k most relevant chunks are retrieved and passed as context to the LM Studio LLM, which generates a grounded answer with source attribution.

## Tips

- **Better image descriptions**: Use a larger vision model in LM Studio for more detailed descriptions. The prompt in `src/convert.py` can be customized.
- **Embedding quality**: Swap `EMBEDDING_MODEL` in `.env` to `BAAI/bge-small-en-v1.5` or `nomic-ai/nomic-embed-text-v1.5` for potentially better retrieval.
- **Separate models**: Load a dedicated text model in LM Studio for RAG responses (faster than using the vision model for text). Just change `CHAT_MODEL` in `.env`.
- **Re-indexing a file**: Remove it first (`python run.py remove file.md`), then ingest it again. Or use the Vector Store tab in the Gradio UI.
- **Full re-index**: To rebuild from scratch, delete `data/vectorstore/` and run `python run.py ingest`.
