# Docker Setup — Docling RAG Pipeline

This document explains how the Docker setup works, what it contains, and how to use it.

---

## What Docker Does Here

Docker packages the entire Docling RAG Pipeline (Python, all dependencies, your source code) into a **single container**. Instead of installing Python, creating a virtual environment, and pip installing everything, users just run one command and the app is ready.

**Your existing project is not modified.** The Docker files (`Dockerfile`, `docker-compose.yml`, `.dockerignore`) are added alongside your code — they don't change any source files.

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Your Host Machine                   │
│                                                  │
│   ┌─────────────┐       ┌─────────────────────┐  │
│   │  LM Studio  │◄──────│   Docker Container  │  │
│   │  port 1234  │       │                     │  │
│   │             │       │  • Python 3.12      │  │
│   │  - Vision   │       │  • Docling          │  │
│   │    model    │       │  • ChromaDB         │  │
│   │  - Chat     │       │  • Gradio (:7860)   │  │
│   │    model    │       │  • Sentence-        │  │
│   │             │       │    Transformers     │  │
│   └─────────────┘       │  • LangChain        │  │
│                         └──────────┬──────────┘  │
│                                    │             │
│   ┌────────────────────────────────┘             │
│   │  Volume Mounts (shared folders)              │
│   │                                              │
│   │  ./data/pdfs/        ← Drop PDFs here        │
│   │  ./data/output/      ← Converted .md files   │
│   │  ./data/vectorstore/ ← ChromaDB database     │
│   │  hf-cache (named)   ← Embedding model cache  │
│   │                                              │
└───┴──────────────────────────────────────────────┘
```

### What runs WHERE:

| Component | Where it runs | Notes |
|---|---|---|
| **LM Studio** | Your host machine | You install and run this yourself. Loads the vision model and chat model. |
| **Gradio web UI** | Inside the container | Accessible at `http://localhost:7860` in your browser. |
| **Docling** (PDF → Markdown) | Inside the container | Converts PDFs. Calls LM Studio on the host for image descriptions. |
| **ChromaDB** (vector store) | Inside the container | Data persists on your host via volume mount. |
| **Embedding model** (Alibaba GTE) | Inside the container | Downloads once from HuggingFace, cached in a named volume. Runs on CPU. |
| **Your data files** | Your host machine | Mounted into the container — changes are visible both ways instantly. |

---

## Files Added to Your Project

| File | Purpose |
|---|---|
| `Dockerfile` | Defines what goes inside the container — base image, system packages, Python packages, your code. |
| `docker-compose.yml` | Orchestration — port mapping, volume mounts, environment variables. This is what you run. |
| `.dockerignore` | Tells Docker to skip `venv/`, `__pycache__/`, `.git/`, `data/` when building. |
| `docker.env.example` | Example environment file users copy and edit for their setup. |

**No existing files are modified.**

---

## How It Works Step by Step

### Building the image (`docker compose build`)

1. Docker reads the `Dockerfile`
2. Starts from `python:3.12-slim` (minimal Debian + Python, ~150MB)
3. Installs system packages needed by docling (image processing libraries)
4. Creates a non-root user `appuser` (security best practice)
5. Copies `requirements.txt` and runs `pip install` (this layer is cached)
6. Copies your source code (`src/`, `run.py`)
7. Tags the resulting image

**First build: ~5-10 minutes. Subsequent rebuilds (code only): ~10 seconds.**

### Starting the container (`docker compose up`)

1. Docker creates a container from the image
2. Maps port 7860 (container) → 7860 (your host)
3. Mounts your `data/` directories as volumes
4. Sets environment variables (LM Studio URL, model names, etc.)
5. Runs `python run.py serve --port 7860`
6. Open `http://localhost:7860` in your browser

### Where your files go

| File type | Host location | Container location |
|---|---|---|
| Input PDFs | `./data/pdfs/` | `/app/data/pdfs/` |
| Converted markdown | `./data/output/` | `/app/data/output/` |
| ChromaDB data | `./data/vectorstore/` | `/app/data/vectorstore/` |
| Embedding model cache | Docker named volume `hf-cache` | `/home/appuser/.cache/huggingface/` |

Because these are mounted volumes:
- Drop a PDF into `data/pdfs/` on your host → the container sees it immediately
- Container converts a PDF → the `.md` file appears in `data/output/` on your host
- Stop/delete the container → all your data is still on your host, nothing lost

---

## The Embedding Model

The embedding model (`Alibaba-NLP/gte-large-en-v1.5`, ~400MB) is NOT included in the Docker image. It downloads from HuggingFace on first launch.

A named Docker volume (`hf-cache`) stores the download:
- **First launch:** downloads the model (~1-2 minutes)
- **Every subsequent launch:** model is already cached, starts instantly
- **Rebuilding the image:** model is still cached, does not re-download
- **Only deleted if** you explicitly run `docker compose down -v`

---

## Terminal Commands (CLI)

The Gradio UI covers everything, but if you prefer the terminal:

```bash
docker compose exec docling-rag python run.py list
docker compose exec docling-rag python run.py ingest "table1.md"
docker compose exec docling-rag python run.py remove "table1.md"
docker compose exec docling-rag python run.py query "What is a graph?"
```

**Tip — create a shell alias:**
```bash
alias rag='docker compose exec docling-rag python run.py'

rag list
rag ingest table1.md
rag query "What is a graph?"
```

---

## How LM Studio Connects

LM Studio runs on your host machine (not inside Docker). The container reaches it using:

- **`host.docker.internal`** — Docker's built-in hostname that points to the host machine
- On Linux, docker-compose adds `extra_hosts` to enable this
- On Windows/Mac (Docker Desktop), it works automatically

**LM Studio must be running with models loaded before you use the app.**

---

## When to Rebuild

| What you changed | What to do | How long |
|---|---|---|
| Source code (`src/`, `run.py`) | `docker compose up --build` | ~10 seconds |
| New Python package | `docker compose up --build` | ~5 minutes |
| `.env` values | `docker compose down && docker compose up` | Instant |
| PDFs or markdown files | Nothing — volumes are live | Instant |

---

## Common Commands

```bash
# Build and start
docker compose up --build

# Start (no rebuild)
docker compose up

# Start in background
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f

# Remove everything including cached models
docker compose down -v
```

---

## What Users Need to Install

1. **Docker Desktop** — [docker.com](https://www.docker.com/products/docker-desktop/)
2. **LM Studio** — [lmstudio.ai](https://lmstudio.ai) — load vision + chat models, start server
3. **This project** — clone the repo, run `docker compose up`

No Python install. No virtual environment. No pip.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Cannot connect to LM Studio | Make sure LM Studio is running with the server started and models loaded |
| Port 7860 already in use | Change the port in `docker-compose.yml`: `"8080:7860"` |
| Gradio not loading | Check `docker compose logs -f` — embedding model might be downloading |
| Code changes not reflected | Rebuild: `docker compose up --build` |
| Data disappeared | Only happens with `docker compose down -v` — don't use `-v` unless you want to wipe volumes |
