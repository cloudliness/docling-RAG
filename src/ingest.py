import logging
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    OUTPUT_DIR,
    VECTORSTORE_DIR,
)

logger = logging.getLogger(__name__)


def load_markdown_files(output_dir: Path) -> list[Document]:
    """Load all markdown files from the output directory as LangChain Documents."""
    md_files = sorted(output_dir.glob("*.md"))

    if not md_files:
        logger.warning(f"No markdown files found in {output_dir}")
        return []

    documents = []
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        doc = Document(
            page_content=text,
            metadata={"source": md_path.name, "path": str(md_path)},
        )
        documents.append(doc)
        logger.info(f"Loaded: {md_path.name} ({len(text)} chars)")

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks using markdown-aware splitting."""

    # First: split on markdown headers to preserve document structure
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # Second: further split oversized chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        # Split by headers first
        header_splits = md_splitter.split_text(doc.page_content)

        # Carry forward source metadata
        for split in header_splits:
            split.metadata.update(doc.metadata)

        # Further split long sections
        final_chunks = text_splitter.split_documents(header_splits)
        all_chunks.extend(final_chunks)

    logger.info(
        f"Split {len(documents)} document(s) into {len(all_chunks)} chunks "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return all_chunks


_embeddings = None
_vectorstore = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model (cached)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True},
        )
    return _embeddings


def load_vectorstore() -> Chroma:
    """Load or return the cached ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(VECTORSTORE_DIR),
            embedding_function=get_embeddings(),
            collection_name="docling_rag",
        )
    return _vectorstore


def ingest(files: list[str] | None = None) -> Chroma:
    """Ingest markdown files into the vectorstore.

    Args:
        files: Specific .md filenames (relative to OUTPUT_DIR) or absolute
               paths.  When *None*, ingest every .md in OUTPUT_DIR.
    """
    if files:
        documents = []
        for f in files:
            path = Path(f) if Path(f).is_absolute() else OUTPUT_DIR / f
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            text = path.read_text(encoding="utf-8")
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": path.name, "path": str(path)},
                )
            )
            logger.info(f"Loaded: {path.name} ({len(text)} chars)")
    else:
        documents = load_markdown_files(OUTPUT_DIR)

    if not documents:
        raise FileNotFoundError(
            f"No markdown files found. Run 'convert' first."
        )

    chunks = chunk_documents(documents)

    # Remove existing chunks for these sources before adding (prevents duplicates)
    vectorstore = load_vectorstore()
    collection = vectorstore._collection
    sources_to_add = {doc.metadata["source"] for doc in documents}
    for source_name in sources_to_add:
        existing = collection.get(where={"source": source_name}, include=[])
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            logger.info(f"Replaced {len(existing['ids'])} existing chunks for '{source_name}'")

    vectorstore.add_documents(chunks)
    logger.info(
        f"Added {len(chunks)} chunks to ChromaDB at {VECTORSTORE_DIR}"
    )
    return vectorstore


def list_indexed_sources() -> dict[str, int]:
    """Return a mapping of source filename -> chunk count from the vectorstore."""
    vectorstore = load_vectorstore()
    collection = vectorstore._collection
    result = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in result["metadatas"]:
        src = meta.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def remove_source(source_name: str) -> int:
    """Remove all chunks for a given source file from the vectorstore.

    Returns the number of chunks removed.
    """
    vectorstore = load_vectorstore()
    collection = vectorstore._collection
    result = collection.get(where={"source": source_name}, include=[])
    ids = result["ids"]
    if ids:
        collection.delete(ids=ids)
    logger.info(f"Removed {len(ids)} chunks for '{source_name}'")
    return len(ids)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
