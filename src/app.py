import logging
import traceback

import gradio as gr
from pathlib import Path
import shutil

from src.ingest import ingest, load_vectorstore, list_indexed_sources, remove_source
from src.convert import convert_pdf
from src.rag import create_rag_chain
from src.config import PDF_DIR, OUTPUT_DIR, VECTORSTORE_DIR, CHAT_MODEL, VISION_MODEL

# Tracks which .md files were produced by the last convert run,
# so the pipeline can ingest only those instead of everything.
_last_converted_md: list[str] = []

logger = logging.getLogger(__name__)

# Module-level chain (lazy-loaded)
_rag_chain = None
_retriever = None


def get_chain():
    global _rag_chain, _retriever
    if _rag_chain is None:
        _rag_chain, _retriever = create_rag_chain()
    return _rag_chain, _retriever


def chat_fn(message: dict, history: list) -> str:
    """Handle a chat message using the RAG chain."""
    text = message if isinstance(message, str) else message.get("text", "")
    if not text.strip():
        return "Please enter a question."

    try:
        chain, retriever = get_chain()
    except Exception as e:
        return (
            f"Error loading vector store: {e}\n\n"
            "Make sure you've run **Convert** and **Ingest** first."
        )

    try:
        answer = chain.invoke(text)

        # Get sources for attribution
        source_docs = retriever.invoke(text)
        sources = set()
        for doc in source_docs:
            src = doc.metadata.get("source", "unknown")
            section = doc.metadata.get("header_1", "")
            label = src
            if section:
                label += f" > {section}"
            sources.add(label)

        if sources:
            answer += "\n\n---\n**Sources:** " + ", ".join(sorted(sources))

        return answer

    except Exception as e:
        return f"Error generating response: {e}"


def run_convert() -> str:
    """Convert all PDFs in the input directory."""
    global _last_converted_md
    _last_converted_md = []
    try:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))
        pdf_count = len(pdf_files)
        if pdf_count == 0:
            return f"No PDF files found in `{PDF_DIR}`. Add PDFs and try again."

        print(f"[convert] Found {pdf_count} PDF(s)")
        md_paths = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"[convert]   [{i}/{pdf_count}] {pdf_path.name} ...")
            try:
                md_path = convert_pdf(pdf_path, OUTPUT_DIR)
                md_paths.append(md_path)
                print(f"[convert]   OK -> {md_path.name}")
            except Exception as e:
                print(f"[convert]   FAILED: {e}")
                logger.exception(f"Failed to convert {pdf_path.name}")

        _last_converted_md = [p.name for p in md_paths]
        return (
            f"Converted {len(md_paths)}/{pdf_count} PDFs to markdown.\n"
            f"Output directory: `{OUTPUT_DIR}`"
        )
    except Exception as e:
        print(f"[convert] ERROR: {e}")
        return f"Conversion error: {e}"


def run_ingest(files: list[str] | None = None) -> str:
    """Run the ingestion pipeline.

    Args:
        files: Specific .md filenames to ingest.  When *None*, only files
               not yet in the vectorstore are ingested.
    """
    global _rag_chain, _retriever
    try:
        # If no explicit file list, figure out which files are new
        if files is None:
            already_indexed = set(list_indexed_sources().keys())
            available = sorted(OUTPUT_DIR.glob("*.md"))
            new_files = [f.name for f in available if f.name not in already_indexed]
            if not new_files:
                return "Nothing new to ingest — all files are already indexed."
            files = new_files
            print(f"[ingest] Found {len(files)} new file(s) to ingest")

        print(f"[ingest] Ingesting: {', '.join(files)}")
        vectorstore = ingest(files=files)
        _rag_chain = None
        _retriever = None

        count = vectorstore._collection.count()
        print(f"[ingest] OK - indexed {count} chunks")
        return (
            f"Ingested {len(files)} file(s). Total: **{count}** chunks.\n"
            f"Vector store: `{VECTORSTORE_DIR}`"
        )
    except FileNotFoundError as e:
        print(f"[ingest] FileNotFoundError: {e}")
        return str(e)
    except Exception as e:
        print(f"[ingest] ERROR: {e}")
        return f"Ingestion error: {e}"


def _save_uploads(files) -> tuple[int, list[str]]:
    """Copy uploaded files into PDF_DIR. Returns (count_saved, filenames)."""
    if not files:
        return 0, []

    saved = 0
    names: list[str] = []
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    for f in files:
        try:
            src = Path(str(f))
            print(f"[upload] raw={f!r}  path={src}  exists={src.exists()}  suffix={src.suffix}")

            if not src.exists():
                print(f"[upload] SKIP (file not found): {src}")
                continue

            # Ensure destination keeps a .pdf extension
            dest_name = src.name
            if not dest_name.lower().endswith(".pdf"):
                dest_name = src.stem + ".pdf"

            dest = PDF_DIR / dest_name
            shutil.copy2(src, dest)
            saved += 1
            names.append(dest_name)
            print(f"[upload] OK -> {dest}")
        except Exception as e:
            print(f"[upload] ERROR: {e}")
            logger.exception("Failed to save uploaded file")

    print(f"[upload] Saved {saved} file(s)")
    return saved, names


def run_pipeline_with_upload(files) -> str:
    """Save any uploaded PDFs, then convert -> ingest. Returns full output."""
    try:
        print(f"\n{'='*60}")
        print(f"[pipeline] Button clicked  |  files={files!r}")
        print(f"{'='*60}")

        # -- Save uploads --
        saved, saved_names = _save_uploads(files)
        if saved:
            progress = f"Uploaded {saved} file(s): {', '.join(saved_names)}\n\n"
        else:
            progress = ""

        # -- Step 1: Convert --
        progress += "Step 1 - Convert\n"
        convert_result = run_convert()
        progress += convert_result + "\n\n"

        if "No PDF" in convert_result:
            return progress

        # -- Step 2: Ingest only the just-converted files --
        progress += "Step 2 - Ingest & Index\n"
        ingest_result = run_ingest(files=_last_converted_md if _last_converted_md else None)
        progress += ingest_result + "\n\n"

        progress += "Pipeline complete."
        print("[pipeline] Done")
        return progress

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[pipeline] UNHANDLED ERROR:\n{tb}")
        return f"Pipeline error:\n{e}\n\nCheck the terminal for details."


def upload_and_run(files) -> str:
    """Upload & Ingest button handler."""
    return run_pipeline_with_upload(files)


def get_status():
    """Get current pipeline status."""
    pdf_count = len(list(PDF_DIR.glob("*.pdf")))
    md_count = len(list(OUTPUT_DIR.glob("*.md")))

    try:
        vs = load_vectorstore()
        chunk_count = vs._collection.count()
    except Exception:
        chunk_count = 0

    return (
        f"**PDFs in input folder:** {pdf_count}\n"
        f"**Markdown files generated:** {md_count}\n"
        f"**Chunks in vector store:** {chunk_count}\n"
        f"**Vision model:** {VISION_MODEL}\n"
        f"**Chat model:** {CHAT_MODEL}"
    )


def get_indexed_files_table():
    """Return indexed sources as a list of [filename, chunks] for the Dataframe."""
    try:
        sources = list_indexed_sources()
    except Exception:
        return []
    if not sources:
        return []
    rows = [[src, count] for src, count in sorted(sources.items())]
    return rows


def handle_remove_sources(selected_sources: list[str]) -> tuple:
    """Remove selected sources from the vectorstore."""
    global _rag_chain, _retriever
    if not selected_sources:
        return "No files selected for removal.", get_indexed_files_table(), gr.CheckboxGroup(choices=list(list_indexed_sources().keys()), value=[])

    names = selected_sources
    results = []
    for name in names:
        try:
            removed = remove_source(name)
            if removed:
                results.append(f"Removed {removed} chunks for '{name}'")
            else:
                results.append(f"No chunks found for '{name}'")
        except Exception as e:
            results.append(f"Error removing '{name}': {e}")

    _rag_chain = None
    _retriever = None
    updated_choices = list(list_indexed_sources().keys())
    return "\n".join(results), get_indexed_files_table(), gr.CheckboxGroup(choices=updated_choices, value=[])


def handle_ingest_selected(selected_files: list[str]) -> tuple[str, list]:
    """Ingest selected markdown files into the vectorstore."""
    global _rag_chain, _retriever
    if not selected_files:
        return "No files selected for ingestion.", get_indexed_files_table()

    try:
        vectorstore = ingest(files=selected_files)
        _rag_chain = None
        _retriever = None
        count = vectorstore._collection.count()
        return (
            f"Ingested {len(selected_files)} file(s). Total chunks: {count}",
            get_indexed_files_table(),
        )
    except Exception as e:
        return f"Ingestion error: {e}", get_indexed_files_table()


def get_available_md_files():
    """Return list of .md filenames in data/output/ for the checkbox group."""
    files = sorted(OUTPUT_DIR.glob("*.md"))
    return [f.name for f in files]


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Docling RAG Pipeline",
    ) as app:
        gr.Markdown("# Docling RAG Pipeline")
        gr.Markdown(
            "PDF → Docling (+ LM Studio vision) → Markdown → ChromaDB → RAG"
        )

        with gr.Tabs():
            # Chat tab
            with gr.Tab("Chat"):
                gr.ChatInterface(
                    fn=chat_fn,
                    title="Ask questions about your documents",
                    examples=[
                        "What are the main topics covered in the documents?",
                        "Summarize the key findings.",
                        "What images or figures are described in the documents?",
                    ],
                )

            # Pipeline tab
            with gr.Tab("Pipeline"):
                gr.Markdown("### Pipeline Controls")

                status_display = gr.Markdown(value=get_status)

                # File uploader for PDFs
                uploader = gr.File(
                    label="Upload PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath",
                )

                with gr.Row():
                    upload_btn = gr.Button("Upload & Ingest", variant="secondary")
                    convert_btn = gr.Button(
                        "1. Convert PDFs", variant="secondary"
                    )
                    ingest_btn = gr.Button(
                        "2. Ingest & Index", variant="secondary"
                    )

                pipeline_btn = gr.Button(
                    "Run Full Pipeline", variant="primary", size="lg"
                )

                refresh_btn = gr.Button("Refresh Status", variant="secondary")

                result_display = gr.Textbox(
                    label="Pipeline Output",
                    lines=12,
                    interactive=False,
                )

                upload_btn.click(fn=upload_and_run, inputs=uploader, outputs=result_display).then(
                    fn=get_status, outputs=status_display
                )

                convert_btn.click(fn=run_convert, outputs=result_display).then(
                    fn=get_status, outputs=status_display
                )
                ingest_btn.click(fn=run_ingest, outputs=result_display).then(
                    fn=get_status, outputs=status_display
                )
                pipeline_btn.click(
                    fn=run_pipeline_with_upload, inputs=uploader, outputs=result_display
                ).then(fn=get_status, outputs=status_display)
                refresh_btn.click(fn=get_status, outputs=status_display)

                gr.Markdown(
                    f"**PDF input folder:** `{PDF_DIR}`\n\n"
                    "Upload PDFs above or drop them in the folder, then click "
                    "**Run Full Pipeline**.\n\n"
                    "*Conversion can take a while — check the terminal for live progress.*"
                )

            # Vector Store tab
            with gr.Tab("Vector Store"):
                gr.Markdown("### Indexed Documents")
                gr.Markdown(
                    "View and manage documents in the vector store. "
                    "Type filenames in the box below (one per line) and click **Remove Selected** to delete them."
                )

                indexed_table = gr.Dataframe(
                    headers=["Source File", "Chunks"],
                    datatype=["str", "number"],
                    value=get_indexed_files_table(),
                    interactive=False,
                    label="Documents in Vector Store",
                )

                remove_picker = gr.CheckboxGroup(
                    choices=list(list_indexed_sources().keys()),
                    label="Select files to remove",
                )

                with gr.Row():
                    remove_btn = gr.Button("Remove Selected", variant="stop")
                    vs_refresh_btn = gr.Button("Refresh", variant="secondary")

                vs_result = gr.Textbox(
                    label="Result",
                    lines=4,
                    interactive=False,
                )

                remove_btn.click(
                    fn=handle_remove_sources,
                    inputs=remove_picker,
                    outputs=[vs_result, indexed_table, remove_picker],
                )

                def refresh_vs_tab():
                    updated_choices = list(list_indexed_sources().keys())
                    return get_indexed_files_table(), gr.CheckboxGroup(choices=updated_choices, value=[])

                vs_refresh_btn.click(
                    fn=refresh_vs_tab,
                    outputs=[indexed_table, remove_picker],
                )

                gr.Markdown("---")
                gr.Markdown("### Selective Ingestion")
                gr.Markdown(
                    "Choose specific markdown files from `data/output/` to add to the vector store."
                )

                md_file_picker = gr.CheckboxGroup(
                    choices=get_available_md_files(),
                    label="Available Markdown Files",
                )

                with gr.Row():
                    ingest_selected_btn = gr.Button("Ingest Selected", variant="primary")
                    picker_refresh_btn = gr.Button("Refresh File List", variant="secondary")

                ingest_result = gr.Textbox(
                    label="Ingestion Result",
                    lines=3,
                    interactive=False,
                )

                ingest_selected_btn.click(
                    fn=handle_ingest_selected,
                    inputs=md_file_picker,
                    outputs=[ingest_result, indexed_table],
                )
                picker_refresh_btn.click(
                    fn=lambda: gr.CheckboxGroup(choices=get_available_md_files()),
                    outputs=md_file_picker,
                )

        # Refresh vectorstore data on every page load (not just at server start)
        def on_page_load():
            table = get_indexed_files_table()
            vs_choices = list(list_indexed_sources().keys())
            md_choices = get_available_md_files()
            return table, gr.CheckboxGroup(choices=vs_choices, value=[]), gr.CheckboxGroup(choices=md_choices)

        app.load(
            fn=on_page_load,
            outputs=[indexed_table, remove_picker, md_file_picker],
        )

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = build_ui()
    app.launch(theme=gr.themes.Soft())
