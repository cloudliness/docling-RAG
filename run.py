#!/usr/bin/env python3
"""CLI entrypoint for the Docling RAG Pipeline."""

import argparse
import logging
import sys


def cmd_convert(args):
    from src.convert import convert_all_pdfs

    paths = convert_all_pdfs()
    if paths:
        print(f"\nConverted {len(paths)} PDF(s):")
        for p in paths:
            print(f"  {p}")
    else:
        print("No PDFs were converted. Add PDF files to data/pdfs/")


def cmd_ingest(args):
    from src.ingest import ingest, list_indexed_sources
    from src.config import OUTPUT_DIR

    files = args.files if args.files else None

    # No explicit files: only ingest files not already in the vectorstore
    if files is None:
        already_indexed = set(list_indexed_sources().keys())
        available = sorted(OUTPUT_DIR.glob("*.md"))
        new_files = [f.name for f in available if f.name not in already_indexed]
        if not new_files:
            print("Nothing new to ingest — all files are already indexed.")
            return
        files = new_files
        print(f"Found {len(files)} new file(s) to ingest:")
        for f in files:
            print(f"  {f}")

    vectorstore = ingest(files=files)
    count = vectorstore._collection.count()
    print(f"\nIngestion complete. {count} chunks indexed.")


def cmd_list(args):
    from src.ingest import list_indexed_sources

    sources = list_indexed_sources()
    if not sources:
        print("Vector store is empty.")
        return
    print(f"\n{'Source File':<55} {'Chunks':>6}")
    print("-" * 63)
    for src, count in sorted(sources.items()):
        print(f"{src:<55} {count:>6}")
    total = sum(sources.values())
    print(f"\nTotal: {total} chunks from {len(sources)} file(s)")


def cmd_remove(args):
    from src.ingest import remove_source

    for source in args.sources:
        removed = remove_source(source)
        if removed:
            print(f"Removed {removed} chunks for '{source}'")
        else:
            print(f"No chunks found for '{source}'")


def cmd_query(args):
    from src.rag import query

    question = " ".join(args.question)
    result = query(question)

    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources ({len(result['sources'])}):")
    for s in result["sources"]:
        src = s["source"]
        section = s["section"]
        label = f"  - {src}"
        if section:
            label += f" > {section}"
        print(label)


def cmd_serve(args):
    from src.app import build_ui

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


def cmd_pipeline(args):
    from src.convert import convert_all_pdfs
    from src.ingest import ingest

    print("Step 1/2: Converting PDFs...")
    paths = convert_all_pdfs()
    if not paths:
        print("No PDFs found in data/pdfs/. Aborting.")
        sys.exit(1)
    print(f"  Converted {len(paths)} PDF(s)")

    print("\nStep 2/2: Ingesting into vector store...")
    converted_names = [p.name for p in paths]
    vectorstore = ingest(files=converted_names)
    count = vectorstore._collection.count()
    print(f"  Indexed {count} chunks")

    print("\nPipeline complete! Run 'python run.py serve' to start the web UI.")


def main():
    parser = argparse.ArgumentParser(
        description="Docling RAG Pipeline - PDF to RAG with LM Studio vision",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # convert
    subparsers.add_parser("convert", help="Convert PDFs to markdown via Docling")

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Chunk and index markdown into ChromaDB")
    ingest_parser.add_argument(
        "files", nargs="*", default=None,
        help="Specific .md files to ingest (default: all in data/output/)",
    )

    # list
    subparsers.add_parser("list", help="List documents currently in the vector store")

    # remove
    remove_parser = subparsers.add_parser("remove", help="Remove documents from the vector store")
    remove_parser.add_argument(
        "sources", nargs="+",
        help="Source filenames to remove (e.g. table1.md)",
    )

    # query
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("question", nargs="+", help="Your question")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch Gradio web UI")
    serve_parser.add_argument(
        "--port", type=int, default=7860, help="Port (default: 7860)"
    )
    serve_parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )

    # pipeline
    subparsers.add_parser(
        "pipeline", help="Run full pipeline: convert -> ingest"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "convert": cmd_convert,
        "ingest": cmd_ingest,
        "list": cmd_list,
        "remove": cmd_remove,
        "query": cmd_query,
        "serve": cmd_serve,
        "pipeline": cmd_pipeline,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
