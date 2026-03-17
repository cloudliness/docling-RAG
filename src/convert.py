import logging
from io import BytesIO
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.io import DocumentStream

from src.config import (
    LMSTUDIO_BASE_URL,
    OUTPUT_DIR,
    PDF_DIR,
    VISION_MODEL,
)

logger = logging.getLogger(__name__)


def build_converter() -> DocumentConverter:
    """Create a Docling DocumentConverter configured to use LM Studio for image descriptions."""

    # Use the simple API-based picture description that talks directly to
    # LM Studio's OpenAI-compatible /v1/chat/completions endpoint.
    picture_description_options = PictureDescriptionApiOptions(
        url=f"{LMSTUDIO_BASE_URL}/chat/completions",
        params={"model": VISION_MODEL},
        prompt="Describe this image in detail. Include all visible text, diagrams, charts, and visual elements.",
        timeout=120.0,
        scale=2.0,
        picture_area_threshold=0.0,  # Describe all images regardless of size
    )

    pipeline_options = PdfPipelineOptions(
        do_picture_description=True,
        picture_description_options=picture_description_options,
        generate_picture_images=True,
        images_scale=2.0,
        enable_remote_services=True,
    )

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    return converter


def convert_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Convert a single PDF to markdown with image descriptions.

    Returns the path to the generated markdown file.
    """
    converter = build_converter()

    # Read the PDF and strip any bytes before the %PDF header.
    # Some PDFs have leading garbage that causes docling's format
    # sniffer to misidentify them (e.g. as audio).
    data = pdf_path.read_bytes()
    pdf_marker = data.find(b"%PDF")
    if pdf_marker < 0:
        raise ValueError(f"Not a valid PDF file: {pdf_path.name}")
    if pdf_marker > 0:
        logger.warning(
            f"{pdf_path.name}: stripping {pdf_marker} bytes before %PDF header"
        )
        data = data[pdf_marker:]

    stream = DocumentStream(name=pdf_path.name, stream=BytesIO(data))

    logger.info(f"Converting: {pdf_path.name}")
    result = converter.convert(stream)
    doc = result.document

    # Export to markdown (image descriptions are inlined automatically)
    markdown = doc.export_to_markdown()

    # Write markdown file
    md_path = output_dir / f"{pdf_path.stem}.md"
    md_path.write_text(markdown, encoding="utf-8")
    logger.info(f"Saved: {md_path}")

    # Save extracted images
    images_dir = output_dir / f"{pdf_path.stem}_images"
    for i, picture in enumerate(doc.pictures):
        if picture.image is not None:
            images_dir.mkdir(parents=True, exist_ok=True)
            img_path = images_dir / f"image_{i}.png"
            picture.image.pil_image.save(str(img_path))

    return md_path


def convert_all_pdfs() -> list[Path]:
    """Convert all PDFs in the input directory to markdown.

    Returns list of paths to generated markdown files.
    """
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {PDF_DIR}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) to convert")
    md_paths = []

    for pdf_path in pdf_files:
        try:
            md_path = convert_pdf(pdf_path, OUTPUT_DIR)
            md_paths.append(md_path)
        except Exception:
            logger.exception(f"Failed to convert {pdf_path.name}")

    logger.info(f"Converted {len(md_paths)}/{len(pdf_files)} PDFs")
    return md_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    convert_all_pdfs()
