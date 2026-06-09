from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .chunking import ChunkingConfig, chunk_page_text, guess_section, normalize_text
from .hashing import sha256_bytes, stable_image_id
from .models import ImageRecord, TextChunk


def _safe_suffix(name: str | None, data: bytes) -> str:
    if name:
        suffix = Path(name).suffix.lower()
        if suffix:
            return suffix
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    return ".bin"


def _extract_with_pypdf(
    *,
    page,
    doc_id: str,
    source_file: str,
    page_number: int,
    section: str,
    surrounding_text: str,
    output_dir: Path,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    images = getattr(page, "images", []) or []
    doc_image_dir = output_dir / doc_id
    doc_image_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, start=1):
        try:
            data = image.data
        except Exception:
            continue
        if not data:
            continue

        image_hash = sha256_bytes(data)
        image_id = stable_image_id(doc_id, page_number, index, image_hash)
        suffix = _safe_suffix(getattr(image, "name", None), data)
        image_path = doc_image_dir / f"page_{page_number:04d}_img_{index:03d}_{image_hash[:10]}{suffix}"
        image_path.write_bytes(data)
        records.append(
            ImageRecord(
                image_id=image_id,
                doc_id=doc_id,
                source_file=source_file,
                page=page_number,
                path=str(image_path),
                section=section,
                caption="",
                surrounding_text=surrounding_text,
                image_hash=image_hash,
            )
        )
    return records


def _render_page_snapshot_with_pymupdf(
    *,
    pdf_path: Path,
    doc_id: str,
    source_file: str,
    page_number: int,
    section: str,
    surrounding_text: str,
    output_dir: Path,
) -> list[ImageRecord]:
    try:
        import fitz  # type: ignore
    except Exception:
        return []

    doc_image_dir = output_dir / doc_id
    doc_image_dir.mkdir(parents=True, exist_ok=True)
    document = None
    try:
        document = fitz.open(str(pdf_path))
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        data = pixmap.tobytes("png")
    except Exception:
        return []
    finally:
        if document is not None:
            document.close()

    image_hash = sha256_bytes(data)
    image_id = stable_image_id(doc_id, page_number, 1, image_hash)
    image_path = doc_image_dir / f"page_{page_number:04d}_snapshot_{image_hash[:10]}.png"
    image_path.write_bytes(data)
    return [
        ImageRecord(
            image_id=image_id,
            doc_id=doc_id,
            source_file=source_file,
            page=page_number,
            path=str(image_path),
            section=section,
            caption="Rendered PDF page snapshot",
            surrounding_text=surrounding_text,
            image_hash=image_hash,
        )
    ]


def load_pdf(
    *,
    pdf_path: Path,
    doc_id: str,
    image_output_dir: Path,
    chunking: ChunkingConfig,
    render_page_snapshots: bool = False,
) -> tuple[list[TextChunk], list[ImageRecord]]:
    reader = PdfReader(str(pdf_path))
    source_file = pdf_path.name
    all_chunks: list[TextChunk] = []
    all_images: list[ImageRecord] = []
    current_section = "Introduction"

    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        page_text = normalize_text(raw_text)
        current_section = guess_section(current_section, page_text)
        chunks = chunk_page_text(
            doc_id=doc_id,
            source_file=source_file,
            page=page_index,
            section=current_section,
            text=page_text,
            config=chunking,
        )
        all_chunks.extend(chunks)

        surrounding_text = " ".join(page_text.split()[:120])
        image_records = _extract_with_pypdf(
            page=page,
            doc_id=doc_id,
            source_file=source_file,
            page_number=page_index,
            section=current_section,
            surrounding_text=surrounding_text,
            output_dir=image_output_dir,
        )
        if render_page_snapshots and not image_records:
            image_records = _render_page_snapshot_with_pymupdf(
                pdf_path=pdf_path,
                doc_id=doc_id,
                source_file=source_file,
                page_number=page_index,
                section=current_section,
                surrounding_text=surrounding_text,
                output_dir=image_output_dir,
            )
        all_images.extend(image_records)

    return all_chunks, all_images


def iter_pdfs(documents_dir: Path) -> Iterable[Path]:
    yield from sorted(documents_dir.rglob("*.pdf"))
