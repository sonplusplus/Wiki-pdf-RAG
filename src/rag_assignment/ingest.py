from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SEPARATORS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_INDEX_WORKERS,
    EXTRACTED_IMAGES_DIR,
    REGISTRY_PATH,
    ensure_data_dirs,
)
from .chunking import ChunkingConfig
from .hashing import sha256_file, stable_doc_id
from .models import ImageRecord, TextChunk
from .pdf_loader import iter_pdfs, load_pdf
from .registry import DocumentRegistry
from .vector_store import FaissFtsVectorStore, build_vector_store, get_embedding_provider


@dataclass
class IngestStats:
    scanned: int = 0
    skipped: int = 0
    indexed: int = 0
    chunks: int = 0
    images: int = 0


@dataclass
class PendingDocument:
    doc_id: str
    source_path: str
    file_hash: str
    chunks: list[TextChunk]
    images: list[ImageRecord]


class Ingestor:
    def __init__(
        self,
        *,
        documents_dir: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        render_page_snapshots: bool = False,
        workers: int = DEFAULT_INDEX_WORKERS,
    ):
        ensure_data_dirs()
        self.documents_dir = documents_dir
        self.chunking = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=tuple(DEFAULT_CHUNK_SEPARATORS),
        )
        self.index_config = {
            "chunking_version": "word-boundary-overlap-v1",
            "chunk_size": self.chunking.chunk_size,
            "chunk_overlap": self.chunking.chunk_overlap,
            "separators": list(self.chunking.separators),
            "render_page_snapshots": render_page_snapshots,
        }
        self.render_page_snapshots = render_page_snapshots
        self.workers = max(1, workers)
        self.embedding = get_embedding_provider()
        self.registry = DocumentRegistry(REGISTRY_PATH)
        self.text_store = build_vector_store("text", self.embedding)
        self.image_store = build_vector_store("image", self.embedding)

    def ingest_all(self, *, prune_missing: bool = False) -> IngestStats:
        stats = IngestStats()
        seen_doc_ids: set[str] = set()
        pending_documents: list[PendingDocument] = []
        pending_pdf_info: list[tuple[Path, str, str]] = []

        for pdf_path in iter_pdfs(self.documents_dir):
            stats.scanned += 1
            doc_id = stable_doc_id(pdf_path, self.documents_dir)
            seen_doc_ids.add(doc_id)
            file_hash = sha256_file(pdf_path)

            registry_record = self.registry.get(doc_id)
            expected_chunks = 0 if not registry_record else registry_record.get("chunk_count", 0)
            expected_images = 0 if not registry_record else registry_record.get("image_count", 0)
            vectors_present = (
                self.text_store.count_by_doc_id(doc_id) >= expected_chunks
                and self.image_store.count_by_doc_id(doc_id) >= expected_images
            )

            if (
                self.registry.is_unchanged(
                    doc_id,
                    file_hash,
                    self.embedding.name,
                    self.index_config,
                )
                and vectors_present
            ):
                stats.skipped += 1
                continue

            pending_pdf_info.append((pdf_path, doc_id, file_hash))

        if pending_pdf_info:
            for _, doc_id, _ in pending_pdf_info:
                self.text_store.delete_by_doc_id(doc_id)
                self.image_store.delete_by_doc_id(doc_id)

            if self.workers == 1:
                for pdf_path, doc_id, file_hash in pending_pdf_info:
                    document = self._load_pending_document(pdf_path, doc_id, file_hash)
                    pending_documents.append(document)
                    stats.indexed += 1
                    stats.chunks += len(document.chunks)
                    stats.images += len(document.images)
            else:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    futures = {
                        executor.submit(
                            self._load_pending_document,
                            pdf_path,
                            doc_id,
                            file_hash,
                        ): (pdf_path, doc_id)
                        for pdf_path, doc_id, file_hash in pending_pdf_info
                    }
                    for future in as_completed(futures):
                        document = future.result()
                        pending_documents.append(document)
                        stats.indexed += 1
                        stats.chunks += len(document.chunks)
                        stats.images += len(document.images)

        if pending_documents:
            text_inputs = [
                {
                    "id": chunk.chunk_id,
                    "content": chunk.text,
                    "metadata": chunk.to_metadata(),
                }
                for document in pending_documents
                for chunk in document.chunks
            ]
            image_inputs = [
                {
                    "id": image.image_id,
                    "content": image.searchable_text,
                    "metadata": image.to_metadata(),
                }
                for document in pending_documents
                for image in document.images
            ]

            self._embed_and_upsert_batches(self.text_store, text_inputs)
            self._embed_and_upsert_batches(self.image_store, image_inputs)

        for document in pending_documents:
            self.registry.upsert(
                doc_id=document.doc_id,
                source_path=document.source_path,
                file_hash=document.file_hash,
                embedding_name=self.embedding.name,
                chunk_ids=[chunk.chunk_id for chunk in document.chunks],
                image_ids=[image.image_id for image in document.images],
                index_config=self.index_config,
            )

        if prune_missing:
            self._prune_missing(seen_doc_ids)

        return stats

    def _load_pending_document(
        self,
        pdf_path: Path,
        doc_id: str,
        file_hash: str,
    ) -> PendingDocument:
        chunks, images = load_pdf(
            pdf_path=pdf_path,
            doc_id=doc_id,
            image_output_dir=EXTRACTED_IMAGES_DIR,
            chunking=self.chunking,
            render_page_snapshots=self.render_page_snapshots,
        )
        return PendingDocument(
            doc_id=doc_id,
            source_path=str(pdf_path),
            file_hash=file_hash,
            chunks=chunks,
            images=images,
        )

    def _embed_and_upsert_batches(
        self,
        store: FaissFtsVectorStore,
        records: list[dict[str, object]],
        batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    ) -> None:
        embedded_records = []
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            embedded_records.extend(store.make_records(batch))
        store.upsert_many(embedded_records)

    def _prune_missing(self, seen_doc_ids: set[str]) -> None:
        for doc_id in sorted(self.registry.known_doc_ids() - seen_doc_ids):
            self.text_store.delete_by_doc_id(doc_id)
            self.image_store.delete_by_doc_id(doc_id)
            self.registry.remove(doc_id)
