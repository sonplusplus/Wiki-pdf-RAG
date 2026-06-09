import hashlib
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(value: str, max_length: int = 80) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return (value or "document")[:max_length]


def stable_doc_id(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path.name
    rel_text = str(rel).replace("\\", "/").lower()
    suffix = hashlib.sha1(rel_text.encode("utf-8")).hexdigest()[:8]
    return f"{slugify(path.stem)}-{suffix}"


def stable_chunk_id(doc_id: str, page: int, index: int, text_hash: str) -> str:
    return f"{doc_id}:p{page:04d}:c{index:04d}:{text_hash[:12]}"


def stable_image_id(doc_id: str, page: int, index: int, image_hash: str) -> str:
    return f"{doc_id}:p{page:04d}:img{index:03d}:{image_hash[:12]}"
