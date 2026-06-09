import re
from dataclasses import dataclass

from .hashing import sha256_text, stable_chunk_id
from .models import TextChunk


HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,()/-]{2,80}$")


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: tuple[str, ...] = (
        r"\n#{1,6}\s",
        r"```\n",
        r"\n\*\*\*+\n",
        r"\n---+\n",
        r"\n___+\n",
        r"\n\n",
        r"\n",
        r" ",
        "",
    )
    is_separator_regex: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def guess_section(current_section: str, page_text: str) -> str:
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line or len(line.split()) > 10:
            continue
        if HEADING_RE.match(line):
            return line
    return current_section or "Unknown section"


def _split_regex_keep_separator(text: str, pattern: str) -> list[str]:
    parts = re.split(f"({pattern})", text)
    chunks: list[str] = []
    pending_separator = ""

    for part in parts:
        if part == "":
            continue
        if re.fullmatch(pattern, part):
            pending_separator = part
            continue
        chunks.append(f"{pending_separator}{part}" if pending_separator else part)
        pending_separator = ""

    if pending_separator:
        chunks.append(pending_separator)
    return chunks


def _split_literal_keep_separator(text: str, separator: str) -> list[str]:
    parts = text.split(separator)
    chunks: list[str] = []
    for index, part in enumerate(parts):
        if not part:
            continue
        prefix = "" if index == 0 else separator
        chunks.append(prefix + part)
    return chunks


def _split_with_separator(text: str, separator: str, *, is_regex: bool) -> list[str]:
    if separator == "":
        return list(text)
    if is_regex:
        return _split_regex_keep_separator(text, separator)
    return _split_literal_keep_separator(text, separator)


def _recursive_split(text: str, config: ChunkingConfig, separator_index: int = 0) -> list[str]:
    if len(text) <= config.chunk_size:
        return [text]
    if separator_index >= len(config.separators):
        return [
            text[index : index + config.chunk_size]
            for index in range(0, len(text), config.chunk_size)
        ]

    separator = config.separators[separator_index]
    pieces = _split_with_separator(
        text,
        separator,
        is_regex=config.is_separator_regex,
    )
    if len(pieces) == 1 and separator != "":
        return _recursive_split(text, config, separator_index + 1)

    output: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= config.chunk_size:
            output.append(piece)
        else:
            output.extend(_recursive_split(piece, config, separator_index + 1))
    return output


def _head_on_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    window = text[:max_chars]
    min_cut = max(1, int(max_chars * 0.65))
    for index in range(len(window) - 1, min_cut, -1):
        if window[index].isspace():
            return window[:index].strip()
    return window.strip()


def _tail_on_boundary(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text.strip()

    start = len(text) - max_chars
    if start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        match = re.search(r"\s+", text[start:])
        if match:
            start += match.end()
    return text[start:].strip()


def _merge_splits(splits: list[str], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []
    current = ""

    for split in splits:
        split = split.strip()
        if not split:
            continue

        if current and re.match(r"^#{1,6}\s", split):
            chunks.append(current)
            current = split
            continue

        candidate = f"{current} {split}".strip() if current else split
        if len(candidate) <= config.chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            max_tail = max(0, config.chunk_size - len(split) - 1)
            tail_size = min(config.chunk_overlap, max_tail)
            tail = _tail_on_boundary(current, tail_size)
            current = f"{tail} {split}".strip() if tail else split
        else:
            current = split

        while len(current) > config.chunk_size:
            head = _head_on_boundary(current, config.chunk_size)
            chunks.append(head)
            remainder = current[len(head) :].strip()
            tail = _tail_on_boundary(head, config.chunk_overlap)
            current = f"{tail} {remainder}".strip() if tail else remainder

    if current:
        chunks.append(current)
    return chunks


def split_text(text: str, config: ChunkingConfig) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    splits = _recursive_split(clean, config)
    return _merge_splits(splits, config)


def chunk_page_text(
    *,
    doc_id: str,
    source_file: str,
    page: int,
    section: str,
    text: str,
    config: ChunkingConfig,
) -> list[TextChunk]:
    chunk_texts = split_text(text, config)
    if not chunk_texts:
        return []

    chunks: list[TextChunk] = []
    for index, chunk_text in enumerate(chunk_texts, start=1):
        text_hash = sha256_text(chunk_text)
        chunks.append(
            TextChunk(
                chunk_id=stable_chunk_id(doc_id, page, index, text_hash),
                doc_id=doc_id,
                source_file=source_file,
                page=page,
                section=section,
                text=chunk_text,
                text_hash=text_hash,
            )
        )
    return chunks
