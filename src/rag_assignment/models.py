from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    source_file: str
    page: int
    section: str
    text: str
    text_hash: str

    def to_metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text")
        return data


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    doc_id: str
    source_file: str
    page: int
    path: str
    section: str
    caption: str
    surrounding_text: str
    image_hash: str

    @property
    def searchable_text(self) -> str:
        parts = [
            self.source_file,
            f"page {self.page}",
            self.section,
            self.caption,
            self.surrounding_text,
        ]
        return "\n".join(part for part in parts if part)

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)
