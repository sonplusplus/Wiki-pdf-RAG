import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"documents": {}}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        self.data.setdefault("documents", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self.data["documents"].get(doc_id)

    def is_unchanged(
        self,
        doc_id: str,
        file_hash: str,
        embedding_name: str,
        index_config: dict[str, Any] | None = None,
    ) -> bool:
        record = self.get(doc_id)
        if not record:
            return False
        unchanged = (
            record.get("file_hash") == file_hash
            and record.get("embedding_name") == embedding_name
        )
        if index_config is not None:
            unchanged = unchanged and record.get("index_config") == index_config
        return unchanged

    def upsert(
        self,
        *,
        doc_id: str,
        source_path: str,
        file_hash: str,
        embedding_name: str,
        chunk_ids: list[str],
        image_ids: list[str],
        index_config: dict[str, Any] | None = None,
    ) -> None:
        self.data["documents"][doc_id] = {
            "doc_id": doc_id,
            "source_path": source_path,
            "file_hash": file_hash,
            "embedding_name": embedding_name,
            "index_config": index_config,
            "chunk_count": len(chunk_ids),
            "image_count": len(image_ids),
            "chunk_ids": chunk_ids,
            "image_ids": image_ids,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def remove(self, doc_id: str) -> dict[str, Any] | None:
        record = self.data["documents"].pop(doc_id, None)
        self.save()
        return record

    def known_doc_ids(self) -> set[str]:
        return set(self.data["documents"])
