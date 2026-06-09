import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Protocol

from .config import (
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_RETRIEVAL_CANDIDATES,
    EMBEDDING_PROVIDER,
    HYBRID_VECTOR_WEIGHT,
    IMAGE_FAISS_PATH,
    IMAGE_SQLITE_PATH,
    TEXT_FAISS_PATH,
    TEXT_SQLITE_PATH,
    effective_keyword_weights,
)


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9+-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, text: str) -> list[float]:
        ...

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install requirements.txt."
            ) from exc

        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self.batch_size = DEFAULT_EMBED_BATCH_SIZE
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors.tolist()]


def get_embedding_provider() -> EmbeddingProvider:
    provider = EMBEDDING_PROVIDER.strip().lower()
    if provider in {"sentence-transformers", "sentence_transformers", "sbert"}:
        model_name = os.getenv(
            "RAG_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        return SentenceTransformerEmbedding(model_name)
    raise ValueError(f"Unsupported embedding provider: {provider}")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _query_terms(query: str) -> list[str]:
    return [token for token in tokenize(query) if token not in STOPWORDS]


def _metadata_text(metadata: dict[str, Any]) -> str:
    return " ".join(
        str(metadata.get(key, ""))
        for key in ("doc_id", "source_file", "page", "section", "caption")
    )


def _keyword_score(query: str, content: str, metadata: dict[str, Any]) -> float:
    terms = set(_query_terms(query))
    if not terms:
        return 0.0
    searchable = f"{_metadata_text(metadata)} {content}".lower()
    searchable_tokens = set(tokenize(searchable))
    overlap = len(terms & searchable_tokens) / len(terms)
    compact_query = re.sub(r"[^a-z0-9]+", "", " ".join(terms))
    compact_metadata = re.sub(r"[^a-z0-9]+", "", _metadata_text(metadata).lower())
    title_boost = 0.5 if compact_query and compact_query in compact_metadata else 0.0
    return overlap + title_boost


def _fts_query(query: str) -> str:
    terms = _query_terms(query)
    if not terms:
        return ""
    # Quoted terms avoid FTS syntax issues for values like USB-C.
    return " OR ".join(f'"{term}"' for term in terms)


class FaissFtsVectorStore:
    def __init__(self, *, faiss_path: Path, sqlite_path: Path, embedding: EmbeddingProvider):
        self.faiss_path = faiss_path
        self.sqlite_path = sqlite_path
        self.embedding = embedding
        self.faiss_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _import_faiss(self):
        try:
            import faiss  # type: ignore
            import numpy as np
        except Exception as exc:
            raise RuntimeError("faiss-cpu is not installed. Install requirements.txt.") from exc
        return faiss, np

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_sqlite(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY,
                    faiss_id INTEGER NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    metadata_text TEXT NOT NULL,
                    embedding_name TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS records_fts
                USING fts5(id UNINDEXED, content, metadata_text)
                """
            )

    def _load_index(self):
        faiss, _ = self._import_faiss()
        if not self.faiss_path.exists():
            return None
        return faiss.read_index(self.faiss_path.resolve().as_posix())

    def _write_index(self, index) -> None:
        faiss, _ = self._import_faiss()
        temp_path = self.faiss_path.with_suffix(self.faiss_path.suffix + ".tmp")
        faiss.write_index(index, temp_path.resolve().as_posix())
        temp_path.replace(self.faiss_path)

    def _new_index(self, dimension: int):
        faiss, _ = self._import_faiss()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    def _faiss_id(self, record_id: str) -> int:
        return int(hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:15], 16)

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        metadata = json.loads(row["metadata_json"])
        return {
            "id": row["id"],
            "faiss_id": int(row["faiss_id"]),
            "content": row["content"],
            "metadata": metadata,
            "embedding_name": row["embedding_name"],
        }

    def make_record(
        self,
        *,
        record_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self.make_records(
            [{"id": record_id, "content": content, "metadata": metadata}]
        )[0]

    def make_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not records:
            return []
        vectors = self.embedding.embed_many([record["content"] for record in records])
        output: list[dict[str, Any]] = []
        for record, vector in zip(records, vectors):
            output.append(
                {
                    "id": record["id"],
                    "faiss_id": self._faiss_id(record["id"]),
                    "content": record["content"],
                    "metadata": record["metadata"],
                    "metadata_text": _metadata_text(record["metadata"]),
                    "embedding_name": self.embedding.name,
                    "vector": vector,
                }
            )
        return output

    def upsert_many(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        faiss, np = self._import_faiss()
        vectors = np.asarray([record["vector"] for record in records], dtype="float32")
        if vectors.ndim != 2:
            raise RuntimeError("FAISS expects a 2D dense vector matrix.")

        index = self._load_index()
        if index is None:
            index = self._new_index(vectors.shape[1])
        elif index.d != vectors.shape[1]:
            raise RuntimeError(
                "Existing FAISS index dimension differs from current embedding model. "
                "Delete FAISS/SQLite index files and rebuild."
            )

        faiss_ids = np.asarray([record["faiss_id"] for record in records], dtype="int64")
        index.remove_ids(faiss_ids)
        index.add_with_ids(vectors, faiss_ids)

        with self._connect() as connection:
            for record in records:
                connection.execute("DELETE FROM records WHERE id = ?", (record["id"],))
                connection.execute("DELETE FROM records_fts WHERE id = ?", (record["id"],))
                connection.execute(
                    """
                    INSERT INTO records (
                        id, faiss_id, content, metadata_json, metadata_text, embedding_name
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        record["faiss_id"],
                        record["content"],
                        json.dumps(record["metadata"], ensure_ascii=False),
                        record["metadata_text"],
                        record["embedding_name"],
                    ),
                )
                connection.execute(
                    "INSERT INTO records_fts (id, content, metadata_text) VALUES (?, ?, ?)",
                    (record["id"], record["content"], record["metadata_text"]),
                )

        self._write_index(index)

    def delete_by_doc_id(self, doc_id: str) -> None:
        faiss, np = self._import_faiss()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, faiss_id FROM records WHERE json_extract(metadata_json, '$.doc_id') = ?",
                (doc_id,),
            ).fetchall()
            faiss_ids = [int(row["faiss_id"]) for row in rows]
            record_ids = [row["id"] for row in rows]
            for record_id in record_ids:
                connection.execute("DELETE FROM records_fts WHERE id = ?", (record_id,))
            connection.execute(
                "DELETE FROM records WHERE json_extract(metadata_json, '$.doc_id') = ?",
                (doc_id,),
            )

        index = self._load_index()
        if index is not None and faiss_ids:
            index.remove_ids(np.asarray(faiss_ids, dtype="int64"))
            self._write_index(index)

    def count_by_doc_id(self, doc_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM records WHERE json_extract(metadata_json, '$.doc_id') = ?",
                (doc_id,),
            ).fetchone()
        return int(row["count"] if row else 0)

    def _records_by_faiss_ids(self, faiss_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not faiss_ids:
            return {}
        placeholders = ",".join("?" for _ in faiss_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM records WHERE faiss_id IN ({placeholders})",
                faiss_ids,
            ).fetchall()
        return {int(row["faiss_id"]): self._row_to_record(row) for row in rows}

    def _fts_candidates(self, query: str, limit: int) -> dict[str, tuple[dict[str, Any], float]]:
        match_query = _fts_query(query)
        if not match_query:
            return {}
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.*, bm25(records_fts) AS rank
                FROM records_fts
                JOIN records r ON r.id = records_fts.id
                WHERE records_fts MATCH ?
                  AND r.embedding_name = ?
                ORDER BY rank
                LIMIT ?
                """,
                (match_query, self.embedding.name, limit),
            ).fetchall()

        candidates: dict[str, tuple[dict[str, Any], float]] = {}
        for rank, row in enumerate(rows, start=1):
            record = self._row_to_record(row)
            candidates[record["id"]] = (record, 1.0 / rank)
        return candidates

    def _vector_candidates(
        self,
        query: str,
        limit: int,
    ) -> dict[str, tuple[dict[str, Any], float, float]]:
        faiss, np = self._import_faiss()
        index = self._load_index()
        if index is None or index.ntotal == 0:
            return {}

        candidate_k = min(index.ntotal, max(limit, DEFAULT_RETRIEVAL_CANDIDATES))
        query_vector = np.asarray([self.embedding.embed(query)], dtype="float32")
        distances, ids = index.search(query_vector, candidate_k)

        candidates: dict[str, tuple[dict[str, Any], float, float]] = {}
        faiss_ids = [int(value) for value in ids[0].tolist() if int(value) >= 0]
        records_by_faiss_id = self._records_by_faiss_ids(faiss_ids)
        for rank, (raw_vector_score, faiss_id) in enumerate(
            zip(distances[0].tolist(), faiss_ids),
            start=1,
        ):
            record = records_by_faiss_id.get(faiss_id)
            if record and record["embedding_name"] == self.embedding.name:
                candidates[record["id"]] = (
                    record,
                    1.0 / rank,
                    float(raw_vector_score),
                )
        return candidates

    def semantic_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for record, vector_rank_score, raw_vector_score in self._vector_candidates(query, top_k).values():
            output.append(
                {
                    **record,
                    "score": vector_rank_score,
                    "vector_score": float(raw_vector_score),
                    "vector_rank_score": vector_rank_score,
                    "bm25_score": 0.0,
                    "lexical_score": _keyword_score(query, record["content"], record["metadata"]),
                }
            )
            if len(output) >= top_k:
                break
        return output

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        candidates = self._fts_candidates(query, top_k)
        scored: list[dict[str, Any]] = []
        for record, bm25_rank_score in candidates.values():
            scored.append(
                {
                    **record,
                    "score": bm25_rank_score,
                    "vector_score": 0.0,
                    "vector_rank_score": 0.0,
                    "bm25_score": bm25_rank_score,
                    "lexical_score": _keyword_score(query, record["content"], record["metadata"]),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def debug_search_components(self, query: str, top_k: int = 5) -> dict[str, list[dict[str, Any]]]:
        return {
            "semantic": self.semantic_search(query, top_k=top_k),
            "bm25": self.keyword_search(query, top_k=top_k),
            "hybrid": self.search(query, top_k=top_k),
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector_candidates = self._vector_candidates(query, top_k)
        candidate_k = max(top_k, DEFAULT_RETRIEVAL_CANDIDATES)
        fts_candidates = self._fts_candidates(query, candidate_k)
        all_ids = set(vector_candidates) | set(fts_candidates)

        scored: list[dict[str, Any]] = []
        bm25_weight, keyword_weight = effective_keyword_weights()
        for record_id in all_ids:
            if record_id in vector_candidates:
                record, vector_rank_score, raw_vector_score = vector_candidates[record_id]
            else:
                record = fts_candidates[record_id][0]
                vector_rank_score = 0.0
                raw_vector_score = 0.0
            bm25_rank_score = fts_candidates.get(record_id, (record, 0.0))[1]
            keyword_score = _keyword_score(query, record["content"], record["metadata"])
            score = (
                HYBRID_VECTOR_WEIGHT * vector_rank_score
                + bm25_weight * bm25_rank_score
                + keyword_weight * keyword_score
            )
            if score <= 0:
                continue
            scored.append(
                {
                    **record,
                    "score": score,
                    "vector_score": raw_vector_score,
                    "vector_rank_score": vector_rank_score,
                    "bm25_score": bm25_rank_score,
                    "lexical_score": keyword_score,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def build_vector_store(role: str, embedding: EmbeddingProvider) -> FaissFtsVectorStore:
    role = role.strip().lower()
    if role == "text":
        return FaissFtsVectorStore(
            faiss_path=TEXT_FAISS_PATH,
            sqlite_path=TEXT_SQLITE_PATH,
            embedding=embedding,
        )
    if role == "image":
        return FaissFtsVectorStore(
            faiss_path=IMAGE_FAISS_PATH,
            sqlite_path=IMAGE_SQLITE_PATH,
            embedding=embedding,
        )
    raise ValueError(f"Unknown vector store role: {role}")
