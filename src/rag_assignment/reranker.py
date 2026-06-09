import os
from typing import Any

from .config import RERANKER_ENABLED, STRICT_MODE


class OptionalCrossEncoderReranker:
    def __init__(self) -> None:
        self.enabled = RERANKER_ENABLED
        self.model_name = os.getenv(
            "RAG_RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        self.model = None
        if not self.enabled:
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)
        except Exception:
            if STRICT_MODE:
                raise
            self.enabled = False

    def rerank(
        self,
        *,
        question: str,
        hits: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not self.enabled or self.model is None or not hits:
            return hits[:top_k]

        pairs = [(question, hit["content"]) for hit in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked: list[dict[str, Any]] = []
        for hit, rerank_score in zip(hits, scores):
            reranked.append(
                {
                    **hit,
                    "base_score": hit.get("score", 0.0),
                    "score": float(rerank_score),
                    "reranker": self.model_name,
                }
            )
        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]
