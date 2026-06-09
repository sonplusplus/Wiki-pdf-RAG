import re
from typing import Any

from .config import (
    DEBUG_RETRIEVAL_DEFAULT,
    DEFAULT_COMPARE_ENTITY_CANDIDATES,
    DEFAULT_IMAGE_TOP_K,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
)
from .query_planner import QueryPlan, plan_query
from .reranker import OptionalCrossEncoderReranker
from .vector_store import build_vector_store, get_embedding_provider


class Retriever:
    def __init__(self):
        embedding = get_embedding_provider()
        self.text_store = build_vector_store("text", embedding)
        self.image_store = build_vector_store("image", embedding)
        self.reranker = OptionalCrossEncoderReranker()

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        debug: bool = DEBUG_RETRIEVAL_DEFAULT,
    ) -> dict[str, Any]:
        plan = plan_query(question)
        text_hits = self._retrieve_text(plan, top_k=top_k)
        image_hits = []
        if plan.image_query:
            image_hits = self.image_store.search(plan.image_query, top_k=DEFAULT_IMAGE_TOP_K)
        retrieval: dict[str, Any] = {"plan": plan, "text_hits": text_hits, "image_hits": image_hits}
        if debug:
            retrieval["debug"] = {
                query: self.text_store.debug_search_components(query, top_k=top_k)
                for query in plan.text_queries
            }
        return retrieval

    def _retrieve_text(self, plan: QueryPlan, top_k: int) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        entity_hits: dict[str, list[dict[str, Any]]] = {}
        candidate_k = max(DEFAULT_RERANK_CANDIDATES, top_k)
        for query in plan.text_queries:
            for hit in self.text_store.search(query, top_k=candidate_k):
                existing = by_id.get(hit["id"])
                if not existing or hit["score"] > existing["score"]:
                    by_id[hit["id"]] = hit

        if plan.question_type == "compare":
            for entity in plan.entities:
                hits = self.text_store.search(entity, top_k=DEFAULT_COMPARE_ENTITY_CANDIDATES)
                hits = sorted(
                    hits,
                    key=lambda item: (
                        self._entity_source_score(item, entity),
                        self._entity_page_bonus(item, entity),
                        item.get("score", 0),
                    ),
                    reverse=True,
                )
                entity_hits[entity] = hits
                for hit in hits:
                    existing = by_id.get(hit["id"])
                    if not existing or hit["score"] > existing["score"]:
                        by_id[hit["id"]] = hit

        hits = sorted(by_id.values(), key=lambda item: item["score"], reverse=True)[
            :candidate_k
        ]
        reranked = self.reranker.rerank(
            question=" ".join(plan.text_queries),
            hits=hits,
            top_k=top_k,
        )
        if plan.question_type == "compare":
            return self._ensure_entity_coverage(reranked, entity_hits, top_k)
        if plan.question_type == "summary" and plan.entities:
            return self._ensure_summary_overview(reranked, plan.entities[0], top_k)
        return reranked

    def _ensure_entity_coverage(
        self,
        hits: list[dict[str, Any]],
        entity_hits: dict[str, list[dict[str, Any]]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not entity_hits:
            return hits

        selected = list(hits)
        selected_ids = {hit["id"] for hit in selected}
        for entity, candidates in entity_hits.items():
            for candidate in candidates:
                if candidate["id"] in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["id"])
                break

        selected.sort(key=lambda item: item.get("score", 0), reverse=True)

        required: list[dict[str, Any]] = []
        required_ids: set[str] = set()
        for entity, candidates in entity_hits.items():
            for candidate in candidates:
                if candidate["id"] in selected_ids and candidate["id"] not in required_ids:
                    required.append(candidate)
                    required_ids.add(candidate["id"])
                    break

        output: list[dict[str, Any]] = []
        for hit in required + selected:
            if hit["id"] in {item["id"] for item in output}:
                continue
            if hit["id"] not in required_ids and self._is_indirect_compare_hit(hit, entity_hits):
                continue
            output.append(hit)
            if len(output) >= top_k:
                break
        return output

    def _ensure_summary_overview(
        self,
        hits: list[dict[str, Any]],
        topic: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates = self.text_store.search(topic, top_k=DEFAULT_COMPARE_ENTITY_CANDIDATES)
        candidates = sorted(
            candidates,
            key=lambda item: (
                self._entity_source_score(item, topic),
                self._entity_page_bonus(item, topic),
                item.get("score", 0),
            ),
            reverse=True,
        )
        if not candidates:
            return hits

        best = candidates[0]
        output = [best]
        for pool in (
            [hit for hit in hits if not self._is_low_value_section(hit)],
            hits,
        ):
            for hit in pool:
                if hit["id"] in {item["id"] for item in output}:
                    continue
                output.append(hit)
                if len(output) >= top_k:
                    return output
        return output

    def _is_low_value_section(self, hit: dict[str, Any]) -> bool:
        section = str(hit.get("metadata", {}).get("section", "")).lower()
        return any(term in section for term in ("references", "external links", "portal"))

    def _hit_mentions_entity(self, hit: dict[str, Any], entity: str) -> bool:
        haystack = " ".join(
            [
                hit.get("content", ""),
                str(hit.get("metadata", {}).get("source_file", "")),
                str(hit.get("metadata", {}).get("section", "")),
            ]
        ).lower()
        haystack_tokens = {
            self._normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", haystack)
        }
        tokens = [
            self._normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", entity.lower())
            if len(token) > 1
        ]
        if not tokens:
            return False
        return all(token in haystack_tokens for token in tokens)

    def _entity_source_score(self, hit: dict[str, Any], entity: str) -> int:
        source = str(hit.get("metadata", {}).get("source_file", "")).lower()
        source_tokens = {self._normalize_token(token) for token in re.findall(r"[a-z0-9]+", source)}
        entity_tokens = [
            self._normalize_token(token)
            for token in re.findall(r"[a-z0-9]+", entity.lower())
            if len(token) > 1
        ]
        if entity_tokens and all(token in source_tokens for token in entity_tokens):
            return 3
        if self._hit_mentions_entity(hit, entity):
            return 1
        return 0

    def _is_indirect_compare_hit(
        self,
        hit: dict[str, Any],
        entity_hits: dict[str, list[dict[str, Any]]],
    ) -> bool:
        for entity, candidates in entity_hits.items():
            has_direct_candidate = any(self._entity_source_score(candidate, entity) >= 3 for candidate in candidates)
            if not has_direct_candidate:
                continue
            if self._hit_mentions_entity(hit, entity) and self._entity_source_score(hit, entity) < 3:
                return True
        return False

    def _normalize_token(self, token: str) -> str:
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

    def _entity_page_bonus(self, hit: dict[str, Any], entity: str) -> int:
        if self._entity_source_score(hit, entity) < 3:
            return 0
        page = hit.get("metadata", {}).get("page")
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            return 0
        if page_num == 1:
            return 4
        if page_num <= 3:
            return 2
        if page_num >= 30:
            return -2
        return 0
