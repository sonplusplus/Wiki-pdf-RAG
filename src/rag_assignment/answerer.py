import json
import importlib.util
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import (
    LLM_PROVIDER,
    MIN_COMPARE_TEXT_SCORE,
    MIN_IMAGE_SCORE,
    MIN_TEXT_SCORE,
    OLLAMA_HOST,
    STRICT_MODE,
)


@dataclass(frozen=True)
class EvidenceItem:
    text: str
    metadata: dict[str, Any]
    score: float


def _clean_inline_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"^([A-Za-z0-9][A-Za-z0-9+-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+-]*){0,3})\s+\1\b",
        r"\1",
        text,
        flags=re.I,
    )
    return text.strip(" -")


def _quote(text: str, max_chars: int = 360) -> str:
    text = _clean_inline_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _citation(metadata: dict[str, Any]) -> str:
    section = metadata.get("section") or "Unknown section"
    return f"{metadata.get('source_file')} | page {metadata.get('page')} | {section}"


QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "compare",
    "difference",
    "differences",
    "do",
    "does",
    "explain",
    "give",
    "how",
    "in",
    "is",
    "key",
    "main",
    "me",
    "of",
    "short",
    "summarize",
    "summary",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}

EVIDENCE_LABELS_RE = re.compile(
    r"\b("
    r"Developer|Manufacturer|Product\s?family|Type|Series|First released|Released|"
    r"Availability by region|Introductory\s?price|Discontinued|Operating system|"
    r"System on a chip|Memory|Storage|Display|Connectivity|Dimensions|Weight|"
    r"Industry|Compatible hardware|Physical range|Website|Predecessor|Successor"
    r")\b",
)

DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s+\d{4}\b",
    re.I,
)

LEAK_PHRASES = (
    "only the cited points should be treated",
    "retrieved materials contain relevant evidence",
    "fallback mode does not infer",
    "the application will add citations",
)


def _question_terms(question: str) -> list[str]:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9+-]*", question.lower())
    return [term for term in terms if term not in QUESTION_STOPWORDS and len(term) > 1]


def _entity_covered(entity: str, hits: list[dict[str, Any]]) -> bool:
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", entity.lower())
        if len(token) > 1
    ]
    if not tokens:
        return False
    haystack = " ".join(
        " ".join(
            [
                hit.get("content", ""),
                str(hit.get("metadata", {}).get("source_file", "")),
                str(hit.get("metadata", {}).get("section", "")),
            ]
        )
        for hit in hits
    ).lower()
    return all(token in haystack for token in tokens)


def _source_covers_entity(entity: str, hit: dict[str, Any]) -> bool:
    source = str(hit.get("metadata", {}).get("source_file", "")).lower()
    source_tokens = {_normalize_token(token) for token in re.findall(r"[A-Za-z0-9]+", source)}
    entity_tokens = [
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", entity.lower())
        if len(token) > 1
    ]
    return bool(entity_tokens) and all(token in source_tokens for token in entity_tokens)


def _normalize_token(token: str) -> str:
    token = token.lower()
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _sentences(text: str) -> list[str]:
    clean = _clean_inline_text(text)
    if not clean:
        return []
    parts: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        sentence = sentence.strip()
        if not sentence:
            continue
        labeled = EVIDENCE_LABELS_RE.sub(r"|| \1", sentence)
        parts.extend(part.strip(" |") for part in labeled.split("||"))
    return [_clean_inline_text(part) for part in parts if len(part.strip()) > 20]


def _sentence_score(sentence: str, terms: list[str]) -> int:
    lower = sentence.lower()
    return sum(1 for term in terms if term in lower)


def _best_definition_sentence(question: str, hits: list[dict[str, Any]]) -> str | None:
    terms = _question_terms(question)
    if not terms:
        return None
    subject = max(terms, key=len)
    definition_re = re.compile(
        rf"\b{re.escape(subject)}\b.*\b(is|are|refers to|means|consists of)\b",
        re.I,
    )
    candidates: list[tuple[int, str]] = []
    for hit in hits:
        for sentence in _sentences(hit["content"]):
            if not definition_re.search(sentence):
                continue
            sentence = _trim_to_definition_start(sentence, subject)
            score = _sentence_score(sentence, terms)
            if hit.get("metadata", {}).get("page") == 1:
                score += 1
            candidates.append((score, sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _trim_to_definition_start(sentence: str, subject: str) -> str:
    lower = sentence.lower()
    subject_lower = subject.lower()
    positions = [match.start() for match in re.finditer(re.escape(subject_lower), lower)]
    if not positions:
        return sentence

    best_position = positions[0]
    best_distance = 10_000
    for position in positions:
        suffix = lower[position : position + 180]
        keyword_match = re.search(r"\b(is|are|refers to|means|consists of)\b", suffix)
        if keyword_match and keyword_match.start() < best_distance:
            best_distance = keyword_match.start()
            best_position = position

    trimmed = sentence[best_position:].strip(" ,;:-")
    trimmed = re.sub(r"^(USB[-\u2011]C)\s+USB[-\u2011]C,\s+or\s+", r"\1, or ", trimmed, flags=re.I)
    return re.sub(rf"^({re.escape(subject)})\s+\1\b", subject, trimmed, flags=re.I)


def _best_key_sentences(
    hits: list[dict[str, Any]],
    terms: list[str],
    limit: int = 3,
) -> list[str]:
    seen: set[str] = set()
    scored: list[tuple[int, float, str]] = []
    for hit in hits:
        for sentence in _sentences(hit["content"]):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            score = _sentence_score(sentence, terms)
            if score > 0:
                scored.append((score, float(hit.get("score", 0)), sentence))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [sentence for _, _, sentence in scored[:limit]]


def _question_subject(question: str) -> str:
    value = re.sub(
        r"^(what|when|where|who|why|how|is|are|do|does|did|was|were|can)\b",
        "",
        question.strip(),
        flags=re.I,
    )
    value = re.sub(
        r"\b(classified as|released|the|a|an|in|provided materials|is|are|was|were|do|does|did|can)\b",
        " ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\?+$", "", value)
    return re.sub(r"\s+", " ", value).strip(" .,:;?")


def _answer_terms(text: str | None) -> list[str]:
    if not text:
        return []
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9+-]*", text.lower())
    stopwords = QUESTION_STOPWORDS | {
        "answer",
        "based",
        "citation",
        "citations",
        "confidence",
        "evidence",
        "final",
        "found",
        "materials",
        "provided",
        "retrieved",
        "source",
        "supported",
    }
    return [term for term in terms if term not in stopwords and len(term) > 1]


def _is_prompt_leak(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in LEAK_PHRASES)


def _usable_evidence_text(text: str) -> bool:
    lower = text.lower()
    if _is_prompt_leak(text):
        return False
    if "retrieved from http" in lower or lower.startswith(("references ", "external links ")):
        return False
    return len(re.findall(r"[A-Za-z0-9]", text)) >= 20


def _dedupe_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:180]


def _evidence_candidates(
    question: str,
    hits: list[dict[str, Any]],
    *,
    final_answer: str | None = None,
) -> list[EvidenceItem]:
    terms = _question_terms(question)
    support_terms = terms + _answer_terms(final_answer)
    subject = _question_subject(question).lower()
    wants_date = bool(re.search(r"\b(when|released|launch|launched)\b", question, re.I))
    wants_definition = bool(re.search(r"\b(what is|what are|define|classified as|is .* a |is .* an )", question, re.I))

    seen: set[str] = set()
    candidates: list[EvidenceItem] = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        page = metadata.get("page")
        section = str(metadata.get("section", "")).lower()
        for sentence in _sentences(hit.get("content", "")):
            if terms and not wants_date:
                sentence = _trim_to_definition_start(sentence, terms[0])
            sentence = _quote(sentence, 300)
            if not _usable_evidence_text(sentence):
                continue
            key = _dedupe_key(sentence)
            if key in seen:
                continue
            seen.add(key)

            lower = sentence.lower()
            score = float(_sentence_score(sentence, support_terms))
            if subject and all(token in lower for token in re.findall(r"[a-z0-9]+", subject) if len(token) > 1):
                score += 3
            if wants_date and DATE_RE.search(sentence):
                score += 4
            if wants_date and re.search(r"\b(released|launched|unveiled|introduced)\b", lower):
                score += 2
            if wants_definition and re.search(r"\b(is|are|refers to|means|classified as)\b", lower):
                score += 3
            if page == 1:
                score += 1
            if any(term in section for term in ("references", "external links", "portal")):
                score -= 4
            score += min(1.0, float(hit.get("score", 0)) / 10.0)

            if score > 0:
                candidates.append(EvidenceItem(text=sentence, metadata=metadata, score=score))

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _select_evidence(
    question: str,
    plan: Any,
    hits: list[dict[str, Any]],
    *,
    final_answer: str | None = None,
    limit: int = 3,
) -> list[EvidenceItem]:
    candidates = _evidence_candidates(question, hits, final_answer=final_answer)
    if not candidates:
        return [
            EvidenceItem(text=_quote(hit.get("content", ""), 260), metadata=hit.get("metadata", {}), score=0.0)
            for hit in hits[:limit]
        ]

    if getattr(plan, "question_type", None) == "compare":
        selected: list[EvidenceItem] = []
        selected_keys: set[str] = set()
        for entity in getattr(plan, "entities", []) or []:
            entity_tokens = [
                _normalize_token(token)
                for token in re.findall(r"[A-Za-z0-9]+", entity.lower())
                if len(token) > 1
            ]
            for item in candidates:
                haystack_tokens = {
                    _normalize_token(token)
                    for token in re.findall(
                        r"[A-Za-z0-9]+",
                        f"{item.text} {item.metadata.get('source_file', '')} {item.metadata.get('section', '')}".lower(),
                    )
                }
                if entity_tokens and all(token in haystack_tokens for token in entity_tokens):
                    key = _dedupe_key(item.text)
                    if key not in selected_keys:
                        selected.append(item)
                        selected_keys.add(key)
                    break
        for item in candidates:
            if len(selected) >= limit:
                break
            key = _dedupe_key(item.text)
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)
        return selected[:limit]

    if re.search(r"\b(when|released|launch|launched|introduced)\b", question, re.I):
        dated_release = [item for item in candidates if _best_release_date(item.text)]
        if dated_release:
            return dated_release[:limit]

    return candidates[:limit]


def _best_release_date(text: str) -> str | None:
    date_match = DATE_RE.search(text)
    if not date_match:
        return None
    before = text[: date_match.start()]
    if re.search(r"\b(released|launched)\b", before, re.I):
        return date_match.group(0)
    after = text[date_match.end() : date_match.end() + 80]
    if re.search(r"\b(release|released|launch|launched)\b", after, re.I):
        return date_match.group(0)
    return None


def _yes_no_target_terms(question: str) -> list[str]:
    patterns = [
        r"\bclassified as\s+(?:a|an|the)?\s*(.+?)(?:\s+in\b|\s+according\b|\?|$)",
        r"\b(?:is|are|was|were)\s+.+?\s+(?:a|an|the)\s+(.+?)(?:\s+in\b|\s+according\b|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.I)
        if not match:
            continue
        terms = [
            _normalize_token(token)
            for token in re.findall(r"[A-Za-z0-9]+", match.group(1).lower())
            if token not in QUESTION_STOPWORDS and len(token) > 1
        ]
        if terms:
            return terms
    return []


def _contains_terms(text: str, terms: list[str]) -> bool:
    haystack = {
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
    }
    return all(term in haystack for term in terms)


class GroundedAnswerer:
    def __init__(self, retriever: Any | None = None):
        self.retriever = retriever
        self._compare_sub_summary_hits: list[dict[str, Any]] = []

    def answer(self, question: str, retrieval: dict[str, Any]) -> str:
        plan = retrieval["plan"]
        text_hits = retrieval["text_hits"]
        image_hits = retrieval["image_hits"]

        if plan.question_type == "image":
            return self._answer_image(image_hits)

        min_score = MIN_COMPARE_TEXT_SCORE if plan.question_type == "compare" else MIN_TEXT_SCORE
        supported_hits = [hit for hit in text_hits if hit.get("score", 0) >= min_score]
        if not supported_hits:
            return self._not_found()

        return self._grounded_answer(question, plan, supported_hits)

    def _answer_image(self, image_hits: list[dict[str, Any]]) -> str:
        supported_images = [hit for hit in image_hits if hit.get("score", 0) >= MIN_IMAGE_SCORE]
        if not supported_images:
            return self._not_found()

        lines = [
            "## Final Answer",
            "Found image candidate(s) from the provided PDF materials.",
            "",
            "## Evidence / Citations",
        ]
        for index, hit in enumerate(supported_images, start=1):
            metadata = hit["metadata"]
            lines.append(
                f"{index}. `{metadata.get('path')}` - {_citation(metadata)} "
                f"(image metadata score: {hit['score']:.3f})"
            )
            if metadata.get("surrounding_text"):
                lines.append(f"   Quote/context: \"{_quote(metadata['surrounding_text'], 220)}\"")

        confidence = min(95, 65 + int(supported_images[0]["score"] * 100))
        lines.extend(
            [
                "",
                "## Confidence",
                f"Confidence: {confidence}% - Image metadata matched the request and points back to source PDF pages.",
                "",
                "## Missing Information Handling",
                "Information was found in the provided materials.",
            ]
        )
        return "\n".join(lines)

    def _grounded_answer(
        self,
        question: str,
        plan: Any,
        hits: list[dict[str, Any]],
    ) -> str:
        question_type = plan.question_type
        final_answer: str | None = None
        if question_type == "compare":
            self._compare_sub_summary_hits = []
            entities = getattr(plan, "entities", [])
            if self.retriever and len(entities) >= 2:
                final_answer = self._compare_via_sub_summaries(question, entities)
            else:
                final_answer = "\n".join(self._fallback_compare_answer(question, hits, entities))
        elif self._ollama_enabled():
            generated = self._try_ollama_answer(question, question_type, hits)
            if generated:
                final_answer = self._extract_final_answer(generated)

        if final_answer is None and self._openai_available():
            generated = self._try_openai_answer(question, question_type, hits)
            if generated:
                final_answer = self._extract_final_answer(generated)

        if not final_answer:
            final_answer = "\n".join(self._fallback_final_answer(question, question_type, hits, plan))

        final_answer = self._normalize_final_answer(question, question_type, final_answer)
        if self._bad_final_answer(question_type, final_answer):
            final_answer = "\n".join(self._fallback_final_answer(question, question_type, hits, plan))
        response_hits = hits
        if question_type == "compare" and self._compare_sub_summary_hits:
            response_hits = self._merge_hits(self._compare_sub_summary_hits, hits)
        evidence = _select_evidence(question, plan, response_hits, final_answer=final_answer)
        return self._format_text_response(final_answer, question, response_hits, plan, evidence)

    def _merge_hits(
        self,
        primary_hits: list[dict[str, Any]],
        secondary_hits: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for hit in primary_hits + secondary_hits:
            metadata = hit.get("metadata", {})
            hit_key = str(
                hit.get("id")
                or (
                    metadata.get("source_file"),
                    metadata.get("page"),
                    metadata.get("section"),
                    _dedupe_key(hit.get("content", "")),
                )
            )
            if hit_key in seen:
                continue
            merged.append(hit)
            seen.add(hit_key)
        return merged

    def _format_text_response(
        self,
        final_answer: str,
        question: str,
        hits: list[dict[str, Any]],
        plan: Any,
        evidence: list[EvidenceItem],
    ) -> str:
        lines = ["## Final Answer", final_answer.strip()]

        lines.extend(["", "## Evidence / Citations"])
        for index, item in enumerate(evidence, start=1):
            lines.append(f"{index}. \"{_quote(item.text)}\"")
            lines.append(f"   Source: {_citation(item.metadata)}")

        confidence, reason = self._confidence(question, hits, plan)
        lines.extend(
            [
                "",
                "## Confidence",
                f"Confidence: {confidence}% - {reason}",
                "",
                "## Missing Information Handling",
                "Information was found in the provided materials.",
            ]
        )
        return "\n".join(lines)

    def _extract_final_answer(self, generated: str) -> str:
        match = re.search(
            r"##\s*Final Answer\s*(.*?)(?=\n##\s*Evidence / Citations|\n##\s*Confidence|\n##\s*Missing Information Handling|\Z)",
            generated,
            flags=re.I | re.S,
        )
        if match:
            return match.group(1).strip()
        return generated.strip()

    def _normalize_final_answer(
        self,
        question: str,
        question_type: str,
        final_answer: str,
    ) -> str:
        final_answer = self._strip_prompt_leaks(final_answer)
        final_answer = re.sub(r"\n{3,}", "\n\n", final_answer).strip()
        if question_type != "yes_no":
            return final_answer

        lower = final_answer.lower()
        if lower.startswith(("yes", "no", "not found")):
            return final_answer
        if any(term in lower for term in (" is classified ", " is a ", " is an ", " are classified ")):
            return f"Yes. {final_answer}"
        if any(term in lower for term in (" is not ", " are not ", " not classified ")):
            return f"No. {final_answer}"
        return final_answer

    def _strip_prompt_leaks(self, final_answer: str) -> str:
        lines = []
        for line in final_answer.splitlines():
            if _is_prompt_leak(line):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def _bad_final_answer(self, question_type: str, final_answer: str) -> bool:
        compact = re.sub(r"[\s.!,;:]+", " ", final_answer).strip().lower()
        if not compact:
            return True
        if _is_prompt_leak(compact):
            return True
        if question_type != "yes_no" and compact in {"yes", "no", "yes it is", "no it is not"}:
            return True
        if question_type != "yes_no" and compact.startswith(("yes ", "no ", "yes no")):
            return True
        if question_type in {"compare", "summary", "complex_lookup"} and len(final_answer.split()) < 8:
            return True
        if "not found in provided materials" in compact:
            return False
        return False

    def _confidence(self, question: str, hits: list[dict[str, Any]], plan: Any | None = None) -> tuple[int, str]:
        if not hits:
            return 0, "Retrieval did not return supporting evidence."

        terms = set(_question_terms(question))
        quoted_text = " ".join(hit.get("content", "") for hit in hits[:3]).lower()
        term_hits = sum(1 for term in terms if term in quoted_text)
        coverage = term_hits / max(1, len(terms))
        source_count = len({hit.get("metadata", {}).get("source_file") for hit in hits[:3]})

        confidence = 70 + int(coverage * 20) + min(5, source_count)
        confidence = max(60, min(95, confidence))

        if plan and getattr(plan, "question_type", None) == "compare":
            entities = getattr(plan, "entities", []) or []
            covered_entities = sum(
                1
                for entity in entities
                if _entity_covered(entity, hits[:3])
            )
            if entities and covered_entities < len(entities):
                return min(confidence, 65), "Retrieved evidence does not cover every compared item."
            if entities:
                confidence = min(88, confidence + 2)
                return confidence, "Retrieved evidence covers each compared item, but direct comparison is limited to extracted evidence."

        if coverage >= 0.75:
            return confidence, "Retrieved evidence directly matches the main question terms and includes source page metadata."
        return confidence, "Retrieved evidence is relevant but only partially covers the question terms."

    def _fallback_final_answer(
        self,
        question: str,
        question_type: str,
        hits: list[dict[str, Any]],
        plan: Any | None = None,
    ) -> list[str]:
        terms = _question_terms(question)
        plan = plan or type("Plan", (), {"question_type": question_type, "entities": []})()
        evidence = _select_evidence(question, plan, hits, limit=4)

        if question_type == "yes_no":
            return self._fallback_yes_no_answer(question, evidence)

        if question_type == "simple_lookup":
            date_answer = self._fallback_date_answer(question, evidence)
            if date_answer:
                return [date_answer]
            definition = _best_definition_sentence(question, hits)
            if definition:
                return [_quote(definition, 700)]

        if question_type == "compare":
            return self._fallback_compare_answer(question, hits, getattr(plan, "entities", []))

        if question_type == "summary":
            return self._fallback_summary_answer(question, hits)

        key_sentences = [item.text for item in evidence] or _best_key_sentences(hits, terms, limit=4)
        lead = {
            "complex_lookup": "The supported details are:",
        }.get(question_type, "The supported answer is:")
        if key_sentences:
            return [lead, *[f"- {_quote(sentence, 300)}" for sentence in key_sentences[:4]]]

        return [
            "Relevant evidence was found, but no concise synthesized statement could be extracted.",
            *[f"- {_quote(hit['content'], 260)}" for hit in hits[:3]],
        ]

    def _fallback_date_answer(self, question: str, evidence: list[EvidenceItem]) -> str | None:
        if not re.search(r"\b(when|released|launch|launched|introduced)\b", question, re.I):
            return None
        subject = _question_subject(question)
        for item in evidence:
            date = _best_release_date(item.text)
            if date:
                if subject:
                    return f"{subject} was released on {date}."
                return f"It was released on {date}."
        return None

    def _fallback_yes_no_answer(self, question: str, evidence: list[EvidenceItem]) -> list[str]:
        if not evidence:
            return ["Not found in provided materials."]
        best = evidence[0].text
        target_terms = _yes_no_target_terms(question)
        if target_terms and not _contains_terms(best, target_terms):
            return ["Not found in provided materials."]
        if re.search(r"\b(is|are|was|were|do|does|did|can)\b", question, re.I):
            return [f"Yes. {_quote(best, 360)}"]
        return [_quote(best, 360)]

    def _fallback_summary_answer(self, question: str, hits: list[dict[str, Any]]) -> list[str]:
        plan = type("Plan", (), {"question_type": "summary", "entities": []})()
        evidence = _select_evidence(question, plan, hits, limit=4)
        overview = _best_definition_sentence(question, hits)
        if not overview and hits:
            sentences = _sentences(hits[0]["content"])
            if sentences:
                overview = sentences[0]

        lines = ["Key supported points:"]
        seen: set[str] = set()
        if overview:
            text = _quote(overview, 320)
            lines.append(f"- {text}")
            seen.add(_dedupe_key(text))
        for item in evidence:
            key = _dedupe_key(item.text)
            if key in seen:
                continue
            lines.append(f"- {_quote(item.text, 280)}")
            seen.add(key)
            if len(lines) >= 4:
                break
        return lines

    def _compare_via_sub_summaries(self, question: str, entities: list[str]) -> str:
        self._compare_sub_summary_hits = []
        lines = ["The retrieved materials support these comparison points:"]
        for entity in entities:
            if not self.retriever:
                break
            sub = self.retriever.retrieve(f"Summarize {entity}")
            hits = [hit for hit in sub["text_hits"] if hit.get("score", 0) >= MIN_TEXT_SCORE]
            self._compare_sub_summary_hits.extend(hits)
            if hits:
                summary_lines = self._fallback_summary_answer(entity, hits)
                sentence = next((line.lstrip("- ") for line in summary_lines if line.startswith("- ")), "")
                if sentence:
                    lines.append(f"- {entity}: {sentence}")
                else:
                    lines.append(f"- {entity}: Not enough evidence found.")
            else:
                lines.append(f"- {entity}: Not enough evidence found.")
        return "\n".join(lines)

    def _fallback_compare_answer(
        self,
        question: str,
        hits: list[dict[str, Any]],
        entities: list[str] | None = None,
    ) -> list[str]:
        terms = _question_terms(question)
        entities = entities or []
        if len(entities) >= 2:
            lines = [
                "The retrieved materials support these comparison points:",
            ]
            for entity in entities:
                sentence = self._best_entity_sentence(entity, hits, terms)
                if sentence:
                    lines.append(f"- {entity}: {_quote(sentence, 300)}")
                else:
                    lines.append(f"- {entity}: Not enough directly retrieved evidence to summarize this item.")
            return lines

        sentences = _best_key_sentences(hits, terms, limit=5)
        if not sentences:
            sentences = [_quote(hit["content"], 260) for hit in hits[:3]]
        return [
            "The supported comparison points are:",
            *[f"- {_quote(sentence, 300)}" for sentence in sentences[:4]],
        ]

    def _best_entity_sentence(
        self,
        entity: str,
        hits: list[dict[str, Any]],
        question_terms: list[str],
    ) -> str | None:
        tokens = re.findall(r"[A-Za-z0-9]+", entity.lower())
        direct_hits = [hit for hit in hits if _source_covers_entity(entity, hit)]
        candidate_hits = direct_hits or hits
        candidates: list[tuple[int, str]] = []
        for hit in candidate_hits:
            haystack = " ".join(
                [
                    hit.get("content", ""),
                    str(hit.get("metadata", {}).get("source_file", "")),
                    str(hit.get("metadata", {}).get("section", "")),
                ]
            ).lower()
            if not all(token in haystack for token in tokens if len(token) > 1):
                continue
            for sentence in _sentences(hit["content"]):
                score = _sentence_score(sentence, question_terms)
                if any(token in sentence.lower() for token in tokens):
                    score += 2
                candidates.append((score, sentence))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _not_found(self) -> str:
        return "\n".join(
            [
                "## Final Answer",
                "Not found in provided materials.",
                "",
                "## Evidence / Citations",
                "No supporting evidence was found in the indexed PDF materials.",
                "",
                "## Confidence",
                "Confidence: 0% - Retrieval did not return evidence above the support threshold.",
                "",
                "## Missing Information Handling",
                "Not found in provided materials.",
            ]
        )

    def _openai_available(self) -> bool:
        if not os.getenv("OPENAI_API_KEY"):
            return False
        return importlib.util.find_spec("openai") is not None

    def _ollama_enabled(self) -> bool:
        return LLM_PROVIDER.strip().lower() == "ollama"

    def _build_generation_prompt(
        self,
        question: str,
        question_type: str,
        hits: list[dict[str, Any]],
    ) -> str:
        plan = type("Plan", (), {"question_type": question_type, "entities": []})()
        evidence_items = _select_evidence(question, plan, hits, limit=6)
        context_blocks = []
        for index, item in enumerate(evidence_items, start=1):
            context_blocks.append(
                f"[{index}] Source: {_citation(item.metadata)}\nEvidence: {item.text}"
            )
        task_instruction = ""
        if question_type == "compare":
            task_instruction = (
                "For comparison questions, discuss both items separately before stating the contrast. "
                "If the context only supports one item, say which item is missing instead of guessing. "
                "Do not answer yes or no unless the question is explicitly yes/no. "
            )
        elif question_type in {"summary", "complex_lookup"}:
            task_instruction = "Synthesize the main supported points; do not merely copy one long context sentence. "

        return (
            "You are a grounded RAG assistant. Answer strictly from the provided context. "
            "Do not use outside knowledge. If the answer is not supported, state exactly: "
            "Not found in provided materials.\n\n"
            "Return only the final answer text, without citations, confidence, or extra headings. "
            "For yes/no questions, start with Yes or No. Synthesize the evidence in your own concise wording; "
            "do not paste long raw snippets or mention internal citation rules.\n\n"
            f"{task_instruction}"
            f"Question type: {question_type}\n"
            f"Question: {question}\n\n"
            "Context:\n" + "\n\n".join(context_blocks)
        )

    def _try_ollama_answer(
        self,
        question: str,
        question_type: str,
        hits: list[dict[str, Any]],
    ) -> str | None:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        host = OLLAMA_HOST.rstrip("/")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_generation_prompt(question, question_type, hits),
                }
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            request = urllib.request.Request(
                f"{host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
            return data.get("message", {}).get("content")
        except Exception:
            if STRICT_MODE:
                raise
            return None

    def _try_openai_answer(
        self,
        question: str,
        question_type: str,
        hits: list[dict[str, Any]],
    ) -> str | None:
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = self._build_generation_prompt(question, question_type, hits)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception:
            return None
