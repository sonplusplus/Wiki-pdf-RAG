import re
from dataclasses import dataclass, field


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "from",
    "give",
    "is",
    "me",
    "of",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
}

QUESTION_PREFIX_RE = re.compile(r"^(what|when|where|who|why|how|is|are|do|does|did|was|were)\b", re.I)
SUMMARY_WORDS_RE = re.compile(
    r"\b(summarize|summary|short|brief|key|features|of|give|me|the|main|tóm tắt)\b",
    re.I,
)


@dataclass(frozen=True)
class QueryPlan:
    question_type: str
    text_queries: list[str]
    image_query: str | None = None
    entities: list[str] = field(default_factory=list)


def classify_question(question: str) -> str:
    q = question.lower().strip()
    if any(term in q for term in ("show me", "picture", "image", "photo", "ảnh", "hình")):
        return "image"
    if q.startswith(("is ", "are ", "do ", "does ", "did ", "can ", "was ", "were ")):
        return "yes_no"
    if any(term in q for term in ("compare", "difference", "different", "differ", " vs ")):
        return "compare"
    if "summarize" in q or "summary" in q or "tóm tắt" in q:
        return "summary"
    if len(q.split()) > 18:
        return "complex_lookup"
    return "simple_lookup"


def plan_query(question: str) -> QueryPlan:
    question_type = classify_question(question)
    cleaned = _strip_question_words(question)

    if question_type == "image":
        image_query = re.sub(r"\b(show me|picture|image|photo|of|a|an|the)\b", " ", question, flags=re.I)
        image_query = re.sub(r"\s+", " ", image_query).strip() or cleaned
        return QueryPlan(question_type="image", text_queries=[cleaned], image_query=image_query)

    if question_type == "compare":
        return QueryPlan(
            question_type="compare",
            text_queries=_compare_queries(question),
            entities=_compare_entities(question),
        )

    if question_type == "complex_lookup":
        words = cleaned.split()
        midpoint = max(1, len(words) // 2)
        return QueryPlan(
            question_type="complex_lookup",
            text_queries=[
                cleaned,
                " ".join(words[:midpoint]),
                " ".join(words[midpoint:]),
            ],
        )

    if question_type == "yes_no":
        return QueryPlan(question_type="yes_no", text_queries=[cleaned, question])

    if question_type == "summary":
        topic = _summary_topic(question, cleaned)
        return QueryPlan(
            question_type="summary",
            text_queries=_summary_queries(question, cleaned),
            entities=[topic] if topic else [],
        )

    return QueryPlan(question_type=question_type, text_queries=[cleaned or question])


def _strip_question_words(question: str) -> str:
    value = QUESTION_PREFIX_RE.sub("", question.strip())
    value = re.sub(r"\?+$", "", value).strip()
    tokens = value.split()
    trimmed = [token for token in tokens if token.lower().strip(".,:;?") not in STOPWORDS]
    return " ".join(trimmed).strip() or value


def _compare_queries(question: str) -> list[str]:
    entities = _compare_entities(question)
    if len(entities) < 2:
        return [question]
    return [
        entities[0],
        entities[1],
        f"{entities[0]} {entities[1]} comparison differences",
        f"{entities[0]} versus {entities[1]} specifications features",
        _strip_question_words(question),
    ]


def _compare_entities(question: str) -> list[str]:
    q = re.sub(r"\?+$", "", question.strip())
    was_direct_compare = bool(re.match(r"^\s*(compare|contrast)\s+", q, flags=re.I))
    q = re.sub(r"^\s*(compare|contrast)\s+", "", q, flags=re.I)

    patterns = [
        r"\bbetween\s+(.+?)\s+\band\b\s+(.+)$",
        r"\bhow\s+do\s+(.+?)\s+differ\s+from\s+(.+?)(?:\s+\bin\b|\s+\bby\b|$)",
        r"\bhow\s+does\s+(.+?)\s+differ\s+from\s+(.+?)(?:\s+\bin\b|\s+\bby\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q, flags=re.I)
        if match:
            return [_clean_entity(match.group(1)), _clean_entity(match.group(2))]

    if was_direct_compare:
        match = re.search(r"^(.+?)\s+\b(?:vs\.?|versus|and|with)\b\s+(.+)$", q, flags=re.I)
        if match:
            return [_clean_entity(match.group(1)), _clean_entity(match.group(2))]

    cleaned = _strip_question_words(question)
    parts = re.split(r"\b(?:vs\.?|versus|with)\b", cleaned, flags=re.I)
    return [_clean_entity(part) for part in parts if len(_clean_entity(part)) > 2][:3]


def _clean_entity(value: str) -> str:
    value = re.sub(r"\b(the|main|key|differences?|different|design|usage)\b", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip(" .,:;?")


def _summary_queries(question: str, cleaned: str) -> list[str]:
    topic = _summary_topic(question, cleaned)
    queries = [
        cleaned,
        topic,
        f"{topic} overview",
        f"{topic} features uses specifications",
    ]
    return list(dict.fromkeys(query for query in queries if query))


def _summary_topic(question: str, cleaned: str) -> str:
    topic = SUMMARY_WORDS_RE.sub(" ", question)
    return re.sub(r"\s+", " ", topic).strip(" .,:;?") or cleaned
