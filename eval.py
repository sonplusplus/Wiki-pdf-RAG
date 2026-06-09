import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_assignment.answerer import GroundedAnswerer
from rag_assignment.retriever import Retriever


EVAL_SET_PATH = Path("eval/eval_set.jsonl")
OUTPUT_PATH = Path("report/eval_results.json")
REQUIRED_SECTIONS = [
    "## Final Answer",
    "## Evidence / Citations",
    "## Confidence",
    "## Missing Information Handling",
]


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def source_files(retrieval: dict) -> set[str]:
    files: set[str] = set()
    for hit in retrieval.get("text_hits", []):
        source_file = hit.get("metadata", {}).get("source_file")
        if source_file:
            files.add(source_file)
    for hit in retrieval.get("image_hits", []):
        source_file = hit.get("metadata", {}).get("source_file")
        if source_file:
            files.add(source_file)
    return files


def image_paths_exist(retrieval: dict) -> bool:
    image_hits = retrieval.get("image_hits", [])
    if not image_hits:
        return False
    for hit in image_hits:
        path = hit.get("metadata", {}).get("path")
        if path and Path(path).exists():
            return True
    return False


def evaluate_case(case: dict, retriever: Retriever, answerer: GroundedAnswerer) -> dict:
    start = time.perf_counter()
    retrieval = retriever.retrieve(case["question"])
    answer = answerer.answer(case["question"], retrieval)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    expected_docs = set(case.get("expected_docs", []))
    retrieved_docs = source_files(retrieval)
    unsupported_expected = bool(case.get("unsupported", False))
    unsupported_answered = "Not found in provided materials." in answer
    must_contain = case.get("must_contain", [])

    return {
        "id": case["id"],
        "type": case["type"],
        "question": case["question"],
        "latency_ms": latency_ms,
        "retrieved_docs": sorted(retrieved_docs),
        "expected_docs": sorted(expected_docs),
        "doc_hit": expected_docs.issubset(retrieved_docs) if expected_docs else unsupported_answered,
        "format_ok": all(section in answer for section in REQUIRED_SECTIONS),
        "must_contain_ok": all(term.lower() in answer.lower() for term in must_contain),
        "unsupported_ok": unsupported_answered == unsupported_expected,
        "image_ok": image_paths_exist(retrieval) if case["type"] == "image" else None,
        "answer_preview": answer[:800],
    }


def summarize(results: list[dict]) -> dict:
    def ratio(key: str) -> float:
        values = [result[key] for result in results if result[key] is not None]
        if not values:
            return 0.0
        return round(sum(1 for value in values if value) / len(values), 3)

    return {
        "cases": len(results),
        "doc_hit_rate": ratio("doc_hit"),
        "format_rate": ratio("format_ok"),
        "must_contain_rate": ratio("must_contain_ok"),
        "unsupported_accuracy": ratio("unsupported_ok"),
        "image_accuracy": ratio("image_ok"),
        "avg_latency_ms": round(
            sum(result["latency_ms"] for result in results) / max(1, len(results)),
            2,
        ),
    }


def main() -> None:
    cases = load_eval_set(EVAL_SET_PATH)
    retriever = Retriever()
    answerer = GroundedAnswerer(retriever=retriever)
    results = [evaluate_case(case, retriever, answerer) for case in cases]
    report = {"summary": summarize(results), "results": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
