from pathlib import Path
from types import SimpleNamespace
import tempfile

from rag_assignment.answerer import GroundedAnswerer
from rag_assignment.chunking import ChunkingConfig, split_text
from rag_assignment.vector_store import FaissFtsVectorStore


class DummyEmbedding:
    name = "dummy-embedding"

    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        return [
            1.0 if "usb" in lower else 0.0,
            1.0 if "bluetooth" in lower else 0.0,
            1.0,
        ]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_recursive_chunking_respects_size_and_overlap():
    text = (
        "Paragraph one has enough words to be meaningful. " * 8
        + "\n\n"
        + "Paragraph two should stay readable when possible. " * 8
    )
    chunks = split_text(text, ChunkingConfig(chunk_size=180, chunk_overlap=40))
    assert len(chunks) > 1
    assert all(len(chunk) <= 180 for chunk in chunks)
    assert any(chunks[index - 1][-20:].strip() in chunks[index] for index in range(1, len(chunks)))


def test_markdown_separator_keeps_heading_with_section():
    text = "# Title\nIntro text.\n\n## Details\n" + ("A long detail sentence. " * 20)
    chunks = split_text(text, ChunkingConfig(chunk_size=120, chunk_overlap=20))
    assert any(chunk.startswith("## Details") for chunk in chunks)


def test_faiss_fts_vector_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = FaissFtsVectorStore(
            faiss_path=Path(tmp) / "vectors.faiss",
            sqlite_path=Path(tmp) / "records.sqlite",
            embedding=DummyEmbedding(),
        )
        store.upsert_many(
            [
                store.make_record(
                    record_id="usb-c-1",
                    content="USB-C is a connector standard.",
                    metadata={"doc_id": "usb-c", "page": 1},
                )
            ]
        )
        hits = store.search("What is USB-C?", top_k=1)
        assert hits
        assert hits[0]["id"] == "usb-c-1"


def test_answerer_uses_clean_evidence_instead_of_raw_fallback(monkeypatch):
    monkeypatch.setattr("rag_assignment.answerer.LLM_PROVIDER", "none")
    retrieval = {
        "plan": SimpleNamespace(question_type="yes_no", entities=[]),
        "image_hits": [],
        "text_hits": [
            {
                "id": "iphone-se-1",
                "score": 0.9,
                "content": (
                    "iPhone SE iPhone SE is a discontinued series of budget smartphones, "
                    "part of the iPhone family developed by Apple. Retrieved from https://example.test"
                ),
                "metadata": {"source_file": "IPhone_SE.pdf", "page": 1, "section": "Introduction"},
            }
        ],
    }

    answer = GroundedAnswerer().answer(
        "Is the iPhone SE classified as a smartphone in the provided materials?",
        retrieval,
    )

    assert "fallback mode" not in answer
    assert "retrieved materials contain" not in answer.lower()
    assert "Yes." in answer
    assert "Source: IPhone_SE.pdf | page 1 | Introduction" in answer


def test_answerer_release_lookup_synthesizes_date(monkeypatch):
    monkeypatch.setattr("rag_assignment.answerer.LLM_PROVIDER", "none")
    retrieval = {
        "plan": SimpleNamespace(question_type="simple_lookup", entities=[]),
        "image_hits": [],
        "text_hits": [
            {
                "id": "switch-1",
                "score": 0.8,
                "content": (
                    "The Nintendo Switch was released worldwide in most regions on March 3, 2017. "
                    "The Switch was unveiled on October 20, 2016."
                ),
                "metadata": {"source_file": "Nintendo_Switch.pdf", "page": 1, "section": "Nintendo Switch"},
            }
        ],
    }

    answer = GroundedAnswerer().answer("When was the Nintendo Switch released?", retrieval)

    assert "Nintendo Switch was released on March 3, 2017." in answer
    assert "unveiled on October 20, 2016" not in answer.split("## Evidence / Citations")[0]
