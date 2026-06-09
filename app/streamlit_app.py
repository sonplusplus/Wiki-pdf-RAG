from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from rag_assignment.answerer import GroundedAnswerer
from rag_assignment.config import (
    DEBUG_RETRIEVAL_DEFAULT,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_INDEX_WORKERS,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    DOCUMENTS_DIR,
    EMBEDDING_PROVIDER,
    HYBRID_VECTOR_WEIGHT,
    LLM_PROVIDER,
    MAX_KEYWORD_WEIGHT,
    OLLAMA_HOST,
    RERANKER_ENABLED,
    effective_keyword_weights,
)
from rag_assignment.ingest import Ingestor
from rag_assignment.retriever import Retriever


st.set_page_config(page_title="PDF RAG", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_answerer() -> GroundedAnswerer:
    return GroundedAnswerer()


st.title("PDF RAG")

with st.sidebar:
    st.header("Index")
    st.caption(f"Default PDF folder: `{DOCUMENTS_DIR}`")
    st.caption(f"LLM: `{LLM_PROVIDER}` / `{os.getenv('OLLAMA_MODEL', '')}`")
    st.caption(f"Ollama host: `{OLLAMA_HOST}`")
    st.caption("Vector store: `FAISS + SQLite FTS5`")
    st.caption(f"Embedding: `{EMBEDDING_PROVIDER}`")
    st.caption(f"Embed batch: `{DEFAULT_EMBED_BATCH_SIZE}`")
    st.caption(f"Index workers: `{DEFAULT_INDEX_WORKERS}`")
    effective_bm25_weight, effective_keyword_weight = effective_keyword_weights()
    st.caption(f"Fusion: FAISS `{HYBRID_VECTOR_WEIGHT}` / FTS5 `{effective_bm25_weight}`")
    if effective_keyword_weight:
        st.caption(f"Lexical keyword weight: `{effective_keyword_weight}`")
    st.caption(f"Keyword cap: `{MAX_KEYWORD_WEIGHT}`")
    st.caption(f"Rerank top `{DEFAULT_RERANK_CANDIDATES}` -> final top `{DEFAULT_TOP_K}`")
    st.caption(f"Reranker: `{RERANKER_ENABLED}`")
    show_retrieval_debug = st.checkbox(
        "Show retrieval debug",
        value=DEBUG_RETRIEVAL_DEFAULT,
    )

    if st.button("Ingest / Update Vector Store"):
        with st.spinner("Indexing PDFs..."):
            stats = Ingestor(documents_dir=DOCUMENTS_DIR).ingest_all()
        st.success(
            f"Scanned {stats.scanned}, skipped {stats.skipped}, "
            f"indexed {stats.indexed}, chunks {stats.chunks}, images {stats.images}"
        )

question = st.text_input("Question", placeholder="Ask from the provided PDFs")

if st.button("Ask", type="primary") and question.strip():
    retriever = get_retriever()
    answerer = get_answerer()
    retrieval = retriever.retrieve(question, debug=show_retrieval_debug)

    st.markdown(answerer.answer(question, retrieval))

    if show_retrieval_debug and retrieval.get("debug"):
        with st.expander("Retrieval debug", expanded=False):
            for query, components in retrieval["debug"].items():
                st.markdown(f"**Query:** `{query}`")
                columns = st.columns(3)
                for column, label in zip(columns, ("semantic", "bm25", "hybrid")):
                    with column:
                        st.markdown(f"**{label.upper()}**")
                        for hit in components.get(label, []):
                            metadata = hit.get("metadata", {})
                            source = metadata.get("source_file")
                            page = metadata.get("page")
                            section = metadata.get("section")
                            snippet = " ".join(hit.get("content", "").split())[:220]
                            st.caption(
                                f"{hit.get('score', 0):.3f} | {source} p.{page} | {section}"
                            )
                            st.write(snippet)

    image_hits = retrieval.get("image_hits", [])
    if image_hits:
        st.subheader("Retrieved Images")
        columns = st.columns(min(3, len(image_hits)))
        for column, hit in zip(columns, image_hits):
            metadata = hit["metadata"]
            image_path = metadata.get("path")
            with column:
                if image_path and Path(image_path).exists():
                    st.image(
                        image_path,
                        caption=f"{metadata.get('source_file')} p.{metadata.get('page')}",
                    )
                st.caption(f"Score: {hit['score']:.3f}")
