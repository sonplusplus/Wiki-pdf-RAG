import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
WIKI_PDFS_DIR = PROJECT_ROOT / "wiki_pdfs"
DOCUMENTS_DIR = WIKI_PDFS_DIR if WIKI_PDFS_DIR.exists() else DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
EXTRACTED_IMAGES_DIR = DATA_DIR / "extracted_images"

REGISTRY_PATH = INDEX_DIR / "registry.json"
TEXT_FAISS_PATH = INDEX_DIR / "text_vectors.faiss"
IMAGE_FAISS_PATH = INDEX_DIR / "image_vectors.faiss"
TEXT_SQLITE_PATH = INDEX_DIR / "text_store.sqlite"
IMAGE_SQLITE_PATH = INDEX_DIR / "image_store.sqlite"

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_EMBED_BATCH_SIZE = 100
DEFAULT_INDEX_WORKERS = 8
DEFAULT_RETRIEVAL_CANDIDATES = 10
DEFAULT_RERANK_CANDIDATES = 15
DEFAULT_COMPARE_ENTITY_CANDIDATES = 20
MAX_KEYWORD_WEIGHT = 0.3
HYBRID_VECTOR_WEIGHT = 0.7
HYBRID_BM25_WEIGHT = 0.3
HYBRID_KEYWORD_WEIGHT = 0.0
LLM_PROVIDER = "ollama"
OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_PROVIDER = "sentence-transformers"
RERANKER_ENABLED = True
STRICT_MODE = False
DEBUG_RETRIEVAL_DEFAULT = False


def effective_keyword_weights() -> tuple[float, float]:
    bm25_weight = max(0.0, HYBRID_BM25_WEIGHT)
    keyword_weight = max(0.0, HYBRID_KEYWORD_WEIGHT)
    total_keyword_weight = bm25_weight + keyword_weight
    if total_keyword_weight <= MAX_KEYWORD_WEIGHT or total_keyword_weight == 0:
        return bm25_weight, keyword_weight
    scale = MAX_KEYWORD_WEIGHT / total_keyword_weight
    return bm25_weight * scale, keyword_weight * scale


DEFAULT_CHUNK_SEPARATORS = [
    r"\n#{1,6}\s",
    r"```\n",
    r"\n\*\*\*+\n",
    r"\n---+\n",
    r"\n___+\n",
    r"\n\n",
    r"\n",
    r" ",
    "",
]
DEFAULT_TOP_K = 3
DEFAULT_IMAGE_TOP_K = 3
MIN_TEXT_SCORE = 0.08
MIN_COMPARE_TEXT_SCORE = 0.05
MIN_IMAGE_SCORE = 0.08


def ensure_data_dirs() -> None:
    for path in (DOCUMENTS_DIR, INDEX_DIR, EXTRACTED_IMAGES_DIR):
        path.mkdir(parents=True, exist_ok=True)
