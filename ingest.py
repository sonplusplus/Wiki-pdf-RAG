from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_assignment.config import DOCUMENTS_DIR
from rag_assignment.ingest import Ingestor


def main() -> None:
    stats = Ingestor(documents_dir=DOCUMENTS_DIR).ingest_all()
    print(
        "Ingestion complete: "
        f"scanned={stats.scanned}, skipped={stats.skipped}, "
        f"indexed={stats.indexed}, chunks={stats.chunks}, images={stats.images}"
    )


if __name__ == "__main__":
    main()
