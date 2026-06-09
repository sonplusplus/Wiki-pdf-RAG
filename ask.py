from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_assignment.answerer import GroundedAnswerer
from rag_assignment.retriever import Retriever


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python ask.py "What is USB-C?"')
        raise SystemExit(2)

    question = " ".join(sys.argv[1:])
    retrieval = Retriever().retrieve(question)
    print(GroundedAnswerer().answer(question, retrieval))


if __name__ == "__main__":
    main()
