import os
from pathlib import Path

from dotenv import load_dotenv

from chroma_store import ChromaVectorStore
from faiss_store import FAISSVectorStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
FAISS_DIR = PROJECT_ROOT / "faiss_db"


def get_vector_store():
    """
    Build the configured vector store.

    Supported:
        VECTOR_DB=chroma
        VECTOR_DB=faiss
    """

    provider = os.getenv("VECTOR_DB")

    if not provider:
        raise ValueError(
            f"VECTOR_DB is not set in {ENV_FILE}"
        )

    provider = provider.strip().lower()

    print(f"Vector database provider: {provider}")

    if provider == "chroma":
        return ChromaVectorStore(
            persist_directory=CHROMA_DIR
        )

    if provider == "faiss":
        return FAISSVectorStore(
            persist_directory=FAISS_DIR
        )

    raise ValueError(
        f"Unsupported VECTOR_DB: {provider}. "
        "Use 'chroma' or 'faiss'."
    )