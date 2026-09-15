"""
Tests for vector_db/faiss_store.py — the retrieval layer.

These exercise the FAISS-backed vector store directly (no embedding
model or LLM involved). One of these tests originally FAILED against
the checked-in code and uncovered a real crash bug — see
test_search_after_deleting_every_vector_does_not_crash below.
"""
import numpy as np
import pytest

from vector_db.faiss_store import FAISSVectorStore


def _unit_vectors(n, dim=8):
    """n simple, distinguishable unit vectors for deterministic tests."""
    vectors = []
    for i in range(n):
        v = np.zeros(dim, dtype="float32")
        v[i % dim] = 1.0
        vectors.append(v.tolist())
    return vectors


@pytest.fixture
def store(tmp_path):
    return FAISSVectorStore(persist_directory=tmp_path / "faiss_db")


def test_new_store_is_empty(store):
    assert store.count() == 0


def test_add_and_search_returns_top_k(store):
    vectors = _unit_vectors(3)
    store.add(
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        embeddings=vectors,
        metadatas=[{"source": "s"}] * 3,
    )

    results = store.search(query_embedding=vectors[0], k=2)

    assert store.count() == 3
    assert len(results["documents"][0]) == 2
    # The exact-match vector should come back as the top hit.
    assert results["documents"][0][0] == "doc a"


def test_search_on_never_populated_store_returns_empty(store):
    results = store.search(query_embedding=[0.0] * 8, k=3)

    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
    assert results["distances"] == [[]]


def test_delete_by_source_removes_matching_chunks(store):
    vectors = _unit_vectors(2)
    store.add(
        ids=["a", "b"],
        documents=["keep me", "delete me"],
        embeddings=vectors,
        metadatas=[{"source": "keep.txt"}, {"source": "delete.txt"}],
    )

    removed = store.delete_by_source("delete.txt")

    assert removed == 1
    assert store.count() == 1


def test_search_after_deleting_every_vector_does_not_crash(store):
    """
    BUG FOUND BY THIS TEST SUITE:

    search() only checked `if self.index is None`. After add() has run
    at least once, self.index is a real (non-None) FAISS index object
    for the rest of that store's lifetime — even once delete_by_source()
    has removed every vector from it, leaving ntotal == 0.

    In that state, `actual_k = min(k, self.index.ntotal)` evaluates to
    0, and `self.index.search(query_vector, 0)` raises inside FAISS
    ("k must be > 0"). Concretely: upload one document, delete it via
    /upload re-processing or admin cleanup, then ask any question —
    the /chat endpoint would 500 instead of gracefully saying
    "I don't know."

    Fixed in vector_db/faiss_store.py by also checking
    `self.index.ntotal == 0` before calling into FAISS.
    """
    vectors = _unit_vectors(1)
    store.add(
        ids=["a"],
        documents=["only document"],
        embeddings=vectors,
        metadatas=[{"source": "only.txt"}],
    )
    store.delete_by_source("only.txt")

    assert store.count() == 0

    # Should NOT raise, and should behave like a freshly created store.
    results = store.search(query_embedding=vectors[0], k=3)

    assert results["documents"] == [[]]
    assert results["metadatas"] == [[]]
    assert results["distances"] == [[]]


def test_reset_clears_the_store(store):
    vectors = _unit_vectors(2)
    store.add(
        ids=["a", "b"],
        documents=["doc a", "doc b"],
        embeddings=vectors,
        metadatas=[{"source": "s"}] * 2,
    )

    store.reset()

    assert store.count() == 0
    assert store.search(query_embedding=vectors[0], k=1)["documents"] == [[]]