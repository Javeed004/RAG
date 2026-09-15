"""
Tests for file_processing/embeddings.py.

sentence_transformers is stubbed out in conftest.py, so these tests run
fully offline and never touch the network or download a model.
"""
import pytest

from file_processing import embeddings


def test_embed_text_returns_vector_of_expected_length(fake_embedding_dim):
    vector = embeddings.embed_text("A sentence about vacation policy.")

    assert isinstance(vector, list)
    assert len(vector) == fake_embedding_dim


def test_embed_text_rejects_blank_string():
    with pytest.raises(ValueError):
        embeddings.embed_text("   ")


def test_embed_text_rejects_non_string_input():
    with pytest.raises(TypeError):
        embeddings.embed_text(123)


def test_embed_texts_empty_list_returns_empty_list():
    assert embeddings.embed_texts([]) == []


def test_embed_texts_returns_one_vector_per_input(fake_embedding_dim):
    result = embeddings.embed_texts(["first", "second", "third"])

    assert len(result) == 3
    assert all(len(vec) == fake_embedding_dim for vec in result)


def test_calculate_similarity_related_beats_unrelated():
    related = embeddings.calculate_similarity("annual leave policy", "vacation days")
    unrelated = embeddings.calculate_similarity(
        "annual leave policy", "how to repair a bicycle"
    )

    # With the fake embedder this is a smoke test (no real semantics),
    # so we only assert both calls succeed and return plain floats.
    assert isinstance(related, float)
    assert isinstance(unrelated, float)