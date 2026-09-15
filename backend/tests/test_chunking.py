"""
Tests for file_processing/chunking.py — the text-splitting layer.
"""
from langchain_core.documents import Document

from file_processing import chunking


def _make_document(text, source="doc.txt"):
    return Document(page_content=text, metadata={"source": source, "page": 0})


def test_create_chunks_splits_long_document():
    long_text = "Sentence about a topic. " * 200  # well over CHUNK_SIZE
    doc = _make_document(long_text)

    chunks = chunking.create_chunks([doc])

    assert len(chunks) > 1
    # No chunk should wildly exceed the configured size.
    assert all(len(c.page_content) <= chunking.CHUNK_SIZE + 50 for c in chunks)


def test_create_chunks_keeps_short_document_as_single_chunk():
    short_text = "Just one short sentence."
    doc = _make_document(short_text)

    chunks = chunking.create_chunks([doc])

    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_create_chunks_preserves_source_metadata():
    long_text = "Some repeated content. " * 100
    doc = _make_document(long_text, source="handbook.pdf")

    chunks = chunking.create_chunks([doc])

    assert len(chunks) > 1
    assert all(c.metadata.get("source") == "handbook.pdf" for c in chunks)


def test_create_chunks_empty_document_list_returns_empty():
    assert chunking.create_chunks([]) == []