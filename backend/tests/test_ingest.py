"""
Tests for file_processing/ingest.py — the document loading layer.
"""
import pytest

from file_processing import ingest


def test_load_document_txt_reads_content_and_sets_source(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("Hello world, this is a test document.", encoding="utf-8")

    docs = ingest.load_document(sample)

    assert len(docs) == 1
    assert "Hello world" in docs[0].page_content
    # Only the filename should be stored, never the full local path.
    assert docs[0].metadata["source"] == "sample.txt"


def test_load_document_skips_empty_pages(tmp_path):
    sample = tmp_path / "empty.txt"
    sample.write_text("   \n\n  ", encoding="utf-8")

    docs = ingest.load_document(sample)

    assert docs == []


def test_load_document_unsupported_extension_raises(tmp_path):
    sample = tmp_path / "notes.xyz"
    sample.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError):
        ingest.load_document(sample)


def test_load_all_documents_ignores_debug_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    (tmp_path / "real.txt").write_text("Real content here.", encoding="utf-8")
    (tmp_path / "extracted_content.txt").write_text(
        "Should be ignored.", encoding="utf-8"
    )

    docs = ingest.load_all_documents()

    sources = {d.metadata["source"] for d in docs}
    assert "real.txt" in sources
    assert "extracted_content.txt" not in sources


def test_load_all_documents_skips_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    (tmp_path / "subdir").mkdir()
    (tmp_path / "real.txt").write_text("Content", encoding="utf-8")

    docs = ingest.load_all_documents()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "real.txt"