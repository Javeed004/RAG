"""
Tests for main.py — the FastAPI application (Task 17's hardened API).

The vector store and LLM are replaced with lightweight fakes so these
tests run instantly, fully offline, with no real vector DB or LLM
provider needing to be configured.
"""
import pytest
from fastapi.testclient import TestClient


class FakeVectorStore:
    """Minimal stand-in satisfying the interface main.py relies on."""

    def __init__(self, chunk_count=5):
        self._chunk_count = chunk_count

    def count(self):
        return self._chunk_count

    def delete_by_source(self, source):
        return 0

    def add(self, **kwargs):
        pass


class FakeLLM:
    """Stand-in chat model. Never actually invoked in these tests because
    llm.answer_generation.answer() is mocked directly, but main.py's
    startup event still expects something LLM-shaped to exist."""

    def invoke(self, prompt):
        return type("Resp", (), {"content": "mock answer"})()


@pytest.fixture
def client(monkeypatch):
    import main  # imported here so conftest's sys.path fix runs first

    monkeypatch.setattr(main, "load_vectorstore", lambda: FakeVectorStore())
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM())

    with TestClient(main.app) as test_client:
        yield test_client, main


def test_ping_endpoint(client):
    test_client, _ = client
    response = test_client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_reports_ready(client):
    test_client, _ = client
    response = test_client.get("/status")

    body = response.json()
    assert response.status_code == 200
    assert body["vector_store_available"] is True
    assert body["llm_available"] is True
    assert body["chunk_count"] == 5
    assert body["ready"] is True


def test_chat_endpoint_returns_answer_and_sources(client, monkeypatch):
    test_client, main = client

    def fake_answer(question, vector_store, history=None, llm=None):
        return {
            "answer": "Refunds are processed within 5 business days.",
            "sources": [
                {
                    "source": "policy.pdf",
                    "page": 2,
                    "chunk": "Refunds take 5 business days.",
                    "distance": 0.12,
                    "similarity": 0.88,
                }
            ],
        }

    monkeypatch.setattr(main, "answer", fake_answer)

    response = test_client.post(
        "/chat", json={"question": "How long do refunds take?", "history": []}
    )

    assert response.status_code == 200
    body = response.json()
    assert "5 business days" in body["answer"]
    assert body["sources"][0]["source"] == "policy.pdf"


def test_chat_endpoint_rejects_blank_question(client):
    test_client, _ = client

    response = test_client.post("/chat", json={"question": "   ", "history": []})

    # Passes pydantic's min_length=1 (whitespace has length > 0), then
    # gets caught by main.py's explicit `.strip()` empty-question check.
    assert response.status_code == 400


def test_chat_endpoint_rejects_oversized_question(client):
    test_client, _ = client

    huge_question = "a" * 5000  # exceeds MAX_QUESTION_LENGTH default (2000)
    response = test_client.post(
        "/chat", json={"question": huge_question, "history": []}
    )

    assert response.status_code == 422  # Pydantic Field max_length validation


def test_chat_endpoint_returns_503_when_vector_store_unavailable(monkeypatch):
    import main

    monkeypatch.setattr(main, "load_vectorstore", lambda: None)
    monkeypatch.setattr(main, "get_llm", lambda: FakeLLM())

    with TestClient(main.app) as test_client:
        response = test_client.post(
            "/chat", json={"question": "Any question?", "history": []}
        )

    assert response.status_code == 503


def test_chat_endpoint_handles_provider_errors_gracefully(client, monkeypatch):
    test_client, main = client

    def broken_answer(*args, **kwargs):
        raise RuntimeError("LLM provider exploded")

    monkeypatch.setattr(main, "answer", broken_answer)

    response = test_client.post("/chat", json={"question": "Anything?", "history": []})

    assert response.status_code == 502
    assert "detail" in response.json()


def test_upload_endpoint_rejects_unsupported_file_type(client):
    test_client, _ = client

    response = test_client.post(
        "/upload",
        files={"file": ("malware.exe", b"binary-content", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_endpoint_rejects_empty_file(client):
    test_client, _ = client

    response = test_client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400