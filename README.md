# RAG Chatbot

## Architecture
[Mermaid diagram: User -> Streamlit (frontend/app.py) -> FastAPI (backend/main.py)
 -> {rewrite_question -> vector_db (FAISS/Chroma) -> rerank -> LLM (Groq/OpenAI/Ollama)}
 -> answer + sources -> back to Streamlit]

## Repo layout
(paste the tree from section 1 above)

## Setup (local)
1. cp .env.example .env   # fill in GROQ_API_KEY etc.
2. docker compose up --build
3. python backend/build_vectorstore.py   # once, to populate the vector DB
4. Frontend: http://localhost:8501, Backend: http://localhost:8000/status

## Environment variables
(table: name, required, default, purpose — pull straight from .env.example)

## Testing
cd backend && pytest

## Advanced retrieval
Cross-encoder re-ranking is on by default (RERANK_ENABLED=true). See
debug/ragas_history.csv for faithfulness/precision before vs. after.

## Known limitations
- FAISS index is a single local file — no concurrent-writer safety.
- Ollama provider only works locally, not on the free deploy tier.
- No auth on /upload — anyone with the URL can add documents.
- Rate limits are per-IP and reset on backend restart.

## Live demo
Frontend: https://<your-app>.streamlit.app
Backend health: https://rag-backend.onrender.com/status