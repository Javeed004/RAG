import os
from pathlib import Path
from datetime import datetime
import uuid
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from config import BACKEND_URL, BACKEND_HOST, BACKEND_PORT
from llm.answer_generation import answer, load_vectorstore
from file_processing.ingest import load_document
from file_processing.chunking import create_chunks
from file_processing.embeddings import embed_texts
from llm.llm_factory import get_llm


# LOGGING
# Use a real logger instead of raw print() so we control what gets
# written to stdout / log files. Never log request bodies, API keys,
# or full document content here.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag_api")


# CONFIGURATION

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

UPLOAD_DEBUG_FILE = DEBUG_DIR / "upload_debug.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# Comma-separated list of origins allowed to call this API.
# Set via env var in production, e.g.:
#   CORS_ORIGINS=https://your-streamlit-app.example.com
# Defaults to the local Streamlit dev server only — never "*" in prod.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:8501"
    ).split(",")
    if origin.strip()
]

MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "2000"))
MAX_UPLOAD_SIZE_BYTES = int(
    os.getenv("MAX_UPLOAD_SIZE_MB", "20")
) * 1024 * 1024

# Rate limits, kept separate because /upload is naturally called in bursts
# (batch-uploading several files from the frontend) while /chat is a
# per-question limit. Override either independently via env vars, e.g.:
#   CHAT_RATE_LIMIT=10/minute
#   UPLOAD_RATE_LIMIT=30/minute
CHAT_RATE_LIMIT = os.getenv("CHAT_RATE_LIMIT", "10/minute")
UPLOAD_RATE_LIMIT = os.getenv("UPLOAD_RATE_LIMIT", "30/minute")

llm = None
vector_store = None


# FASTAPI APPLICATION

app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for the local conversational RAG chatbot",
    version="1.0.0",
)

# RATE LIMITING
# Simple per-client-IP limiter. Prevents one user from hammering
# the LLM/embedding endpoints and running up provider bills.

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS
# Only the configured frontend origin(s) may call this API from a browser.

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# REQUEST / RESPONSE MODELS

class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="Question to ask the RAG chatbot",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=20,
        description="Recent conversation history",
    )


class SourceInfo(BaseModel):
    source: str
    page: int | str
    chunk: str
    distance: float | None = None
    similarity: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]


# STARTUP

@app.on_event("startup")
def startup_event():

    global vector_store
    global llm

    logger.info("=" * 60)
    logger.info("STARTING RAG API")
    logger.info(f"Backend URL: {BACKEND_URL}")
    logger.info("=" * 60)

    try:
        logger.info("Loading VectorDB...")
        vector_store = load_vectorstore()
        chunk_count = vector_store.count()
        logger.info("VectorDB loaded successfully.")
        logger.info(f"Chunks available: {chunk_count}")

        if chunk_count == 0:
            logger.warning(
                "Vector store is EMPTY (0 chunks). The chatbot will not "
                "be able to answer questions until documents are ingested. "
                "Run 'python build_vectorstore.py' or use /upload."
            )

    except Exception as error:
        # Never crash startup entirely just because the store failed to
        # load — the API should still come up and report /status as
        # not-ready instead of refusing to start.
        logger.error(f"Failed to load vector store: {error}")
        vector_store = None

    try:
        logger.info("Loading LLM...")
        llm = get_llm()
        logger.info("LLM loaded successfully.")

    except Exception as error:
        # Same reasoning: don't let a bad/missing API key or an
        # unreachable provider take the whole API down.
        logger.error(f"Failed to load LLM: {error}")
        llm = None

    logger.info("=" * 60)
    logger.info("RAG API READY (see /status for health)")
    logger.info("=" * 60)


# STATUS ENDPOINT

@app.get("/status")
def status():
    """
    Report whether the API is actually ready to answer questions,
    not just whether the server process is up.
    """
    try:
        chunk_count = vector_store.count() if vector_store is not None else 0
    except Exception as error:
        logger.error(f"Error checking vector store status: {error}")
        chunk_count = 0

    return {
        "vector_store_available": vector_store is not None,
        "llm_available": llm is not None,
        "chunk_count": chunk_count,
        "ready": vector_store is not None and llm is not None and chunk_count > 0,
    }
    

# HEALTH CHECK

@app.get("/ping")
def ping():
    return {"status": "ok"}


# CHAT ENDPOINT

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(CHAT_RATE_LIMIT)
def chat(request: Request, chat_request: ChatRequest):
    global vector_store, llm

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is currently unavailable. Please try again shortly.",
        )

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="The AI model is currently unavailable. Please try again shortly.",
        )

    question = chat_request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long (max {MAX_QUESTION_LENGTH} characters).",
        )

    history = [
        {"role": message.role, "content": message.content}
        for message in chat_request.history
    ]

    logger.info(f"Chat request received ({len(question)} chars, {len(history)} history messages).")

    # --- Retrieval + LLM call, isolated so we can give specific,
    # user-friendly errors instead of a generic crash. ---
    try:
        result = answer(question, vector_store, history, llm=llm)

    except ValueError as error:
        # e.g. empty query reaching search() after rewriting
        logger.warning(f"Bad request during chat: {error}")
        raise HTTPException(status_code=400, detail="Could not process this question.")

    except (ConnectionError, TimeoutError) as error:
        logger.error(f"LLM/provider connection error: {error}")
        raise HTTPException(
            status_code=502,
            detail="The AI provider is temporarily unavailable. Please try again in a moment.",
        )

    except RuntimeError as error:
        # Raised deliberately by llm_factory for config/provider failures
        logger.error(f"LLM runtime error: {error}")
        raise HTTPException(
            status_code=502,
            detail="The AI provider failed to respond. Please try again shortly.",
        )

    except Exception as error:
        # Catch-all: log full detail server-side, but never leak
        # internals (stack traces, keys, prompts) to the client.
        logger.error(f"Unexpected error generating answer: {error}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating an answer. Please try again.",
        )

    formatted_sources = [
        {
            "source": source.get("source", "Unknown"),
            "page": source.get("page", "N/A"),
            "chunk": source.get("chunk", ""),
            "distance": source.get("distance"),
            "similarity": source.get("similarity"),
        }
        for source in result.get("sources", [])
    ]

    logger.info(f"Answer generated successfully with {len(formatted_sources)} sources.")

    return ChatResponse(
        answer=result.get("answer", "I don't know."),
        sources=formatted_sources,
    )


# UPLOAD DEBUGGING

def save_upload_debug(
    filename: str,
    file_size_bytes: int,
    removed_chunk_count: int,
    documents,
    chunks,
    ids,
    metadatas,
    status: str,
    error_message: str | None = None,
) -> None:
    """
    Save document ingestion information to a text file
    for debugging and manual inspection. Never write raw
    environment variables, request headers, or API keys here.
    """
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        with open(UPLOAD_DEBUG_FILE, "a", encoding="utf-8") as file:
            file.write("\n" + "=" * 100 + "\n")
            file.write("UPLOAD DEBUG RESULT\n")
            file.write("=" * 100 + "\n\n")

            file.write(f"Timestamp: {datetime.now().isoformat()}\n")
            file.write(f"Filename: {filename}\n")
            file.write(f"File size (bytes): {file_size_bytes}\n")
            file.write(f"Status: {status}\n")

            if error_message:
                file.write(f"Error: {error_message}\n")

            file.write(f"Old chunks removed: {removed_chunk_count}\n\n")

            if documents:
                file.write("=" * 100 + "\nLOADED DOCUMENT OBJECTS\n" + "=" * 100 + "\n\n")
                file.write(f"Total Document objects: {len(documents)}\n\n")

                for index, document in enumerate(documents, start=1):
                    file.write(f"Document {index}\n")
                    file.write(f"Source: {document.metadata.get('source', 'Unknown')}\n")
                    file.write(f"Page: {document.metadata.get('page', 'N/A')}\n")
                    file.write(f"Characters: {len(document.page_content)}\n")
                    file.write("-" * 100 + "\n")
                file.write("\n")

            if chunks:
                file.write("=" * 100 + "\nCHUNKS CREATED\n" + "=" * 100 + "\n\n")
                file.write(f"Total chunks: {len(chunks)}\n\n")

                for index, (chunk, chunk_id, metadata) in enumerate(
                    zip(chunks, ids, metadatas), start=1
                ):
                    file.write(f"Chunk {index}\n")
                    file.write(f"ID: {chunk_id}\n")
                    file.write(f"Source: {metadata.get('source', 'Unknown')}\n")
                    file.write(f"Page: {metadata.get('page', 'N/A')}\n")
                    file.write(f"Characters: {len(chunk.page_content)}\n\n")
                    file.write(chunk.page_content[:300])
                    if len(chunk.page_content) > 300:
                        file.write("... [truncated]")
                    file.write("\n\n" + "-" * 100 + "\n\n")

            file.write("=" * 100 + "\nEND OF UPLOAD RESULT\n" + "=" * 100 + "\n")

    except Exception as error:
        # Debug logging must never crash the actual request.
        logger.error(f"Failed to write upload debug log: {error}")


# FILE UPLOAD ENDPOINT

@app.post("/upload")
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(request: Request, file: UploadFile = File(...)):
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge base is currently unavailable. Please try again shortly.",
        )

    allowed_extensions = {".pdf", ".txt", ".docx"}

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, TXT, and DOCX are supported.",
        )

    try:
        contents = await file.read()
    except Exception as error:
        logger.error(f"Failed to read upload stream: {error}")
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    file_size_bytes = len(contents)

    if file_size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file_size_bytes > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the maximum allowed size of "
                   f"{MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    file_path = DATA_DIR / filename

    try:
        with open(file_path, "wb") as output_file:
            output_file.write(contents)
    except Exception as error:
        logger.error(f"Upload save error: {error}")
        save_upload_debug(
            filename=filename,
            file_size_bytes=0,
            removed_chunk_count=0,
            documents=None,
            chunks=None,
            ids=None,
            metadatas=None,
            status="FAILED - could not save file",
            error_message=str(error),
        )
        raise HTTPException(status_code=500, detail="Failed to save the uploaded file.")

    removed_chunk_count = 0

    # --- Delete previous chunks for this filename ---
    try:
        vector_store.delete_by_source(filename)
    except Exception as error:
        logger.error(f"Failed to delete old chunks for {filename}: {error}")
        # Not fatal — continue, but note it in the debug log.
        removed_chunk_count = -1

    # --- Load + chunk ---
    try:
        documents = load_document(file_path)
    except Exception as error:
        logger.error(f"Failed to load document {filename}: {error}")
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=None,
            chunks=None, ids=None, metadatas=None,
            status="FAILED - could not parse file", error_message=str(error),
        )
        raise HTTPException(status_code=400, detail="Could not read this file. It may be corrupted or unsupported.")

    if not documents:
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=documents,
            chunks=None, ids=None, metadatas=None,
            status="FAILED - no extractable text",
        )
        raise HTTPException(status_code=400, detail="No extractable text found in the file.")

    try:
        chunks = create_chunks(documents)
    except Exception as error:
        logger.error(f"Chunking failed for {filename}: {error}")
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=documents,
            chunks=None, ids=None, metadatas=None,
            status="FAILED - chunking error", error_message=str(error),
        )
        raise HTTPException(status_code=500, detail="Failed to process the document's content.")

    if not chunks:
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=documents,
            chunks=chunks, ids=None, metadatas=None,
            status="FAILED - no chunks produced",
        )
        raise HTTPException(status_code=400, detail="File produced no usable content after splitting.")

    # --- Embeddings ---
    try:
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_texts(texts)
    except Exception as error:
        logger.error(f"Embedding failed for {filename}: {error}")
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=documents,
            chunks=chunks, ids=None, metadatas=None,
            status="FAILED - embedding error", error_message=str(error),
        )
        raise HTTPException(
            status_code=502,
            detail="Failed to generate embeddings for this document. Please try again.",
        )

    # --- Index into vector store ---
    try:
        ids = [
            f"{filename}_{index}_{uuid.uuid4().hex[:8]}"
            for index in range(len(chunks))
        ]
        metadatas = [
            {
                "source": chunk.metadata.get("source", filename),
                "page": chunk.metadata.get("page", -1),
            }
            for chunk in chunks
        ]

        vector_store.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    except Exception as error:
        logger.error(f"Indexing error for {filename}: {error}")
        save_upload_debug(
            filename=filename, file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count, documents=documents,
            chunks=chunks, ids=None, metadatas=None,
            status="FAILED - indexing error", error_message=str(error),
        )
        raise HTTPException(status_code=500, detail="File was processed but could not be indexed. Please try again.")

    logger.info(f"Indexed {len(chunks)} chunks from {filename}.")

    save_upload_debug(
        filename=filename, file_size_bytes=file_size_bytes,
        removed_chunk_count=removed_chunk_count, documents=documents,
        chunks=chunks, ids=ids, metadatas=metadatas, status="SUCCESS",
    )

    return {
        "message": f"File uploaded and indexed successfully ({len(chunks)} chunks added).",
        "filename": filename,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)