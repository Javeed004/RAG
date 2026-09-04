from pathlib import Path
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from answer_generation import answer, load_vectorstore
from ingest import load_document
from chunking import create_chunks
from embeddings import embed_texts
from llm_factory import get_llm


# CONFIGURATION

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"

UPLOAD_DEBUG_FILE = DEBUG_DIR / "upload_debug.txt"

DATA_DIR.mkdir(parents=True, exist_ok=True)

llm = None
vector_store = None

# FASTAPI APPLICATION

app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for the local conversational RAG chatbot",
    version="1.0.0",
)


# REQUEST / RESPONSE MODELS

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the RAG chatbot",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
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


# VECTOR DATABASE

vector_store = None


# STARTUP

@app.on_event("startup")
def startup_event():

    global vector_store
    global llm

    print("\n" + "=" * 60)
    print("STARTING RAG API")
    print("=" * 60)

    try:

        print("\nLoading VectorDB...")

        vector_store = load_vectorstore()

        chunk_count = vector_store.count()

        print(
            f"VectorDB loaded successfully."
        )

        print(
            f"Chunks available: {chunk_count}"
        )

        if chunk_count == 0:
            print(
                "\n" + "!" * 60 + "\n"
                "WARNING: Vector store is EMPTY (0 chunks).\n"
                "The chatbot will not be able to answer questions "
                "until documents are ingested.\n"
                "Run 'python build_vectorstore.py' to index everything "
                "in the data/ folder, or upload a document via /upload.\n"
                + "!" * 60
            )

        print("\nLoading LLM...")

        llm = get_llm()

        print("LLM loaded successfully.")

        print("=" * 60)
        print("RAG API READY")
        print("=" * 60)

    except Exception as error:

        print(
            f"\nFailed to initialize RAG API: {error}"
        )

        vector_store = None
        llm = None


# STATUS ENDPOINT

@app.get("/status")
def status():
    """
    Report whether the API is actually ready to answer questions,
    not just whether the server process is up.
    """

    chunk_count = vector_store.count() if vector_store is not None else 0

    return {
        "vector_store_available": vector_store is not None,
        "llm_available": llm is not None,
        "chunk_count": chunk_count,
        "ready": vector_store is not None and llm is not None and chunk_count > 0,
    }

# ROOT

@app.get("/")
def root():
    return {"status": "200"}


# HEALTH CHECK

@app.get("/ping")
def ping():
    return {"status": "ok"}


# CHAT ENDPOINT

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database is not available.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        # Convert Pydantic ChatMessage objects
        # into dictionaries expected by answer_generation.py
        history = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.history
        ]

        print("\n" + "=" * 80)
        print("CHAT REQUEST")
        print("=" * 80)

        print(f"Question: {question}")
        print(f"History messages: {len(history)}")

        # Run conversational RAG
        result = answer(question, vector_store, history, llm=llm)

        # Format sources for API response
        # Keep complete retrieval information
        formatted_sources = []

        for source in result.get("sources", []):

            formatted_sources.append(
                {
                    "source": source.get(
                        "source",
                        "Unknown",
                    ),
                    "page": source.get(
                        "page",
                        "N/A",
                    ),
                    "chunk": source.get(
                        "chunk",
                        "",
                    ),
                    "distance": source.get(
                        "distance",
                    ),
                    "similarity": source.get(
                        "similarity",
                    ),
                }
            )

        print("\nAnswer generated successfully.")

        return ChatResponse(
            answer=result.get("answer", "I don't know."),
            sources=formatted_sources,
        )

    except Exception as error:
        print(f"\nError generating answer: {error}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
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
    for debugging and manual inspection.
    """

    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    with open(UPLOAD_DEBUG_FILE, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100 + "\n")
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
            file.write("=" * 100 + "\n")
            file.write("LOADED DOCUMENT OBJECTS\n")
            file.write("=" * 100 + "\n\n")

            file.write(f"Total Document objects: {len(documents)}\n\n")

            for index, document in enumerate(documents, start=1):
                file.write(f"Document {index}\n")
                file.write(
                    f"Source: {document.metadata.get('source', 'Unknown')}\n"
                )
                file.write(
                    f"Page: {document.metadata.get('page', 'N/A')}\n"
                )
                file.write(f"Characters: {len(document.page_content)}\n")
                file.write("-" * 100 + "\n")

            file.write("\n")

        if chunks:
            file.write("=" * 100 + "\n")
            file.write("CHUNKS CREATED\n")
            file.write("=" * 100 + "\n\n")

            file.write(f"Total chunks: {len(chunks)}\n\n")

            for index, (chunk, chunk_id, metadata) in enumerate(
                zip(chunks, ids, metadatas), start=1
            ):
                file.write(f"Chunk {index}\n")
                file.write(f"ID: {chunk_id}\n")
                file.write(
                    f"Source: {metadata.get('source', 'Unknown')}\n"
                )
                file.write(
                    f"Page: {metadata.get('page', 'N/A')}\n"
                )
                file.write(f"Characters: {len(chunk.page_content)}\n\n")

                file.write(chunk.page_content[:300])
                if len(chunk.page_content) > 300:
                    file.write("... [truncated]")

                file.write("\n\n")
                file.write("-" * 100 + "\n\n")

        file.write("=" * 100 + "\n")
        file.write("END OF UPLOAD RESULT\n")
        file.write("=" * 100 + "\n")


# FILE UPLOAD ENDPOINT

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vector_store

    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector database is not available.",
        )

    allowed_extensions = {".pdf", ".txt", ".docx"}

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only PDF, TXT, and DOCX are supported.",
        )

    file_path = DATA_DIR / filename

    try:
        contents = await file.read()
        with open(file_path, "wb") as output_file:
            output_file.write(contents)
    except Exception as error:
        print(f"Upload error: {error}")
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
        raise HTTPException(
            status_code=500,
            detail="Failed to upload file.",
        )

    file_size_bytes = len(contents)
    removed_chunk_count = 0

    try:
        # Remove previous chunks if the same file was uploaded before
        vector_store.delete_by_source(filename)

        # Load document
        documents = load_document(file_path)

        if not documents:
            save_upload_debug(
                filename=filename,
                file_size_bytes=file_size_bytes,
                removed_chunk_count=removed_chunk_count,
                documents=documents,
                chunks=None,
                ids=None,
                metadatas=None,
                status="FAILED - no extractable text",
            )
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the file.",
            )

        # Create chunks
        chunks = create_chunks(documents)

        if not chunks:
            save_upload_debug(
                filename=filename,
                file_size_bytes=file_size_bytes,
                removed_chunk_count=removed_chunk_count,
                documents=documents,
                chunks=chunks,
                ids=None,
                metadatas=None,
                status="FAILED - no chunks produced",
            )
            raise HTTPException(
                status_code=400,
                detail="File produced no chunks after splitting.",
            )

        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embed_texts(texts)

        # Generate unique IDs
        ids = [
            f"{filename}_{index}_{uuid.uuid4().hex[:8]}"
            for index in range(len(chunks))
        ]

        # Preserve metadata
        metadatas = [
            {
                "source": chunk.metadata.get("source", filename),
                "page": chunk.metadata.get("page", -1),
            }
            for chunk in chunks
        ]

        # Add to ChromaDB
        vector_store.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"Indexed {len(chunks)} chunks from {filename}.")
        print(f"Collection now has {vector_store
              .count()} chunks.")

        save_upload_debug(
            filename=filename,
            file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count,
            documents=documents,
            chunks=chunks,
            ids=ids,
            metadatas=metadatas,
            status="SUCCESS",
        )

        return {
            "message": f"File uploaded and indexed successfully ({len(chunks)} chunks added).",
            "filename": filename,
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Indexing error: {error}")
        save_upload_debug(
            filename=filename,
            file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count,
            documents=None,
            chunks=None,
            ids=None,
            metadatas=None,
            status="FAILED - indexing error",
            error_message=str(error),
        )
        raise HTTPException(
            status_code=500,
            detail="File was saved but could not be indexed.",
        )