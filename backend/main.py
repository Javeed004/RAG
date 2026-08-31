import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from answer_generation import answer, load_vectorstore
from ingest import load_document
from chunking import create_chunks
from embeddings import embed_texts

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
UPLOAD_DEBUG_FILE = DEBUG_DIR / "upload_debug.txt"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# FASTAPI APPLICATION

app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for the local RAG chatbot",
    version="1.0.0",
)

# REQUEST MODEL

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask the RAG chatbot",
    )


# RESPONSE MODEL

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


# VECTOR DATABASE

collection = None


# STARTUP

@app.on_event("startup")
def startup_event():

    global collection

    print("\n" + "=" * 60)
    print("STARTING RAG API")
    print("=" * 60)

    try:

        print("\nLoading ChromaDB...")

        collection = load_vectorstore()

        print(
            f"ChromaDB loaded successfully."
        )

        print(
            f"Chunks available: {collection.count()}"
        )

        print("=" * 60)
        print("RAG API READY")
        print("=" * 60)

    except Exception as error:

        print(
            f"\nFailed to load ChromaDB: {error}"
        )

        collection = None


# ROOT

@app.get("/")
def root():

    return {
        "status": "200"
    }


# HEALTH CHECK

@app.get("/ping")
def ping():

    return {
        "status": "ok"
    }


# CHAT

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    global collection

    if collection is None:

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

        print(
            f"\nQuestion received: {question}"
        )

        result = answer(
            question,
            collection,
        )

        formatted_sources = []

        for source in result["sources"]:

            source_name = source.get(
                "source",
                "Unknown",
            )

            page = source.get(
                "page",
                "N/A",
            )

            formatted_sources.append(
                f"{source_name} - Page {page}"
            )

        print(
            "Answer generated successfully."
        )

        return ChatResponse(
            answer=result["answer"],
            sources=formatted_sources,
        )

    except Exception as error:

        print(
            f"Error generating answer: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer.",
        )


# UPLOAD DEBUGGING

def save_upload_debug(
    filename,
    file_size_bytes,
    removed_chunk_count,
    documents,
    chunks,
    ids,
    metadatas,
    status,
    error_message=None,
):
    """
    Save document ingestion information to a text file
    for debugging and manual inspection.
    """

    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        UPLOAD_DEBUG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write("\n")
        file.write("=" * 100 + "\n")
        file.write("UPLOAD DEBUG RESULT\n")
        file.write("=" * 100 + "\n\n")

        file.write(
            f"Timestamp: {datetime.now().isoformat()}\n"
        )
        file.write(
            f"Filename: {filename}\n"
        )
        file.write(
            f"File size (bytes): {file_size_bytes}\n"
        )
        file.write(
            f"Status: {status}\n"
        )

        if error_message:

            file.write(
                f"Error: {error_message}\n"
            )

        file.write(
            f"Old chunks removed: {removed_chunk_count}\n\n"
        )

        if documents:

            file.write(
                "=" * 100 + "\n"
            )
            file.write(
                "LOADED DOCUMENT OBJECTS\n"
            )
            file.write(
                "=" * 100 + "\n\n"
            )

            file.write(
                f"Total Document objects: {len(documents)}\n\n"
            )

            for index, document in enumerate(
                documents,
                start=1
            ):

                file.write(
                    f"Document {index}\n"
                )
                file.write(
                    f"Source: "
                    f"{document.metadata.get('source', 'Unknown')}\n"
                )
                file.write(
                    f"Page: "
                    f"{document.metadata.get('page', 'N/A')}\n"
                )
                file.write(
                    f"Characters: {len(document.page_content)}\n"
                )
                file.write(
                    "-" * 100 + "\n"
                )

            file.write("\n")

        if chunks:

            file.write(
                "=" * 100 + "\n"
            )
            file.write(
                "CHUNKS CREATED\n"
            )
            file.write(
                "=" * 100 + "\n\n"
            )

            file.write(
                f"Total chunks: {len(chunks)}\n\n"
            )

            for index, (chunk, chunk_id, metadata) in enumerate(
                zip(chunks, ids, metadatas),
                start=1
            ):

                file.write(
                    f"Chunk {index}\n"
                )
                file.write(
                    f"ID: {chunk_id}\n"
                )
                file.write(
                    f"Source: {metadata.get('source', 'Unknown')}\n"
                )
                file.write(
                    f"Page: {metadata.get('page', 'N/A')}\n"
                )
                file.write(
                    f"Characters: {len(chunk.page_content)}\n\n"
                )
                file.write(
                    chunk.page_content[:300]
                )

                if len(chunk.page_content) > 300:

                    file.write(
                        "... [truncated]"
                    )

                file.write(
                    "\n\n"
                )
                file.write(
                    "-" * 100 + "\n\n"
                )

        file.write(
            "=" * 100 + "\n"
        )
        file.write(
            "END OF UPLOAD RESULT\n"
        )
        file.write(
            "=" * 100 + "\n"
        )


# FILE UPLOAD ENDPOINT

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    global collection

    if collection is None:

        raise HTTPException(
            status_code=503,
            detail="Vector database is not available.",
        )

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".docx",
    }

    extension = Path(
        file.filename
    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Only PDF, TXT, and DOCX are supported."
            ),
        )

    file_path = DATA_DIR / file.filename

    # Save the raw file to disk

    try:

        contents = await file.read()

        with open(
            file_path,
            "wb"
        ) as output_file:

            output_file.write(contents)

    except Exception as error:

        print(
            f"Upload error: {error}"
        )

        save_upload_debug(
            filename=file.filename,
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

    # Ingest the file into the vector store

    try:

        # If this filename was already indexed, remove its old
        # chunks first so re-uploads don't create duplicates.

        existing = collection.get(
            where={"source": file.filename}
        )

        if existing and existing.get("ids"):

            removed_chunk_count = len(existing["ids"])

            collection.delete(
                ids=existing["ids"]
            )

            print(
                f"Removed {removed_chunk_count} old chunks "
                f"for {file.filename}"
            )

        documents = load_document(file_path)

        if not documents:

            save_upload_debug(
                filename=file.filename,
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

        chunks = create_chunks(documents)

        if not chunks:

            save_upload_debug(
                filename=file.filename,
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

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = embed_texts(texts)

        ids = [
            f"{file.filename}_{index}_{uuid.uuid4().hex[:8]}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "source": chunk.metadata.get(
                    "source",
                    file.filename
                ),
                "page": chunk.metadata.get(
                    "page",
                    -1
                ),
            }
            for chunk in chunks
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(
            f"Indexed {len(chunks)} chunks from {file.filename}. "
            f"Collection now has {collection.count()} chunks."
        )

        save_upload_debug(
            filename=file.filename,
            file_size_bytes=file_size_bytes,
            removed_chunk_count=removed_chunk_count,
            documents=documents,
            chunks=chunks,
            ids=ids,
            metadatas=metadatas,
            status="SUCCESS",
        )

        return {
            "message": (
                f"File uploaded and indexed successfully "
                f"({len(chunks)} chunks added)."
            ),
            "filename": file.filename,
        }

    except HTTPException:

        raise

    except Exception as error:

        print(
            f"Indexing error: {error}"
        )

        save_upload_debug(
            filename=file.filename,
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