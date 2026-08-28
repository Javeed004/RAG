from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from answer_generation import answer, load_vectorstore

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

    # Check vector database

    if collection is None:

        raise HTTPException(
            status_code=503,
            detail="Vector database is not available.",
        )

    # Validate question

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    # Run RAG

    try:

        print(
            f"\nQuestion received: {question}"
        )

        result = answer(
            question,
            collection,
        )

        # Format sources

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