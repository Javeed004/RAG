import os
from pathlib import Path
from llm_factory import get_llm
from llm_factory import get_llm_config
from embeddings import embed_texts
from vector_store_factory import get_vector_store


# CONFIGURATION

TOP_K = 3

DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
DEBUG_FILE = DEBUG_DIR / "rag_debug.txt"


# LLM (used only when running this file directly)

llm = None


# RAG PROMPT

RAG_PROMPT = """
You are a question-answering assistant.
Answer the user's question using ONLY the provided context.

Rules:
- Use the context as your only source of information.
- If the context contains information that can reasonably answer
  the question, answer using that information.
- Do not require the exact wording of the question to appear
  in the context.
- You may combine information from multiple context chunks.
- Do not invent facts that are not supported by the context.
- If the context genuinely does not contain enough information
  to answer the question, respond exactly:
  I don't know.

Context:
{context}

Question:
{question}

Answer:
"""

# Search function

def search(vector_store, query, k=3):
    """
    Perform semantic search using the configured vector store.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    query_embedding = embed_texts(
        [query]
    )[0]

    results = vector_store.search(
        query_embedding=query_embedding,
        k=k,
    )

    return results

# LOAD VECTOR STORE

def load_vectorstore():
    """
    Load the vector store configured through VECTOR_DB.

    Supported:
        VECTOR_DB=chroma
        VECTOR_DB=faiss
    """

    vector_store = get_vector_store()

    print(
        f"Loaded vector store: "
        f"{os.getenv('VECTOR_DB', 'chroma').lower()}"
    )

    print(
        f"Vector store contains "
        f"{vector_store.count()} chunks."
    )

    return vector_store


# FORMAT RETRIEVED CONTEXT

def format_context(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas), start=1
    ):
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")

        context_parts.append(
            f"""
Context Chunk {index}
Source: {source}
Page: {page}
Content:
{document}
""".strip()
        )

    return "\n\n".join(context_parts)


# EXTRACT SOURCES

def extract_sources(results):
    """
    Extract complete retrieval information for transparency.

    Returns:
        List of dictionaries containing:
        - source
        - page
        - chunk
        - distance
        - similarity
    """

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    sources = []

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas)
    ):
        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        # ChromaDB distance:
        # lower distance = more similar
        #
        # Convert it into a similarity-style score
        # where higher = more relevant.
        similarity = (
            1 - distance
            if distance is not None
            else None
        )

        sources.append(
            {
                "source": metadata.get(
                    "source",
                    "Unknown",
                ),
                "page": metadata.get(
                    "page",
                    "N/A",
                ),
                "chunk": document,
                "distance": distance,
                "similarity": similarity,
            }
        )

    return sources

# REWRITE FOLLOW-UP QUESTION

def rewrite_question(question, history, llm):
    """
    Convert a conversational follow-up question into a
    standalone question before retrieval.
    """

    if not history:
        return question

    history_text = ""

    for message in history[-6:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if content:
            history_text += f"{role}: {content}\n"

    prompt = f"""
You are a question rewriting assistant for a RAG system.
Your task is to rewrite the user's latest question into a
standalone question that can be understood without conversation history.

Use the conversation history ONLY when the latest question depends
on previous messages.

If the latest question is already standalone, return it unchanged.

Do NOT answer the question.
Do NOT add information that is not present in the conversation.
Return ONLY the rewritten question.

Conversation history:
{history_text}

Latest question:
{question}

Standalone question:
"""

    response = llm.invoke(prompt)

    if isinstance(response, str):
        rewritten_question = response.strip()
    else:
        rewritten_question = response.content.strip()

    # Safety fallback
    if not rewritten_question:
        return question

    return rewritten_question


# DEBUGGING

def save_debug_result(
    question,
    history,
    standalone_question,
    results,
    context,
    prompt,
    answer_text,
    sources,
):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]
    provider, model_name = get_llm_config()


    with open(DEBUG_FILE, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100 + "\n")
        file.write("CONVERSATIONAL RAG DEBUG RESULT\n")
        file.write("=" * 100 + "\n\n")

        file.write(f"LLM Provider: {provider}\n")
        file.write(f"LLM Model: {model_name}\n")
        file.write(f"Top K: {TOP_K}\n\n")

        file.write("=" * 100 + "\n")
        file.write("ORIGINAL QUESTION\n")
        file.write("=" * 100 + "\n\n")
        file.write(question)
        file.write("\n\n")

        file.write("=" * 100 + "\n")
        file.write("CONVERSATION HISTORY\n")
        file.write("=" * 100 + "\n\n")

        for message in history:
            file.write(
                f"{message.get('role')}: {message.get('content')}\n"
            )

        file.write("\n")

        file.write("=" * 100 + "\n")
        file.write("STANDALONE QUESTION\n")
        file.write("=" * 100 + "\n\n")
        file.write(standalone_question)
        file.write("\n\n")

        file.write("=" * 100 + "\n")
        file.write("RETRIEVED CHUNKS\n")
        file.write("=" * 100 + "\n\n")

        for index, (document, metadata) in enumerate(
            zip(documents, metadatas), start=1
        ):
            distance = (
                distances[index - 1]
                if index - 1 < len(distances)
                else "N/A"
            )

            file.write(f"Chunk {index}\n")
            file.write(f"Distance: {distance}\n")
            file.write(f"Source: {metadata.get('source', 'Unknown')}\n")
            file.write(f"Page: {metadata.get('page', 'N/A')}\n\n")
            file.write(document)
            file.write("\n\n")
            file.write("-" * 100 + "\n\n")

        file.write("=" * 100 + "\n")
        file.write("CONTEXT SENT TO LLM\n")
        file.write("=" * 100 + "\n\n")
        file.write(context)
        file.write("\n\n")

        file.write("=" * 100 + "\n")
        file.write("PROMPT\n")
        file.write("=" * 100 + "\n\n")
        file.write(prompt)
        file.write("\n\n")

        file.write("=" * 100 + "\n")
        file.write("GENERATED ANSWER\n")
        file.write("=" * 100 + "\n\n")
        file.write(answer_text)
        file.write("\n\n")

        file.write("=" * 100 + "\n")
        file.write("SOURCES\n")
        file.write("=" * 100 + "\n\n")

        for source in sources:
            file.write(f"- {source['source']} (Page {source['page']})\n")


# ANSWER FUNCTION

def answer(question, vector_store, history=None, llm=None):
    """
    Complete conversational RAG pipeline.
    1. Rewrite follow-up question.
    2. Retrieve relevant chunks.
    3. Build context.
    4. Build grounded prompt.
    5. Generate answer using Ollama.
    6. Return answer and sources.
    """

    if history is None:
        history = []

    # Use global llm only if none is provided (for standalone script usage)
    if llm is None:
        llm = get_llm()

    # STEP 1: REWRITE QUESTION

    standalone_question = rewrite_question(question, history, llm)

    print("\n" + "=" * 80)
    print("CONVERSATIONAL RAG DEBUG")
    print("=" * 80)

    print(f"\nOriginal question:\n{question}")
    print("\nConversation history:")

    if history:
        for message in history:
            print(f"{message.get('role')}: {message.get('content')}")
    else:
        print("No conversation history.")

    print(f"\nStandalone question:\n{standalone_question}")
    print("=" * 80)

    # STEP 2: RETRIEVE

    results = search(vector_store=vector_store, query=standalone_question, k=TOP_K)

    # STEP 3: FORMAT CONTEXT

    context = format_context(results)

    # STEP 4: BUILD PROMPT

    prompt = RAG_PROMPT.format(
        context=context,
        question=standalone_question,
    )

    # STEP 5: GENERATE ANSWER

    response = llm.invoke(prompt)

    if isinstance(response, str):
        answer_text = response.strip()
    else:
        answer_text = response.content.strip()

    # STEP 6: EXTRACT SOURCES

    sources = extract_sources(results)

    # STEP 7: SAVE DEBUG INFORMATION

    save_debug_result(
        question=question,
        history=history,
        standalone_question=standalone_question,
        results=results,
        context=context,
        prompt=prompt,
        answer_text=answer_text,
        sources=sources,
    )

    # RETURN

    return {
        "answer": answer_text,
        "sources": sources,
    }


# PRINT RESULT

def print_result(question, result):
    print("\n")
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(question)

    print("\n")
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(result["answer"])

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

    for source in result["sources"]:
        print(f"- {source['source']} (Page {source['page']})")


# TEST QUESTIONS

TEST_QUESTIONS = [
    "What is the role of governments in social participation?",
    "How can digital parenting interventions improve inclusivity?",
    "What does the World Health Statistics 2021 report contain?",
    "How can parenting interventions reach families in remote areas?",
    "Why is communication important for social participation?",
]


# MAIN

if __name__ == "__main__":
    print("=" * 80)
    print("TASK 8 / TASK 11 - CONVERSATIONAL RAG")
    print("=" * 80)
    
    provider, model_name = get_llm_config()

    print(f"\nLLM Provider: {provider}")
    print(f"LLM Model: {model_name}")

    print(f"Top K: {TOP_K}")

    print("\nLoading existing vector database...")

    vector_store = load_vectorstore()

    print("\nVector database loaded successfully.")
    print("\nStarting RAG tests...")

    # DOCUMENT TESTS

    for question in TEST_QUESTIONS:
        try:
            result = answer(question, vector_store)
            print_result(question, result)
        except Exception as error:
            print("\nERROR while answering question:")
            print(question)
            print(f"\n{error}")

    # OUT OF SCOPE TEST

    out_of_scope_question = "Who won the 2026 FIFA World Cup?"

    print("\n")
    print("=" * 80)
    print("OUT-OF-SCOPE TEST")
    print("=" * 80)

    try:
        result = answer(out_of_scope_question, vector_store)
        print_result(out_of_scope_question, result)
    except Exception as error:
        print(f"\nError: {error}")

    print("\n")
    print("=" * 80)
    print("TASK 11 TESTING COMPLETE")
    print("=" * 80)

    print("\nDebug results saved to:")
    print(DEBUG_FILE)