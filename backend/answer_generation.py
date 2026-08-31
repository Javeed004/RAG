from pathlib import Path
from langchain_ollama import OllamaLLM
from build_vectorstore import (
    create_chroma_client,
    COLLECTION_NAME,
    search,
)


# CONFIGURATION

MODEL_NAME = "llama3.2:3b"
TOP_K = 3

DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
DEBUG_FILE = DEBUG_DIR / "rag_debug.txt"


# LLM (used only when running this file directly)

llm = OllamaLLM(
    model=MODEL_NAME,
    temperature=0,
)


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


# ============================================================
# LOAD CHROMADB
# ============================================================

def load_vectorstore():
    client = create_chroma_client()

    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Loaded ChromaDB collection: {COLLECTION_NAME}")
    print(f"Collection contains {collection.count()} chunks.")

    return collection


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
    metadatas = results["metadatas"][0]

    sources = []

    for metadata in metadatas:
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")

        source_info = {
            "source": source,
            "page": page,
        }

        if source_info not in sources:
            sources.append(source_info)

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

    with open(DEBUG_FILE, "a", encoding="utf-8") as file:
        file.write("\n")
        file.write("=" * 100 + "\n")
        file.write("CONVERSATIONAL RAG DEBUG RESULT\n")
        file.write("=" * 100 + "\n\n")

        file.write(f"Model: {MODEL_NAME}\n")
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

def answer(question, collection, history=None, llm=None):
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
        llm = globals().get("llm")
        if llm is None:
            raise RuntimeError(
                "No LLM instance provided and no global 'llm' available."
            )

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

    results = search(collection=collection, query=standalone_question, k=TOP_K)

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

    print(f"\nLLM: {MODEL_NAME}")
    print(f"Top K: {TOP_K}")

    print("\nLoading existing vector database...")

    collection = load_vectorstore()

    print("\nVector database loaded successfully.")
    print("\nStarting RAG tests...")

    # DOCUMENT TESTS

    for question in TEST_QUESTIONS:
        try:
            result = answer(question, collection)
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
        result = answer(out_of_scope_question, collection)
        print_result(out_of_scope_question, result)
    except Exception as error:
        print(f"\nError: {error}")

    print("\n")
    print("=" * 80)
    print("TASK 11 TESTING COMPLETE")
    print("=" * 80)

    print("\nDebug results saved to:")
    print(DEBUG_FILE)