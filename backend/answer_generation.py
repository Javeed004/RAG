from pathlib import Path
from langchain_ollama import ChatOllama
from build_vectorstore import (
    create_chroma_client,
    COLLECTION_NAME,
    search,
)

# CONFIGURATION

MODEL_NAME = "llama3.2:3b"
TOP_K = 3
DEBUG_DIR = (Path(__file__).resolve().parent.parent / "debug")
DEBUG_FILE = DEBUG_DIR / "rag_debug.txt"

# LLM

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


# RAG PROMPT

RAG_PROMPT = """
You are a question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Use the context as your only source of information.
2. If the context contains information that can reasonably answer the question, answer using that information.
3. Do not require the exact wording of the question to appear in the context.
4. You may combine information from multiple context chunks.
5. Do not invent facts that are not supported by the context.
6. If the context genuinely does not contain enough information to answer the question, respond exactly:
I don't know.

Context:
{context}

Question:
{question}

Answer:
"""


# LOAD CHROMADB

def load_vectorstore():
    """
    Load the existing persistent ChromaDB collection.

    This does NOT rebuild the vector database.
    """

    client = create_chroma_client()

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Loaded ChromaDB collection: "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Collection contains "
        f"{collection.count()} chunks."
    )

    return collection


# FORMAT RETRIEVED CONTEXT

def format_context(results):
    """
    Convert ChromaDB search results into a text context
    that can be passed to the LLM.
    """

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for index, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):
        source = metadata.get(
            "source",
            "Unknown"
        )
        page = metadata.get(
            "page",
            "N/A"
        )
        context_parts.append(
            f"""
Context Chunk {index}

Source: {source}
Page: {page}

Content:
{document}
""".strip()
        )

    return "\n\n" + "\n\n".join(
        context_parts
    )


# EXTRACT SOURCES

def extract_sources(results):
    """
    Extract source filename and page number from
    the retrieved chunks.
    """

    metadatas = results["metadatas"][0]
    sources = []

    for metadata in metadatas:
        source = metadata.get(
            "source",
            "Unknown"
        )
        page = metadata.get(
            "page",
            "N/A"
        )
        source_info = {
            "source": source,
            "page": page,
        }
        if source_info not in sources:
            sources.append(source_info)

    return sources


# DEBUGGING

def save_debug_result(
    question,
    results,
    context,
    prompt,
    answer_text,
    sources,
):
    """
    Save retrieval and generation information to a text file
    for debugging and manual inspection.
    """
    DEBUG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get(
        "distances",
        [[]]
    )[0]
    with open(
        DEBUG_FILE,
        "a",
        encoding="utf-8"
    ) as file:
        file.write("\n")
        file.write("=" * 100 + "\n")
        file.write("RAG DEBUG RESULT\n")
        file.write("=" * 100 + "\n\n")
        file.write(
            f"Model: {MODEL_NAME}\n"
        )
        file.write(
            f"Top K: {TOP_K}\n"
        )
        file.write(
            f"Question: {question}\n\n"
        )
        file.write(
            "=" * 100 + "\n"
        )
        file.write(
            "RETRIEVED CHUNKS\n"
        )
        file.write(
            "=" * 100 + "\n\n"
        )
        for index, (
            document,
            metadata
        ) in enumerate(
            zip(documents, metadatas),
            start=1
        ):
            distance = (
                distances[index - 1]
                if index - 1 < len(distances)
                else "N/A"
            )
            file.write(
                f"Chunk {index}\n"
            )
            file.write(
                f"Distance: {distance}\n"
            )
            file.write(
                f"Source: "
                f"{metadata.get('source', 'Unknown')}\n"
            )
            file.write(
                f"Page: "
                f"{metadata.get('page', 'N/A')}\n\n"
            )
            file.write(
                document
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
            "CONTEXT SENT TO LLM\n"
        )
        file.write(
            "=" * 100 + "\n\n"
        )
        file.write(
            context
        )
        file.write(
            "\n\n"
        )
        file.write(
            "=" * 100 + "\n"
        )
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
            file.write(
                f"- {source['source']} "
                f"(Page {source['page']})\n"
            )
        file.write("\n")

# ANSWER FUNCTION

def answer(question, collection):
    """
    Complete RAG pipeline.

    1. Retrieve relevant chunks.
    2. Build context.
    3. Build grounded prompt.
    4. Send prompt to local Ollama LLM.
    5. Return answer and sources.
    """

    # STEP 1: RETRIEVE

    results = search(
        collection,
        question,
        k=TOP_K
    )

    # STEP 2: FORMAT CONTEXT

    context = format_context(
        results
    )

    # STEP 3: BUILD PROMPT

    prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # STEP 4: GENERATE ANSWER

    response = llm.invoke(
        prompt
    )

    answer_text = response.content.strip()

    # STEP 5: EXTRACT SOURCES

    sources = extract_sources(
        results
    )

    # STEP 6: DEBUG

    save_debug_result(
        question=question,
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
    """
    Print a RAG result in a readable format.
    """

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

        print(
            f"- {source['source']} "
            f"(Page {source['page']})"
        )


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
    print("TASK 8 - RAG QUESTION ANSWERING")
    print("=" * 80)

    print(f"\nLLM: {MODEL_NAME}")
    print(f"Top K: {TOP_K}")
    print("\nLoading existing vector database...")

    collection = load_vectorstore()

    print("\nVector database loaded successfully.")
    print("\nStarting RAG tests...")

    # TEST DOCUMENT QUESTIONS

    for question in TEST_QUESTIONS:
        try:
            result = answer(
                question,
                collection
            )

            print_result(question,result)

        except Exception as error:

            print("\nERROR while answering question:")

            print(question)

            print(f"\n{error}")

    # OUT-OF-SCOPE TEST

    out_of_scope_question = ("Who won the 2026 FIFA World Cup?")

    print("\n")
    print("=" * 80)
    print("OUT-OF-SCOPE TEST")
    print("=" * 80)

    try:
        result = answer(
            out_of_scope_question,
            collection
        )

        print_result(out_of_scope_question, result)

    except Exception as error:

        print(f"\nError: {error}")

    print("\n")
    print("=" * 80)
    print("TASK 8 TESTING COMPLETE")
    print("=" * 80)

    print(f"\nDebug results saved to:")

    print(DEBUG_FILE)