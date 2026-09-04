from pathlib import Path
from time import perf_counter

from ingest import load_all_documents
from chunking import create_chunks
from embeddings import embed_texts
from vector_store_factory import get_vector_store


def build_vectorstore():

    print("=" * 60)
    print("BUILDING VECTOR STORE")
    print("=" * 60)

    total_start = perf_counter()

    # 1. LOAD

    start = perf_counter()

    print("\n[1/4] Loading documents...")

    documents = load_all_documents()

    load_time = perf_counter() - start

    print(
        f"Loaded {len(documents)} Document objects."
    )

    print(
        f"Loading time: {load_time:.2f}s"
    )

    # 2. CHUNK

    start = perf_counter()

    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(documents)

    chunk_time = perf_counter() - start

    print(
        f"Created {len(chunks)} chunks."
    )

    print(
        f"Chunking time: {chunk_time:.2f}s"
    )

    # 3. EMBEDDINGS

    start = perf_counter()

    print("\n[3/4] Generating embeddings...")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embed_texts(texts)

    embedding_time = perf_counter() - start

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    print(
        f"Embedding time: {embedding_time:.2f}s"
    )

    # 4. VECTOR DATABASE

    start = perf_counter()

    print("\n[4/4] Storing vectors...")

    vector_store = get_vector_store()

    # Important for benchmark:
    # start with an empty database.
    vector_store.reset()

    ids = [
        f"chunk_{index:06d}"
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "source": chunk.metadata.get(
                "source",
                "unknown"
            ),
            "page": chunk.metadata.get(
                "page",
                -1
            ),
        }
        for chunk in chunks
    ]

    vector_store.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    storage_time = perf_counter() - start

    total_time = (
        perf_counter()
        - total_start
    )

    print(
        f"Stored {vector_store.count()} chunks."
    )

    print(
        f"Storage time: {storage_time:.2f}s"
    )

    print(
        f"Total ingestion time: "
        f"{total_time:.2f}s"
    )

    print("=" * 60)

    return vector_store


if __name__ == "__main__":
    build_vectorstore()