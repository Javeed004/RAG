from pathlib import Path

import chromadb

from ingest import load_all_documents
from chunking import create_chunks
from embeddings import embed_texts


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "rag_documents"


def create_chroma_client():
    """
    Create a persistent ChromaDB client.
    """

    CHROMA_DIR.mkdir(exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


def create_collection(client):
    """
    Create or recreate the ChromaDB collection.
    """

    try:
        client.delete_collection(
            name=COLLECTION_NAME
        )

        print("Existing collection deleted.")

    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def build_vectorstore():
    """
    Run the complete ingestion pipeline:

    documents
        ↓
    chunks
        ↓
    embeddings
        ↓
    ChromaDB
    """

    print("=" * 60)
    print("BUILDING VECTOR STORE")
    print("=" * 60)

    # Step 1: Load documents
    print("\n[1/4] Loading documents...")

    documents = load_all_documents()

    print(
        f"Loaded {len(documents)} Document objects."
    )

    # Step 2: Create chunks
    print("\n[2/4] Creating chunks...")

    chunks = create_chunks(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embed_texts(texts)

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # Step 4: Store in ChromaDB
    print("\n[4/4] Storing in ChromaDB...")

    client = create_chroma_client()

    collection = create_collection(client)

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

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(
        f"Stored {collection.count()} chunks "
        "in ChromaDB."
    )

    print(
        f"\nDatabase location: {CHROMA_DIR}"
    )

    return collection


def search(
    collection,
    query,
    k=3
):
    """
    Search ChromaDB for the top-k
    semantically similar chunks.
    """

    if not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    query_embedding = embed_texts(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    return results


def print_search_results(
    query,
    results
):
    """
    Display search results in a readable format.
    """

    print("\n" + "=" * 80)
    print("SEARCH RESULTS")
    print("=" * 80)

    print(f"\nQuery: {query}")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        print("\n" + "-" * 80)

        print(f"Result {index}")
        print(f"Distance: {distance}")
        print(
            f"Source: "
            f"{metadata.get('source')}"
        )
        print(
            f"Page: "
            f"{metadata.get('page', 'N/A')}"
        )

        print("\nContent:")
        print(document)


def test_search(collection):
    """
    Test semantic search using real questions
    about the document collection.
    """

    test_queries = [
        "What is the role of governments in social participation?",
        "How can digital parenting interventions improve inclusivity?",
        "What are the key indicators included in the World Health Statistics report?",
    ]

    for query in test_queries:

        results = search(
            collection,
            query,
            k=3
        )

        print_search_results(
            query,
            results
        )

def load_vectorstore():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return collection

if __name__ == "__main__":
    collection = build_vectorstore()
    test_search(collection)