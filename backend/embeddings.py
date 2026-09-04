from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


model = SentenceTransformer(MODEL_NAME)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG_DIR = PROJECT_ROOT / "debug"


def embed_text(text):
    """
    Convert a text string into a numerical embedding vector.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text cannot be empty")

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )
    return embedding.tolist()


def embed_texts(texts):
    """
    Convert multiple text strings into embedding vectors.
    """

    if not texts:
        return []

    embeddings = model.encode(texts)

    return embeddings.tolist()


def calculate_similarity(text1, text2):
    """
    Calculate cosine similarity between two texts.
    """

    embeddings = model.encode([
        text1,
        text2
    ])

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return float(similarity)


def test_sample_chunks():
    """
    Embed three sample chunks and verify vector dimensions.
    """

    sample_chunks = [
        "Employees are entitled to annual leave.",
        "Parents should receive support during early childhood.",
        "Universal health coverage requires accessible healthcare services.",
    ]

    print("\n" + "=" * 60)
    print("SAMPLE EMBEDDING TEST")
    print("=" * 60)

    for index, text in enumerate(sample_chunks, start=1):

        embedding = embed_text(text)

        print(f"\nChunk {index}")
        print(f"Text: {text}")
        print(f"Vector length: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

def test_similarity():
    """
    Compare semantically related and unrelated sentences.
    """

    related_1 = "annual leave policy"
    related_2 = "vacation days"

    unrelated_1 = "annual leave policy"
    unrelated_2 = "how to repair a bicycle"

    related_score = calculate_similarity(
        related_1,
        related_2
    )

    unrelated_score = calculate_similarity(
        unrelated_1,
        unrelated_2
    )

    print("\n" + "=" * 60)
    print("COSINE SIMILARITY TEST")
    print("=" * 60)

    print("\nRelated sentences:")
    print(f"  '{related_1}'")
    print(f"  '{related_2}'")
    print(f"  Similarity: {related_score:.4f}")

    print("\nUnrelated sentences:")
    print(f"  '{unrelated_1}'")
    print(f"  '{unrelated_2}'")
    print(f"  Similarity: {unrelated_score:.4f}")

    print("\nComparison:")

    if related_score > unrelated_score:
        print("PASS: Related sentences have higher similarity.")
    else:
        print("WARNING: Related sentences did not score higher.")

    return (
        (related_1, related_2),
        (unrelated_1, unrelated_2),
        related_score,
        unrelated_score,
    )

def save_debug_results(
    sample_chunks,
    related_pair,
    unrelated_pair,
    related_score,
    unrelated_score,
):
    """
    Save embedding and similarity test results
    to a text file for debugging and verification.
    """

    DEBUG_DIR.mkdir(exist_ok=True)

    output_file = DEBUG_DIR / "embedding_debug.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        file.write("=" * 80 + "\n")
        file.write("EMBEDDING DEBUG RESULTS\n")
        file.write("=" * 80 + "\n\n")

        file.write(f"Model: {MODEL_NAME}\n")
        file.write("Embedding dimension: 384\n\n")

        # Sample embeddings
        file.write("=" * 80 + "\n")
        file.write("SAMPLE EMBEDDINGS\n")
        file.write("=" * 80 + "\n\n")

        for index, text in enumerate(sample_chunks, start=1):

            embedding = embed_text(text)

            file.write(f"Chunk {index}\n")
            file.write(f"Text: {text}\n")
            file.write(f"Vector length: {len(embedding)}\n")
            file.write(
                f"First 10 values: {embedding[:10]}\n"
            )
            file.write("\n" + "-" * 80 + "\n\n")

        # Similarity results
        file.write("=" * 80 + "\n")
        file.write("COSINE SIMILARITY TEST\n")
        file.write("=" * 80 + "\n\n")

        file.write("Related sentences:\n")
        file.write(f"Sentence 1: {related_pair[0]}\n")
        file.write(f"Sentence 2: {related_pair[1]}\n")
        file.write(
            f"Cosine similarity: {related_score:.4f}\n\n"
        )

        file.write("Unrelated sentences:\n")
        file.write(f"Sentence 1: {unrelated_pair[0]}\n")
        file.write(f"Sentence 2: {unrelated_pair[1]}\n")
        file.write(
            f"Cosine similarity: {unrelated_score:.4f}\n\n"
        )

        file.write("=" * 80 + "\n")
        file.write("COMPARISON\n")
        file.write("=" * 80 + "\n\n")

        if related_score > unrelated_score:
            file.write(
                "PASS: Related sentences have higher "
                "similarity than unrelated sentences.\n"
            )
        else:
            file.write(
                "WARNING: Related sentences did not have "
                "higher similarity.\n"
            )

    print(f"\nDebug results saved to: {output_file}")
    
if __name__ == "__main__":
    
    sample_chunks = [
        "Employees are entitled to annual leave.",
        "Parents should receive support during early childhood.",
        "Universal health coverage requires accessible healthcare services.",
    ]
    
    test_sample_chunks()
    (
        related_pair,
        unrelated_pair,
        related_score,
        unrelated_score,
    ) = test_similarity()

    save_debug_results(
        sample_chunks=sample_chunks,
        related_pair=related_pair,
        unrelated_pair=unrelated_pair,
        related_score=related_score,
        unrelated_score=unrelated_score,
    )
    test_similarity()