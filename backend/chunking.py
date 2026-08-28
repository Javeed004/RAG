import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from ingest import load_all_documents


CHUNK_SIZE = 1000 #500
CHUNK_OVERLAP = 100 #50

def create_chunks(documents):
    """
    Split LangChain Documents into smaller overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def print_chunk_summary(chunks):
    """
    Print the total number of chunks and
    the number of chunks per source.
    """

    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)

    print(f"Total chunks: {len(chunks)}")

    sources = {}

    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        sources[source] = sources.get(source, 0) + 1

    print("\nChunks per source:")

    for source, count in sorted(sources.items()):
        print(f"- {source}: {count} chunks")


def verify_metadata(chunks):
    """
    Verify that chunks retain source and page metadata.
    """

    missing_source = 0
    missing_page = 0

    for chunk in chunks:

        if not chunk.metadata.get("source"):
            missing_source += 1

        if "page" not in chunk.metadata:
            missing_page += 1

    print("\n" + "=" * 60)
    print("METADATA VERIFICATION")
    print("=" * 60)

    print(f"Chunks missing source: {missing_source}")
    print(f"Chunks missing page: {missing_page}")


def inspect_chunks(chunks, number_of_chunks=5):
    """
    Randomly select chunks for manual quality inspection.
    """

    sample_size = min(number_of_chunks, len(chunks))

    selected_chunks = random.sample(
        chunks,
        sample_size
    )

    print("\n" + "=" * 60)
    print(f"RANDOM CHUNK INSPECTION - {sample_size} CHUNKS")
    print("=" * 60)

    for index, chunk in enumerate(selected_chunks, start=1):

        print(f"\nChunk {index}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Page: {chunk.metadata.get('page', 'N/A')}")
        print(f"Characters: {len(chunk.page_content)}")

        print("\nContent:")
        print(chunk.page_content)

        print("\n" + "-" * 60)
        
def analyze_chunk_sizes(chunks):
    """
    Identify unusually small chunks.
    """

    small_chunks = [
        chunk
        for chunk in chunks
        if len(chunk.page_content.strip()) < 150
    ]

    print("\n" + "=" * 60)
    print("CHUNK SIZE ANALYSIS")
    print("=" * 60)

    print(f"Total chunks: {len(chunks)}")
    print(f"Chunks under 150 characters: {len(small_chunks)}")

    if small_chunks:
        print("\nExamples of small chunks:")

        for index, chunk in enumerate(small_chunks[:5], start=1):
            print(f"\nSmall Chunk {index}")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"Characters: {len(chunk.page_content)}")
            print(chunk.page_content)
            
def save_chunks_for_review(chunks):
    
    DEBUG_DIR = Path("D:/Fobes Internship/RAG-Project/debug/")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    output_file = DEBUG_DIR / "chunks.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        for index, chunk in enumerate(chunks, start=1):
            file.write("=" * 80 + "\n")
            file.write(f"Chunk {index}\n")
            file.write(f"Source: {chunk.metadata.get('source')}\n")
            file.write(f"Page: {chunk.metadata.get('page', 'N/A')}\n")
            file.write(f"Characters: {len(chunk.page_content)}\n")
            file.write("=" * 80 + "\n\n")

            file.write(chunk.page_content)
            file.write("\n\n")

    print(f"\nChunks saved to: {output_file}")


if __name__ == "__main__":
    documents = load_all_documents()
    chunks = create_chunks(documents)
    print_chunk_summary(chunks)
    verify_metadata(chunks)
    inspect_chunks(chunks, number_of_chunks=5)
    save_chunks_for_review(chunks)
    analyze_chunk_sizes(chunks)