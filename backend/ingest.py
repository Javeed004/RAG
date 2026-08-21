from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEBUG_DIR = PROJECT_ROOT / "debug"


# Files that should not be treated as source documents
IGNORED_FILES = {
    "extracted_content.txt",
}


def load_document(file_path):
    """
    Load a single PDF, TXT, or DOCX file and return
    a list of LangChain Document objects.
    """

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    clean_documents = []

    for document in documents:

        # Skip completely empty documents/pages
        if not document.page_content.strip():
            continue

        # Store only the filename instead of the full local path
        document.metadata["source"] = file_path.name

        clean_documents.append(document)

    return clean_documents


def load_all_documents():
    """
    Load all supported documents from the data directory.
    """

    all_documents = []

    for file_path in DATA_DIR.iterdir():

        # Ignore directories
        if not file_path.is_file():
            continue

        # Ignore generated/debug files
        if file_path.name in IGNORED_FILES:
            continue

        try:
            documents = load_document(file_path)

            all_documents.extend(documents)

            print(
                f"Loaded: {file_path.name} "
                f"({len(documents)} Document objects)"
            )

        except ValueError as error:
            print(
                f"Skipping {file_path.name}: {error}"
            )

        except Exception as error:
            print(
                f"Error loading {file_path.name}: {error}"
            )

    return all_documents


def verify_documents(documents):
    """
    Print basic information about every loaded Document.
    """

    print("\n" + "=" * 60)
    print("DOCUMENT INGESTION RESULTS")
    print("=" * 60)

    print(f"\nTotal Document objects: {len(documents)}")

    for index, document in enumerate(documents, start=1):

        content = document.page_content

        print(f"\nDocument {index}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page', 'N/A')}")
        print(f"Characters: {len(content)}")
        print("First 200 characters:")
        print(content[:200])
        print("-" * 60)


def save_extracted_text(documents):
    """
    Save all extracted text to a debug TXT file
    for manual verification.
    """

    DEBUG_DIR.mkdir(exist_ok=True)

    output_file = DEBUG_DIR / "extracted_content.txt"

    with open(output_file, "w", encoding="utf-8") as file:

        for index, document in enumerate(documents, start=1):

            file.write("=" * 80 + "\n")
            file.write(f"Document {index}\n")
            file.write(
                f"Source: {document.metadata.get('source')}\n"
            )
            file.write(
                f"Page: {document.metadata.get('page', 'N/A')}\n"
            )
            file.write(
                f"Characters: {len(document.page_content)}\n"
            )
            file.write("=" * 80 + "\n\n")

            file.write(document.page_content)
            file.write("\n\n")

    print(
        f"\nExtracted content saved to: {output_file}"
    )


def print_summary(documents):
    """
    Print a summary of the ingestion process.
    """

    total = len(documents)

    sources = {}

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        sources[source] = sources.get(source, 0) + 1

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(f"Total Document objects: {total}")
    print(f"Unique source files: {len(sources)}")

    print("\nDocuments per source:")

    for source, count in sorted(sources.items()):
        print(f"- {source}: {count} documents")


if __name__ == "__main__":
    documents = load_all_documents()
    verify_documents(documents)
    save_extracted_text(documents)
    print_summary(documents)