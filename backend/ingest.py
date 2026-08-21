from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_document(file_path):
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

    for document in documents:
        document.metadata["source"] = str(file_path)

    return documents


def load_all_documents():
    all_documents = []

    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():
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
    print("\n" + "=" * 60)
    print("DOCUMENT INGESTION RESULTS")
    print("=" * 60)

    print(f"\nTotal Document objects: {len(documents)}")

    for index, document in enumerate(documents, start=1):

        content = document.page_content

        print(f"\nDocument {index}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Characters: {len(content)}")
        print("First 200 characters:")
        print(content[:200])
        print("-" * 60)


if __name__ == "__main__":
    documents = load_all_documents()
    verify_documents(documents)