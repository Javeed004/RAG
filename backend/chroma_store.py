from pathlib import Path

import chromadb


COLLECTION_NAME = "rag_documents"


class ChromaVectorStore:

    def __init__(self, persist_directory):
        self.persist_directory = Path(
            persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

    def add(
        self,
        ids,
        documents,
        embeddings,
        metadatas
    ):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding,
        k=3
    ):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

    def delete_by_source(
        self,
        source
    ):
        existing = self.collection.get(
            where={
                "source": source
            }
        )

        ids = existing.get(
            "ids",
            []
        )

        if ids:
            self.collection.delete(
                ids=ids
            )

        return len(ids)

    def count(self):
        return self.collection.count()

    def reset(self):
        try:
            self.client.delete_collection(
                name=COLLECTION_NAME
            )
        except Exception:
            pass

        self.collection = (
            self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )