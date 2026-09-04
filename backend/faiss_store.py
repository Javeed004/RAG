from pathlib import Path
import json

import faiss
import numpy as np


class FAISSVectorStore:
    """
    FAISS-backed vector store.

    Uses:
        IndexFlatIP + normalized embeddings

    Inner product on normalized vectors is equivalent
    to cosine similarity.
    """

    def __init__(self, persist_directory):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.index_path = (
            self.persist_directory / "index.faiss"
        )

        self.metadata_path = (
            self.persist_directory / "metadata.json"
        )

        self.index = None
        self.records = {}

        self._load()

    # LOAD

    def _load(self):
        """
        Load an existing FAISS index and metadata.
        """

        if (
            self.index_path.exists()
            and self.metadata_path.exists()
        ):
            self.index = faiss.read_index(
                str(self.index_path)
            )

            with open(
                self.metadata_path,
                "r",
                encoding="utf-8"
            ) as file:
                self.records = json.load(file)

    # SAVE

    def _save(self):
        """
        Persist FAISS index and metadata.
        """

        faiss.write_index(
            self.index,
            str(self.index_path)
        )

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.records,
                file,
                ensure_ascii=False,
                indent=2
            )

    # ADD

    def add(
        self,
        ids,
        documents,
        embeddings,
        metadatas
    ):
        """
        Add documents and embeddings to FAISS.
        """

        if not embeddings:
            return

        vectors = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]

        # Create index on first insert
        if self.index is None:

            base_index = faiss.IndexFlatIP(
                dimension
            )

            self.index = faiss.IndexIDMap2(
                base_index
            )

        # FAISS requires integer IDs
        numeric_ids = np.arange(
            len(self.records),
            len(self.records) + len(ids),
            dtype=np.int64
        )

        self.index.add_with_ids(
            vectors,
            numeric_ids
        )

        for numeric_id, (
            chunk_id,
            document,
            metadata
        ) in zip(
            numeric_ids,
            zip(
                ids,
                documents,
                metadatas
            )
        ):

            self.records[str(int(numeric_id))] = {
                "id": chunk_id,
                "document": document,
                "metadata": metadata,
            }

        self._save()

    # SEARCH

    def search(
        self,
        query_embedding,
        k=3
    ):
        """
        Search FAISS for top-k similar chunks.

        Returns a Chroma-like response structure so
        the rest of the application doesn't care which
        vector database is being used.
        """

        if self.index is None:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        query_vector = np.asarray(
            [query_embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(query_vector)

        actual_k = min(
            k,
            self.index.ntotal
        )

        scores, ids = self.index.search(
            query_vector,
            actual_k
        )

        documents = []
        metadatas = []
        distances = []

        for score, numeric_id in zip(
            scores[0],
            ids[0]
        ):

            if numeric_id == -1:
                continue

            record = self.records.get(
                str(int(numeric_id))
            )

            if record is None:
                continue

            documents.append(
                record["document"]
            )

            metadatas.append(
                record["metadata"]
            )

            # Convert cosine similarity to a
            # distance-like value so that:
            # lower = better
            distance = 1.0 - float(score)

            distances.append(distance)

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    # DELETE BY SOURCE

    def delete_by_source(
        self,
        source
    ):
        """
        Remove all chunks belonging to a source.
        """

        matching_ids = []

        for numeric_id, record in self.records.items():

            if (
                record["metadata"].get("source")
                == source
            ):
                matching_ids.append(
                    int(numeric_id)
                )

        if not matching_ids:
            return 0

        id_array = np.asarray(
            matching_ids,
            dtype=np.int64
        )

        self.index.remove_ids(
            id_array
        )

        for numeric_id in matching_ids:
            self.records.pop(
                str(numeric_id),
                None
            )

        self._save()

        return len(matching_ids)

    # COUNT

    def count(self):
        """
        Return number of stored vectors.
        """

        if self.index is None:
            return 0

        return self.index.ntotal

    # RESET

    def reset(self):
        """
        Delete the entire FAISS index.
        """

        self.index = None
        self.records = {}

        if self.index_path.exists():
            self.index_path.unlink()

        if self.metadata_path.exists():
            self.metadata_path.unlink()