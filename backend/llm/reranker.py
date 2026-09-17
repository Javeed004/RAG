import os
from sentence_transformers import CrossEncoder

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query, results, top_n=3):
    """
    Re-scores retrieved chunks with a cross-encoder and returns the same
    Chroma-style results dict, reordered and truncated to top_n.
    """
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        return results

    pairs = [[query, doc] for doc in documents]
    scores = get_reranker().predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, distances, scores),
        key=lambda item: item[3],
        reverse=True,
    )[:top_n]

    return {
        "documents": [[r[0] for r in ranked]],
        "metadatas": [[r[1] for r in ranked]],
        "distances": [[r[2] for r in ranked]],
    }