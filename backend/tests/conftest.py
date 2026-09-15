"""
Shared pytest fixtures for the RAG backend test suite.

Stubs out sentence-transformers so importing file_processing.embeddings
never downloads a real model or makes a network call. This module is
collected by pytest automatically before any test file, and before
file_processing.embeddings gets imported anywhere else, so the stub is
guaranteed to be in place first.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


class _FakeSentenceTransformer:
    """Tiny stand-in for SentenceTransformer with a fixed embedding dim."""

    DIM = 384

    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, texts, normalize_embeddings=False, **_kwargs):
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.random(self.DIM).astype("float32")
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)

        result = np.array(vectors)
        return result[0] if single else result


_fake_module = types.ModuleType("sentence_transformers")
_fake_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", _fake_module)


@pytest.fixture
def fake_embedding_dim():
    return _FakeSentenceTransformer.DIM