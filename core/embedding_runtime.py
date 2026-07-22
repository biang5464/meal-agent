"""Lazy, process-level singleton for the BGE embedding model.

Shared by recipe, nutrition, and food_safety Chroma collections.
Conversation memory uses Chroma's default embedding — do NOT pass this to it.
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "BAAI/bge-small-zh"
_lock = threading.Lock()
_ef = None  # SentenceTransformerEmbeddingFunction once loaded


def get_embedding_function():
    """Return the shared SentenceTransformerEmbeddingFunction, loading it once.

    Thread-safe: double-checked locking so the model is created at most once.
    """
    global _ef
    if _ef is not None:
        return _ef
    with _lock:
        if _ef is None:
            logger.info("[embedding_runtime] loading %s", _EMBEDDING_MODEL)
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            _ef = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL)
            logger.info("[embedding_runtime] %s ready", _EMBEDDING_MODEL)
    return _ef


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via the shared model. Returns empty inner lists on failure."""
    try:
        return list(get_embedding_function()(texts))
    except Exception as exc:
        logger.warning("[embedding_runtime] embed_texts failed: %s", exc)
        return [[] for _ in texts]


def reset_embedding_runtime_for_tests() -> None:
    """Clear the singleton. Test-only — never call in production code."""
    global _ef
    with _lock:
        _ef = None
