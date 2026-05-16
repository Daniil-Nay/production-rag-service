"""sentence-transformers wrapper that applies the correct embedding prefixes."""

from functools import lru_cache

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from src.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("loading embedding model: {m}", m=settings.embed_model)
    return SentenceTransformer(settings.embed_model)


def embed_query(text: str) -> np.ndarray:
    """Embed a query. The 'query:' prefix is required by e5/bge models."""
    text = text.strip()
    if not text:
        raise ValueError("empty query")
    model = _get_model()
    vec = model.encode([f"query: {text}"], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def embed_passages(texts: list[str]) -> np.ndarray:
    """Embed document chunks. The 'passage:' prefix is required by e5/bge models."""
    if not texts:
        return np.zeros((0, settings.embed_dimension), dtype=np.float32)
    model = _get_model()
    vecs = model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=32)
    return np.asarray(vecs, dtype=np.float32)
