"""Hybrid retrieval: dense + BM25 fused with Reciprocal Rank Fusion (RRF)."""

from src.config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.embedder import embed_query
from src.retrieval.vector_store import search_dense


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def hybrid_search(query: str, bm25: BM25Index | None = None) -> list[dict]:
    """Hybrid retrieval. Returns the top-k_final chunks with extra scoring info."""
    dense_results = search_dense(embed_query(query), settings.top_k_dense)

    if bm25 is None:
        # fall back to dense-only retrieval
        return dense_results[: settings.top_k_final]

    bm25_results = bm25.search(query, settings.top_k_bm25)

    # Fuse by chunk_id (as str, since rrf_fuse expects string ids)
    dense_ranking = [str(r["chunk_id"]) for r in dense_results]
    bm25_ranking = [str(r["chunk_id"]) for r in bm25_results]

    fused = rrf_fuse([dense_ranking, bm25_ranking])

    # Rebuild chunk_data keyed by chunk_id
    by_id: dict[str, dict] = {}
    for r in dense_results + bm25_results:
        by_id[str(r["chunk_id"])] = r

    top_final = []
    for cid_str, fused_score in fused[: settings.top_k_final]:
        chunk = by_id[cid_str]
        chunk_copy = dict(chunk)
        chunk_copy["fused_score"] = fused_score
        top_final.append(chunk_copy)

    return top_final
