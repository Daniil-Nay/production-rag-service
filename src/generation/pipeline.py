"""End-to-end pipeline: retrieve → generate → verify."""

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from src.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from src.generation.verifier import Citation, verify_citations
from src.llm import LLMClient, parse_structured
from src.retrieval.bm25 import BM25Index, BM25_PATH
from src.retrieval.hybrid import hybrid_search


class CitationModel(BaseModel):
    doc_id: int
    quote: str = Field(..., min_length=1)


class RAGAnswer(BaseModel):
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)


class RAGResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    verified: bool
    invalid_citations: list[CitationModel] = Field(default_factory=list)


_bm25_cache: BM25Index | None = None


def _get_bm25() -> BM25Index | None:
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache
    if not Path(BM25_PATH).exists():
        logger.warning("BM25 index not found at {p} — dense-only retrieval", p=BM25_PATH)
        return None
    _bm25_cache = BM25Index.load()
    return _bm25_cache


def answer_question(query: str, llm: LLMClient) -> RAGResponse:
    """Entry point: turn a user query into a verified RAG answer."""
    chunks = hybrid_search(query, bm25=_get_bm25())
    if not chunks:
        return RAGResponse(
            answer="В корпусе нет данных, релевантных запросу.",
            citations=[],
            verified=True,
        )

    user_prompt = build_rag_user_prompt(query, chunks)
    raw = parse_structured(
        llm=llm,
        user_prompt=user_prompt,
        schema=RAGAnswer,
        system_prompt=RAG_SYSTEM_PROMPT,
    )

    # post-hoc verification
    citations_typed = [Citation(c.doc_id, c.quote) for c in raw.citations]
    context_map = {c["doc_id"]: c["text"] for c in chunks}
    ok, invalid = verify_citations(citations_typed, context_map)
    invalid_models = [CitationModel(doc_id=c.doc_id, quote=c.quote) for c in invalid]

    if not ok:
        logger.warning("invalid citations detected: {n}", n=len(invalid))

    return RAGResponse(
        answer=raw.answer,
        citations=raw.citations,
        verified=ok,
        invalid_citations=invalid_models,
    )
