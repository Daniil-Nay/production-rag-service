"""CLI script for indexing the document corpus.

Usage:
    python -m src.indexing --corpus-dir data/corpus
"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.chunker import fixed_chunker
from src.retrieval.embedder import embed_passages
from src.retrieval.vector_store import (
    get_all_chunks,
    init_schema,
    upsert_chunks,
)


def index_corpus(corpus_dir: Path) -> None:
    init_schema()

    chunks_to_insert: list[dict] = []
    doc_id = 0

    for path in sorted(corpus_dir.glob("*.txt")):
        doc_id += 1
        text = path.read_text(encoding="utf-8")
        logger.info("doc {id}: {name}, {n} chars", id=doc_id, name=path.name, n=len(text))

        chunks = fixed_chunker(
            text, settings.chunk_size_words, settings.chunk_overlap_words
        )
        if not chunks:
            continue

        embeddings = embed_passages(chunks)
        for chunk_text, embedding in zip(chunks, embeddings):
            chunks_to_insert.append({
                "doc_id": doc_id,
                "text": chunk_text,
                "embedding": embedding,
                "meta": {"source": path.name},
            })

    upsert_chunks(chunks_to_insert)
    logger.info("inserted {n} chunks total", n=len(chunks_to_insert))

    # Build the BM25 index from all chunks (loaded from the database)
    all_chunks = get_all_chunks()
    bm25 = BM25Index(all_chunks)
    bm25.save()
    logger.info("BM25 index saved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.corpus_dir.exists():
        raise SystemExit(f"corpus dir not found: {args.corpus_dir}")

    index_corpus(args.corpus_dir)


if __name__ == "__main__":
    main()
