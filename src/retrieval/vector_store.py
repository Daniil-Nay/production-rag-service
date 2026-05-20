"""pgvector wrapper — schema management and similarity search."""

import numpy as np
import psycopg
from loguru import logger
from pgvector.psycopg import register_vector

from src.config import settings


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(settings.pg_dsn)
    register_vector(conn)
    return conn


def _ensure_extension() -> None:
    """Create the pgvector extension. Runs on a bare connection because
    get_conn() calls register_vector(), which requires the type to already exist."""
    with psycopg.connect(settings.pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()


def init_schema() -> None:
    """Create the table and HNSW index. Idempotent."""
    _ensure_extension()
    with get_conn() as conn, conn.cursor() as cur:
        dim = int(settings.embed_dimension)
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id          serial PRIMARY KEY,
                doc_id      integer NOT NULL,
                text        text    NOT NULL,
                embedding   vector({dim}) NOT NULL,
                meta        jsonb   NOT NULL DEFAULT '{{}}'::jsonb
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        conn.commit()
        logger.info("pgvector schema ready")


def upsert_chunks(chunks: list[dict]) -> None:
    """chunks: [{doc_id, text, embedding (np.array), meta}, ...]"""
    if not chunks:
        return
    with get_conn() as conn, conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (doc_id, text, embedding, meta) VALUES (%s, %s, %s, %s::jsonb)",
            [
                (c["doc_id"], c["text"], c["embedding"], _json(c.get("meta", {})))
                for c in chunks
            ],
        )
        conn.commit()
    logger.info("inserted {n} chunks", n=len(chunks))


def search_dense(query_embedding: np.ndarray, k: int) -> list[dict]:
    """Return the top-k chunks ranked by cosine similarity."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, doc_id, text, (1 - (embedding <=> %s)) AS score
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, query_embedding, k),
        )
        rows = cur.fetchall()
    return [
        {"chunk_id": r[0], "doc_id": r[1], "text": r[2], "score": float(r[3])}
        for r in rows
    ]


def get_all_chunks() -> list[dict]:
    """Return all chunks — needed to build the BM25 index."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, doc_id, text FROM chunks ORDER BY id")
        rows = cur.fetchall()
    return [{"chunk_id": r[0], "doc_id": r[1], "text": r[2]} for r in rows]


def _json(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)
