# production-rag-service

A production-oriented Retrieval-Augmented Generation service: hybrid retrieval, citation-grounded generation, post-hoc citation verification, a golden-set evaluation harness, and a baseline prompt-injection filter — wired together as a deployable FastAPI service.

The point of this repository is not "RAG that answers questions." It is the engineering scaffolding around RAG that decides whether it survives production: retrieval you can measure, answers you can verify, regressions you can catch in CI, and an attack surface you have actually thought about.

## Architecture

```
User → FastAPI /ask
        │
        ▼
   hybrid retrieval        dense (pgvector) + sparse (BM25)
        │
        ▼  RRF fusion
   reranker (optional)     cross-encoder top-K refinement
        │
        ▼  top-5 chunks
   LLM generation          prompt enforces inline citations
        │
        ▼  structured (Pydantic) output
   post-hoc verification   every citation must exist in the retrieved context
        │
        ▼
     Response { answer, citations[], verified }
```

Two design decisions worth calling out:

- **Hybrid retrieval, not dense-only.** Dense embeddings miss exact-match terms (codes, names, rare tokens); BM25 misses paraphrase. Reciprocal Rank Fusion combines both without tuning a weight.
- **Verification is separate from generation.** The model is asked to cite; a deterministic verifier then checks that each cited quote is actually present in the retrieved context and flags the response if not. The LLM is never trusted to self-certify.

## Stack

- **Python 3.11+**, dependencies managed with `uv`
- **FastAPI** — HTTP API
- **pgvector** (PostgreSQL extension) — dense retrieval
- **rank_bm25** — sparse retrieval
- **sentence-transformers** (`intfloat/multilingual-e5-base` by default) — embeddings
- **OpenAI-compatible endpoint** for the LLM (model and base URL configured via env)
- **Docker Compose** — local one-command run

## Layout

```
.
├── pyproject.toml          # dependencies (uv)
├── docker-compose.yml      # pgvector + service
├── Dockerfile
├── .env.example            # configuration template
├── src/
│   ├── config.py           # pydantic-settings
│   ├── llm.py              # LLM client: retry + provider fallback
│   ├── retrieval/
│   │   ├── chunker.py      # fixed-size chunking with overlap
│   │   ├── embedder.py     # sentence-transformers wrapper
│   │   ├── vector_store.py # pgvector access layer
│   │   ├── bm25.py         # rank_bm25 wrapper
│   │   └── hybrid.py       # reciprocal rank fusion
│   ├── generation/
│   │   ├── prompts.py      # prompt templates
│   │   ├── verifier.py     # post-hoc citation verification
│   │   └── pipeline.py     # end-to-end retrieve + generate
│   ├── security/
│   │   └── injection.py    # baseline prompt-injection detector
│   ├── api.py              # FastAPI app
│   └── indexing.py         # CLI: build the index from a corpus
├── eval/
│   ├── golden.jsonl        # query → relevant_doc_ids pairs
│   ├── metrics.py          # recall@k, MRR, bootstrap confidence intervals
│   └── run_golden.py       # evaluation runner (CI gate)
├── data/corpus/            # drop documents here (empty by default)
└── tests/                  # smoke tests
```

## Run

### 1. Local, via Docker Compose

```bash
cp .env.example .env
# edit .env: set LLM_API_KEY and (optionally) LLM_BASE_URL / LLM_MODEL

docker-compose up --build
```

After startup: pgvector on `localhost:5432`, the service on `localhost:8000`.

### 2. Index a corpus

```bash
# put .txt / .md documents into data/corpus/
docker-compose exec service python -m src.indexing --corpus-dir /app/data/corpus
```

### 3. Query

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

Response shape:

```json
{
  "answer": "RAG retrieves relevant documents before generation [doc_id:1].",
  "citations": [
    {"doc_id": 1, "quote": "Retrieval-Augmented Generation retrieves ..."}
  ],
  "verified": true
}
```

`verified` is `false` when any cited quote cannot be found in the retrieved context — a caller-visible honesty signal, not a thrown error.

### 4. Evaluation

```bash
python -m eval.run_golden --golden eval/golden.jsonl --k 5
```

Reports `recall@k`, `MRR`, and a bootstrap confidence interval over the golden set. The same runner is intended as a CI gate: a retrieval change that drops recall below threshold fails the build instead of shipping silently. (Numbers depend entirely on the corpus and golden set you supply — none are baked into this repo.)

## Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

## Deliberately out of scope

Left unimplemented on purpose, to keep the core readable rather than to hide work:

- Cross-encoder reranker — interface is present in the pipeline, wiring is left out.
- Embedding cache — placeholder in `embedder.py`.
- SSE streaming responses and any frontend.
- Rate limiting — primitives live in `security/`, not yet attached to the API.

These are extension points, not unknowns; each has a defined seam in the code.

## Notes

- The default embedding model is multilingual; the service is corpus-language agnostic. Example prompts and the golden set ship with non-English content as sample data — the engineering is independent of corpus language.
- No telemetry, no external calls beyond the configured LLM endpoint and the local Postgres.
