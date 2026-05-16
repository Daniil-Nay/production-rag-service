FROM python:3.11-slim

WORKDIR /app

# Build essentials for sentence-transformers / pgvector binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        fastapi>=0.110 \
        "uvicorn[standard]>=0.27" \
        pydantic>=2.5 \
        pydantic-settings>=2.0 \
        openai>=1.40 \
        sentence-transformers>=2.7 \
        rank-bm25>=0.2.2 \
        "psycopg[binary]>=3.1" \
        pgvector>=0.2.5 \
        loguru>=0.7 \
        tenacity>=8.2 \
        numpy>=1.26 \
        tiktoken>=0.7

COPY src ./src
COPY eval ./eval

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
