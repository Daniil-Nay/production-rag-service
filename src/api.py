"""FastAPI HTTP endpoints for the RAG service."""

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.config import settings
from src.generation.pipeline import RAGResponse, answer_question
from src.llm import LLMClient
from src.security.injection import detection_reasons, is_likely_injection


app = FastAPI(title="demo-rag-service", version="0.1.0")
llm = LLMClient()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=RAGResponse)
async def ask(req: AskRequest) -> RAGResponse:
    if is_likely_injection(req.question):
        logger.warning(
            "injection attempt detected: {reasons}", reasons=detection_reasons(req.question)
        )
        raise HTTPException(
            status_code=400,
            detail="The request shows signs of prompt injection. Please rephrase.",
        )

    try:
        response = answer_question(req.question, llm)
    except Exception as e:
        logger.exception("pipeline failed: {e}", e=e)
        raise HTTPException(status_code=500, detail="internal processing error")

    return response
