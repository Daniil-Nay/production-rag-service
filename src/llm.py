"""LLM client with retry and fallback to a cheaper model."""

import json
import time
from typing import Callable, Type, TypeVar

from loguru import logger
from openai import OpenAI, APITimeoutError, RateLimitError
from pydantic import BaseModel, ValidationError

from src.config import settings


T = TypeVar("T", bound=BaseModel)
RETRYABLE = (RateLimitError, APITimeoutError)


class LLMClient:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            timeout=settings.request_timeout_s,
        )

    def _call(self, model: str, messages: list[dict], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[dict], model: str | None = None, **kwargs) -> str:
        target = model or settings.llm_model_primary
        last_exc: Exception | None = None

        for attempt in range(settings.max_retries):
            try:
                return self._call(target, messages, **kwargs)
            except RETRYABLE as e:
                last_exc = e
                wait = min(2 ** attempt, 10)
                logger.warning(
                    "retry {n}/{m} on {exc}, sleep {s}s",
                    n=attempt + 1, m=settings.max_retries, exc=type(e).__name__, s=wait,
                )
                time.sleep(wait)

        if target != settings.llm_model_cheap:
            logger.warning("primary exhausted, falling back to cheap model")
            try:
                return self._call(settings.llm_model_cheap, messages, **kwargs)
            except Exception as e:
                logger.error("fallback also failed: {exc}", exc=type(e).__name__)
                last_exc = e

        assert last_exc is not None
        raise last_exc


def parse_structured(
    llm: LLMClient,
    user_prompt: str,
    schema: Type[T],
    max_retries: int = 3,
    system_prompt: str | None = None,
) -> T:
    """Extract structured output, retrying on JSONDecodeError / ValidationError."""
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    default_system = (
        f"Respond strictly as JSON conforming to the schema:\n{schema_json}\n"
        f"No text before or after the JSON."
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt or default_system},
        {"role": "user", "content": user_prompt},
    ]

    last_err: Exception | None = None
    for attempt in range(max_retries):
        raw = llm.chat(messages)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            logger.warning("attempt {n}: invalid JSON: {err}", n=attempt + 1, err=e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Invalid JSON: {e}. Return a valid object that matches the schema."})
            continue

        try:
            return schema.model_validate(data)
        except ValidationError as e:
            last_err = e
            logger.warning("attempt {n}: schema mismatch: {err}", n=attempt + 1, err=e)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"JSON does not match the schema: {e}. Fix it."})
            continue

    assert last_err is not None
    raise last_err
