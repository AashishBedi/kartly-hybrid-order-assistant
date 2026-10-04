"""LLM-only baseline used to compare against Kartly's grounded pipeline."""

from typing import Any

from app.config import settings
from app.deps import get_llm_client


_SYSTEM_PROMPT = "You are a customer support assistant for Kartly, an online store."


def answer_baseline(question: str) -> dict[str, Any]:
    """Answer a question with one LLM call and no Kartly context or tools."""
    response = get_llm_client().chat(
        settings.ANSWER_MODEL,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stage="baseline",
        temperature=0,
        max_tokens=400,
    )
    return {
        "answer": response.text,
        "model": response.model,
        "latency_ms": response.latency_ms,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.prompt_tokens + response.completion_tokens,
        "cost_usd": response.cost_usd,
        "attempts": response.attempts,
        "cache_hit": response.cache_hit,
    }
