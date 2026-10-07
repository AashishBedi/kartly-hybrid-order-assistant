"""Baselines used to compare against Kartly's grounded pipeline."""

import json
from typing import Any

from app.config import settings
from app.deps import get_embedder, get_llm_client, get_store
from app.llm.client import LLMClient, LLMResponse
from app.rag.embeddings import Embedder
from app.rag.store import PolicyStore


_SYSTEM_PROMPT = "You are a customer support assistant for Kartly, an online store."
_VECTOR_ONLY_SYSTEM_PROMPT = (
    "You are a customer support assistant for Kartly. Answer using only the "
    "policy excerpts provided. If they do not contain the answer, say you "
    "cannot answer from the policy documents."
)


def answer_baseline(
    question: str,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Answer a question with one LLM call and no Kartly context or tools."""
    client = llm_client if llm_client is not None else get_llm_client()
    response = client.chat(
        settings.ANSWER_MODEL,
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        stage="baseline",
        temperature=0,
        max_tokens=400,
    )
    return _result(response)


def answer_vector_only(
    question: str,
    *,
    store: PolicyStore | None = None,
    embedder: Embedder | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Answer from the four nearest policy chunks with one model call."""
    policy_store = store if store is not None else get_store()
    policy_embedder = embedder if embedder is not None else get_embedder()
    client = llm_client if llm_client is not None else get_llm_client()

    embedding = policy_embedder.embed([question])[0]
    chunks = policy_store.query(embedding, top_k=4)
    excerpts = [
        {
            "chunk_id": str(chunk["chunk_id"]),
            "text": str(chunk["text"]),
        }
        for chunk in chunks
    ]
    response = client.chat(
        settings.ANSWER_MODEL,
        [
            {"role": "system", "content": _VECTOR_ONLY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nPolicy excerpts (JSON):\n"
                    + json.dumps(excerpts, ensure_ascii=False, sort_keys=True)
                ),
            },
        ],
        stage="vector_baseline",
        temperature=0,
        max_tokens=400,
    )
    result = _result(response)
    result["chunk_ids"] = [excerpt["chunk_id"] for excerpt in excerpts]
    return result


def _result(response: LLMResponse) -> dict[str, Any]:
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
