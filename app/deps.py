import sqlite3
from collections.abc import Generator
from functools import lru_cache

from app.config import settings
from app.db.connection import get_readonly_connection as open_readonly_connection
from app.jobs.store import JobStore
from app.llm.client import LLMClient
from app.rag.embeddings import SentenceTransformerEmbedder
from app.rag.store import PolicyStore


@lru_cache(maxsize=1)
def get_store() -> PolicyStore:
    return PolicyStore(settings.CHROMA_PATH)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    return JobStore(settings.JOBS_DB_PATH)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        price_in=settings.PRICE_INPUT_PER_MTOK,
        price_out=settings.PRICE_OUTPUT_PER_MTOK,
        cache_enabled=settings.EVAL_CACHE == "1",
    )


def get_readonly_connection() -> Generator[sqlite3.Connection, None, None]:
    connection = open_readonly_connection()
    try:
        yield connection
    finally:
        connection.close()
