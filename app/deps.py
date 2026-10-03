from functools import lru_cache

from app.config import settings
from app.jobs.store import JobStore
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
