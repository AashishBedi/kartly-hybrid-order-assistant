import re
from dataclasses import dataclass
from typing import TypedDict

from app.rag.embeddings import Embedder
from app.rag.store import PolicyStore


_ORDER_REFERENCE_PATTERN = re.compile(
    r"(?:\border\s*#?\s*\d+\b|(?<!\w)#\d+\b)",
    flags=re.IGNORECASE,
)
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([?!,.;:])")


class PolicyChunk(TypedDict):
    chunk_id: str
    doc_id: str
    text: str
    distance: float


@dataclass(frozen=True, slots=True)
class PolicyEvidence:
    chunks: list[PolicyChunk]
    found: bool
    best_distance: float | None


def strip_order_reference(question: str) -> str:
    """Remove order identifiers that add noise to policy retrieval."""
    stripped = _ORDER_REFERENCE_PATTERN.sub("", question)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", stripped)


def gather_policy(
    question: str,
    store: PolicyStore,
    embedder: Embedder,
    top_k: int,
    max_distance: float,
) -> PolicyEvidence:
    """Retrieve relevant policy chunks using an explicit distance threshold."""
    if store.count() == 0:
        return PolicyEvidence(chunks=[], found=False, best_distance=None)

    search_question = strip_order_reference(question)
    embedding = embedder.embed([search_question])[0]
    candidates = store.query(embedding, top_k)
    chunks = [
        PolicyChunk(
            chunk_id=str(candidate["chunk_id"]),
            doc_id=str(candidate["doc_id"]),
            text=str(candidate["text"]),
            distance=float(candidate["distance"]),
        )
        for candidate in candidates
        if float(candidate["distance"]) <= max_distance
    ]
    best_distance = min(float(chunk["distance"]) for chunk in candidates)
    return PolicyEvidence(
        chunks=chunks,
        found=bool(chunks),
        best_distance=best_distance,
    )
