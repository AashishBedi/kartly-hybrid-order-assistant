import hashlib
import re
from pathlib import Path

from app.rag.chunking import chunk_text
from app.rag.embeddings import Embedder
from app.rag.store import PolicyStore


def ingest_document(
    doc_id: str,
    text: str,
    store: PolicyStore,
    embedder: Embedder,
    chunk_size: int,
    overlap: int,
) -> dict:
    normalized_text = " ".join(text.split())
    if not normalized_text:
        raise ValueError("document text must not be empty")

    version = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:12]
    current_version = store.get_doc_version(doc_id)

    if current_version == version:
        return {
            "status": "unchanged",
            "doc_id": doc_id,
            "version": version,
            "chunks": store.count_for_doc(doc_id),
        }

    chunks = chunk_text(text, chunk_size, overlap)
    embeddings = embedder.embed(chunks)
    store.add_chunks(doc_id, version, chunks, embeddings)
    store.delete_other_versions(doc_id, version)

    status = "replaced" if current_version is not None else "ingested"
    return {
        "status": status,
        "doc_id": doc_id,
        "version": version,
        "chunks": len(chunks),
    }


def doc_id_from_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    return re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
