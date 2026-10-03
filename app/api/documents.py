import threading
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from app.config import settings
from app.deps import get_embedder, get_job_store, get_store
from app.jobs.store import JobStore
from app.rag.embeddings import Embedder
from app.rag.ingest import doc_id_from_filename, ingest_document
from app.rag.store import PolicyStore


router = APIRouter()
_doc_locks: dict[str, threading.Lock] = {}
_doc_locks_guard = threading.Lock()


def _get_doc_lock(doc_id: str) -> threading.Lock:
    with _doc_locks_guard:
        if doc_id not in _doc_locks:
            _doc_locks[doc_id] = threading.Lock()
        return _doc_locks[doc_id]


def _ingest_in_background(
    job_id: str,
    doc_id: str,
    text: str,
    store: PolicyStore,
    embedder: Embedder,
    job_store: JobStore,
) -> None:
    try:
        job_store.mark_running(job_id)
        with _get_doc_lock(doc_id):
            result = ingest_document(
                doc_id=doc_id,
                text=text,
                store=store,
                embedder=embedder,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP,
            )
        job_store.mark_succeeded(job_id, result)
    except Exception as error:
        job_store.mark_failed(job_id, str(error))


@router.post("/documents", status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_id: str | None = Form(default=None),
    store: PolicyStore = Depends(get_store),
    embedder: Embedder = Depends(get_embedder),
    job_store: JobStore = Depends(get_job_store),
) -> dict:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="only .md and .txt files are allowed",
        )

    try:
        content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds maximum size of {settings.MAX_UPLOAD_BYTES} bytes",
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="file must be valid UTF-8") from error

    if not text.strip():
        raise HTTPException(status_code=400, detail="file must not be empty")

    resolved_doc_id = doc_id or doc_id_from_filename(filename)
    job_id = job_store.create(resolved_doc_id, filename)
    background_tasks.add_task(
        _ingest_in_background,
        job_id,
        resolved_doc_id,
        text,
        store,
        embedder,
        job_store,
    )
    return {"job_id": job_id, "doc_id": resolved_doc_id, "status": "queued"}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store),
) -> dict:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
