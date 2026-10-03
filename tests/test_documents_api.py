import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.deps import get_embedder, get_job_store, get_store
from app.jobs.store import JobStore
from app.main import app
from app.rag.store import PolicyStore


ORIGINAL_TEXT = b"""# Returns Policy

## Return window

Items may be returned within 30 days of delivery.
"""

CHANGED_TEXT = b"""# Returns Policy

## Return window

Items may be returned within 45 days of delivery.
"""


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.should_fail = False

    def warm_up(self) -> None:
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.should_fail:
            raise RuntimeError("embedding failed")

        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([(value + 1) / 256 for value in digest[:4]])
        return vectors


@pytest.fixture
def api(tmp_path: Path):
    store = PolicyStore(tmp_path / "chroma")
    job_store = JobStore(tmp_path / "jobs.db")
    embedder = FakeEmbedder()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_job_store] = lambda: job_store

    with TestClient(app) as client:
        yield client, store, job_store, embedder

    app.dependency_overrides.clear()


def upload(
    client: TestClient,
    content: bytes,
    filename: str = "Returns Policy.md",
    doc_id: str | None = None,
):
    form_data = {"doc_id": doc_id} if doc_id is not None else {}
    return client.post(
        "/documents",
        files={"file": (filename, content, "text/plain")},
        data=form_data,
    )


def get_job(client: TestClient, job_id: str) -> dict:
    response = client.get(f"/jobs/{job_id}")
    assert response.status_code == 200
    return response.json()


def test_markdown_upload_succeeds(api) -> None:
    client, _, _, _ = api

    response = upload(client, ORIGINAL_TEXT)

    assert response.status_code == 202
    assert response.json()["doc_id"] == "returns_policy"
    assert response.json()["status"] == "queued"
    job = get_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert job["result"]["chunks"] > 0


def test_changed_upload_replaces_old_version(api) -> None:
    client, store, _, _ = api
    first_response = upload(client, ORIGINAL_TEXT, doc_id="returns")
    first_job = get_job(client, first_response.json()["job_id"])
    old_version = first_job["result"]["version"]

    second_response = upload(client, CHANGED_TEXT, doc_id="returns")
    second_job = get_job(client, second_response.json()["job_id"])
    stored = store.collection.get(
        where={"doc_id": "returns"},
        include=["metadatas"],
    )

    assert second_job["result"]["status"] == "replaced"
    assert second_job["result"]["version"] != old_version
    assert all(
        metadata["version"] == second_job["result"]["version"]
        for metadata in stored["metadatas"]
    )


def test_identical_upload_is_unchanged(api) -> None:
    client, _, _, embedder = api
    first_response = upload(client, ORIGINAL_TEXT, doc_id="returns")
    first_job = get_job(client, first_response.json()["job_id"])

    second_response = upload(client, ORIGINAL_TEXT, doc_id="returns")
    second_job = get_job(client, second_response.json()["job_id"])

    assert first_job["status"] == "succeeded"
    assert second_job["status"] == "succeeded"
    assert second_job["result"]["status"] == "unchanged"
    assert embedder.calls == 1


def test_pdf_upload_is_rejected(api) -> None:
    client, _, _, _ = api

    response = upload(client, b"PDF content", filename="policy.pdf")

    assert response.status_code == 400
    assert response.json() == {"detail": "only .md and .txt files are allowed"}


def test_empty_upload_is_rejected(api) -> None:
    client, _, _, _ = api

    response = upload(client, b" \n\t")

    assert response.status_code == 400
    assert response.json() == {"detail": "file must not be empty"}


def test_non_utf8_upload_is_rejected(api) -> None:
    client, _, _, _ = api

    response = upload(client, b"\xff\xfe")

    assert response.status_code == 400
    assert response.json() == {"detail": "file must be valid UTF-8"}


def test_oversized_upload_is_rejected(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _, _ = api
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 10)

    response = upload(client, b"12345678901")

    assert response.status_code == 413
    assert response.json() == {"detail": "file exceeds maximum size of 10 bytes"}


def test_unknown_job_returns_404(api) -> None:
    client, _, _, _ = api

    response = client.get("/jobs/unknown-job-id")

    assert response.status_code == 404
    assert response.json() == {"detail": "job not found"}


def test_embedding_failure_preserves_previous_version(api) -> None:
    client, store, _, embedder = api
    first_response = upload(client, ORIGINAL_TEXT, doc_id="returns")
    first_job = get_job(client, first_response.json()["job_id"])
    old_version = first_job["result"]["version"]
    old_ids = set(
        store.collection.get(where={"doc_id": "returns"}, include=[])["ids"]
    )
    embedder.should_fail = True

    failed_response = upload(client, CHANGED_TEXT, doc_id="returns")
    failed_job = get_job(client, failed_response.json()["job_id"])
    current_ids = set(
        store.collection.get(where={"doc_id": "returns"}, include=[])["ids"]
    )

    assert failed_response.status_code == 202
    assert failed_job["status"] == "failed"
    assert failed_job["result"] is None
    assert failed_job["error"] == "embedding failed"
    assert store.get_doc_version("returns") == old_version
    assert current_ids == old_ids
