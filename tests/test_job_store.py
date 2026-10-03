from pathlib import Path

from app.jobs.store import JobStore


def test_create_then_get_shows_queued(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")

    job_id = store.create("returns_policy", "Returns Policy.md")
    job = store.get(job_id)

    assert job is not None
    assert job["id"] == job_id
    assert job["doc_id"] == "returns_policy"
    assert job["filename"] == "Returns Policy.md"
    assert job["status"] == "queued"
    assert job["result"] is None
    assert job["error"] is None
    assert job["created_at"] == job["updated_at"]


def test_running_and_succeeded_transitions(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create("shipping_policy", "shipping_policy.md")

    store.mark_running(job_id)
    running_job = store.get(job_id)
    assert running_job is not None
    assert running_job["status"] == "running"

    result = {"status": "ingested", "chunks": 3}
    store.mark_succeeded(job_id, result)
    succeeded_job = store.get(job_id)
    assert succeeded_job is not None
    assert succeeded_job["status"] == "succeeded"
    assert succeeded_job["result"] == result
    assert succeeded_job["error"] is None


def test_failed_transition(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create("refund_policy", "refund_policy.md")

    store.mark_running(job_id)
    store.mark_failed(job_id, "embedding failed")
    job = store.get(job_id)

    assert job is not None
    assert job["status"] == "failed"
    assert job["result"] is None
    assert job["error"] == "embedding failed"


def test_unknown_job_returns_none(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")

    assert store.get("missing-job-id") is None
