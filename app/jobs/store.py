import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings


CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    filename TEXT,
    status TEXT CHECK(status IN ('queued', 'running', 'succeeded', 'failed')),
    result TEXT,
    error TEXT,
    created_at TEXT,
    updated_at TEXT
)
"""


def _current_time() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: str | Path = settings.JOBS_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(CREATE_JOBS_TABLE)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, doc_id: str, filename: str) -> str:
        job_id = str(uuid.uuid4())
        now = _current_time()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO jobs
                        (id, doc_id, filename, status, result, error,
                         created_at, updated_at)
                    VALUES (?, ?, ?, 'queued', NULL, NULL, ?, ?)
                    """,
                    (job_id, doc_id, filename, now, now),
                )
        return job_id

    def get(self, job_id: str) -> dict | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()

        if row is None:
            return None

        job = dict(row)
        if job["result"] is not None:
            job["result"] = json.loads(job["result"])
        return job

    def _update(
        self,
        job_id: str,
        status: str,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        result_json = json.dumps(result) if result is not None else None
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE jobs
                    SET status = ?, result = ?, error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (status, result_json, error, _current_time(), job_id),
                )

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, "running")

    def mark_succeeded(self, job_id: str, result_dict: dict) -> None:
        self._update(job_id, "succeeded", result=result_dict)

    def mark_failed(self, job_id: str, error_str: str) -> None:
        self._update(job_id, "failed", error=error_str)
