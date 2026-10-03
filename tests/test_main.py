import logging

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.deps import get_embedder, get_store
from app.main import app


class FakeEmbedder:
    def __init__(self, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.warmed_up = False

    def warm_up(self) -> None:
        self.warmed_up = True
        if self.should_fail:
            raise RuntimeError("model unavailable")


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_lifespan_warms_up_embedder_and_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "WARMUP_ON_STARTUP", True)
    embedder = FakeEmbedder()
    store = object()
    store_calls = 0

    def open_store() -> object:
        nonlocal store_calls
        store_calls += 1
        return store

    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_store] = open_store

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

    assert embedder.warmed_up is True
    assert store_calls == 1


def test_lifespan_skips_warmup_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "WARMUP_ON_STARTUP", False)

    def fail_if_called() -> None:
        raise AssertionError("warm-up dependency should not be loaded")

    app.dependency_overrides[get_embedder] = fail_if_called
    app.dependency_overrides[get_store] = fail_if_called

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_lifespan_logs_failures_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(settings, "WARMUP_ON_STARTUP", True)
    embedder = FakeEmbedder(should_fail=True)

    def fail_store() -> None:
        raise RuntimeError("store unavailable")

    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_store] = fail_store

    with caplog.at_level(logging.WARNING, logger="app.main"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    assert "Embedder warm-up failed" in caplog.text
    assert "Chroma warm-up failed" in caplog.text
