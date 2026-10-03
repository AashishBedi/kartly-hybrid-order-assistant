import hashlib
from pathlib import Path

import pytest

from app.rag.ingest import doc_id_from_filename, ingest_document
from app.rag.store import PolicyStore


ORIGINAL_TEXT = """# Returns Policy

## Return window

Items may be returned within 30 days of delivery.

Items must be unused and in their original packaging.
"""

CHANGED_TEXT = """# Returns Policy

## Return window

Items may be returned within 30 days of delivery.

Final-sale items cannot be returned.
"""


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.should_fail = False

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
def store(tmp_path: Path) -> PolicyStore:
    return PolicyStore(tmp_path / "chroma")


def ingest(text: str, store: PolicyStore, embedder: FakeEmbedder, doc_id: str = "returns") -> dict:
    return ingest_document(
        doc_id=doc_id,
        text=text,
        store=store,
        embedder=embedder,
        chunk_size=70,
        overlap=10,
    )


def test_first_ingest_stores_all_chunks(store: PolicyStore) -> None:
    embedder = FakeEmbedder()

    result = ingest(ORIGINAL_TEXT, store, embedder)

    assert result["status"] == "ingested"
    assert result["chunks"] > 0
    assert store.count() == result["chunks"]
    assert store.count_for_doc("returns") == result["chunks"]


def test_identical_reingest_is_unchanged_without_embedding(store: PolicyStore) -> None:
    embedder = FakeEmbedder()
    first_result = ingest(ORIGINAL_TEXT, store, embedder)
    original_count = store.count_for_doc("returns")

    second_result = ingest(ORIGINAL_TEXT, store, embedder)

    assert second_result["status"] == "unchanged"
    assert second_result["version"] == first_result["version"]
    assert second_result["chunks"] == original_count
    assert store.count_for_doc("returns") == original_count
    assert embedder.calls == 1


def test_changed_text_replaces_old_version_without_duplicate_ids(
    store: PolicyStore,
) -> None:
    embedder = FakeEmbedder()
    first_result = ingest(ORIGINAL_TEXT, store, embedder)

    second_result = ingest(CHANGED_TEXT, store, embedder)
    stored = store.collection.get(
        where={"doc_id": "returns"},
        include=["metadatas"],
    )

    assert second_result["status"] == "replaced"
    assert second_result["version"] != first_result["version"]
    assert all(
        metadata["version"] == second_result["version"]
        for metadata in stored["metadatas"]
    )
    assert len(stored["ids"]) == len(set(stored["ids"]))
    assert store.count_for_doc("returns") == second_result["chunks"]


def test_embedding_failure_keeps_old_version(store: PolicyStore) -> None:
    embedder = FakeEmbedder()
    first_result = ingest(ORIGINAL_TEXT, store, embedder)
    old_ids = set(
        store.collection.get(where={"doc_id": "returns"}, include=[])["ids"]
    )
    old_count = store.count_for_doc("returns")
    embedder.should_fail = True

    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest(CHANGED_TEXT, store, embedder)

    current_ids = set(
        store.collection.get(where={"doc_id": "returns"}, include=[])["ids"]
    )
    assert store.get_doc_version("returns") == first_result["version"]
    assert store.count_for_doc("returns") == old_count
    assert current_ids == old_ids


def test_different_doc_ids_do_not_affect_each_other(store: PolicyStore) -> None:
    embedder = FakeEmbedder()
    ingest(ORIGINAL_TEXT, store, embedder, doc_id="returns")
    shipping_result = ingest(ORIGINAL_TEXT, store, embedder, doc_id="shipping")
    shipping_count = store.count_for_doc("shipping")

    ingest(CHANGED_TEXT, store, embedder, doc_id="returns")

    assert store.get_doc_version("shipping") == shipping_result["version"]
    assert store.count_for_doc("shipping") == shipping_count
    assert store.list_doc_ids() == ["returns", "shipping"]


def test_empty_text_raises_value_error(store: PolicyStore) -> None:
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="document text must not be empty"):
        ingest(" \n\t ", store, embedder)

    assert embedder.calls == 0
    assert store.count() == 0


def test_doc_id_from_filename() -> None:
    assert doc_id_from_filename("Returns Policy.md") == "returns_policy"
