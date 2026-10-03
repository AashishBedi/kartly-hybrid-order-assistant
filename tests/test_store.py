from pathlib import Path

from app.rag.store import PolicyStore


def make_store(path: Path) -> PolicyStore:
    return PolicyStore(path=path)


def test_add_chunks_then_count(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_chunks(
        doc_id="returns",
        version="v1",
        chunks=["Returns Policy\n\nFirst chunk", "Returns Policy\n\nSecond chunk"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    assert store.count() == 2
    assert store.count_for_doc("returns") == 2
    assert store.list_doc_ids() == ["returns"]


def test_get_doc_version(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_chunks("shipping", "2026-01", ["Shipping Policy\n\nText"], [[1.0, 0.0]])

    assert store.get_doc_version("shipping") == "2026-01"
    assert store.get_doc_version("missing") is None


def test_delete_other_versions_only_affects_requested_doc(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_chunks("returns", "v1", ["Returns v1\n\nOld"], [[1.0, 0.0]])
    store.add_chunks("returns", "v2", ["Returns v2\n\nNew"], [[1.0, 0.0]])
    store.add_chunks("shipping", "v1", ["Shipping\n\nKeep"], [[0.0, 1.0]])

    store.delete_other_versions("returns", keep_version="v2")

    assert store.count_for_doc("returns") == 1
    assert store.get_doc_version("returns") == "v2"
    assert store.count_for_doc("shipping") == 1
    assert store.count() == 2


def test_query_returns_closest_chunk_first(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add_chunks(
        "returns",
        "v1",
        ["Return window\n\nClose", "Return window\n\nFar"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = store.query([1.0, 0.0], top_k=2)

    assert results[0]["chunk_id"] == "returns:v1:0"
    assert results[0]["doc_id"] == "returns"
    assert results[0]["text"] == "Return window\n\nClose"
    assert results[0]["distance"] < results[1]["distance"]
