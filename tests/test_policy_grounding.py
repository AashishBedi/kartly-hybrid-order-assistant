from pathlib import Path

import pytest

from app.pipeline.data_evidence import DataEvidence
from app.pipeline.grounding import (
    ORDER_NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    POLICY_NOT_FOUND_MESSAGE,
    check_grounding,
)
from app.pipeline.policy_evidence import (
    PolicyEvidence,
    gather_policy,
    strip_order_reference,
)
from app.rag.store import PolicyStore
from app.routing.router import Route


class FakeEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0]
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vector[:] for _ in texts]


@pytest.fixture
def store(tmp_path: Path) -> PolicyStore:
    return PolicyStore(tmp_path / "chroma")


def test_chunks_above_threshold_are_dropped(store: PolicyStore) -> None:
    store.add_chunks(
        doc_id="returns",
        version="v1",
        chunks=["Relevant returns policy", "Unrelated shipping policy"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    embedder = FakeEmbedder()

    evidence = gather_policy(
        "Can I return order 1042?",
        store,
        embedder,
        top_k=2,
        max_distance=0.62,
    )

    assert evidence.found is True
    assert evidence.best_distance == pytest.approx(0.0)
    assert len(evidence.chunks) == 1
    assert evidence.chunks[0]["text"] == "Relevant returns policy"
    assert evidence.chunks[0]["distance"] == pytest.approx(0.0)
    assert embedder.calls == [["Can I return?"]]


def test_found_is_false_when_no_chunks_meet_threshold(store: PolicyStore) -> None:
    store.add_chunks(
        doc_id="shipping",
        version="v1",
        chunks=["Shipping policy"],
        embeddings=[[0.0, 1.0]],
    )

    evidence = gather_policy(
        "What is the return policy?",
        store,
        FakeEmbedder(),
        top_k=1,
        max_distance=0.62,
    )

    assert evidence.chunks == []
    assert evidence.found is False
    assert evidence.best_distance == pytest.approx(1.0)


def test_empty_store_skips_embedding_and_has_no_best_distance(
    store: PolicyStore,
) -> None:
    embedder = FakeEmbedder()

    evidence = gather_policy(
        "What is the return policy?",
        store,
        embedder,
        top_k=4,
        max_distance=0.62,
    )

    assert evidence.chunks == []
    assert evidence.found is False
    assert evidence.best_distance is None
    assert embedder.calls == []


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Can I return order 1042?", "Can I return?"),
        ("What is the warranty for Order #1042?", "What is the warranty for?"),
        ("Does #1042 qualify for a refund?", "Does qualify for a refund?"),
        ("Compare order 1042 and order 2048.", "Compare and."),
        ("Is the return window 30 days?", "Is the return window 30 days?"),
    ],
)
def test_strip_order_reference(question: str, expected: str) -> None:
    assert strip_order_reference(question) == expected


def _data_evidence(found: bool) -> DataEvidence:
    return DataEvidence(
        tool="list_orders",
        query_results=[],
        rows=[{}] if found else [],
        found=found,
        facts={},
    )


def _policy_evidence(found: bool) -> PolicyEvidence:
    return PolicyEvidence(
        chunks=(
            [
                {
                    "chunk_id": "returns:v1:0",
                    "doc_id": "returns",
                    "text": "Returns policy",
                    "distance": 0.1,
                }
            ]
            if found
            else []
        ),
        found=found,
        best_distance=0.1 if found else None,
    )


@pytest.mark.parametrize(
    ("route", "data_found", "policy_found", "ok", "message"),
    [
        ("data", False, False, False, ORDER_NOT_FOUND_MESSAGE),
        ("data", False, True, False, ORDER_NOT_FOUND_MESSAGE),
        ("data", True, False, True, None),
        ("data", True, True, True, None),
        ("policy", False, False, False, POLICY_NOT_FOUND_MESSAGE),
        ("policy", False, True, True, None),
        ("policy", True, False, False, POLICY_NOT_FOUND_MESSAGE),
        ("policy", True, True, True, None),
        ("combined", False, False, False, ORDER_NOT_FOUND_MESSAGE),
        ("combined", False, True, False, ORDER_NOT_FOUND_MESSAGE),
        ("combined", True, False, False, POLICY_NOT_FOUND_MESSAGE),
        ("combined", True, True, True, None),
        ("out_of_scope", False, False, False, OUT_OF_SCOPE_MESSAGE),
        ("out_of_scope", False, True, False, OUT_OF_SCOPE_MESSAGE),
        ("out_of_scope", True, False, False, OUT_OF_SCOPE_MESSAGE),
        ("out_of_scope", True, True, False, OUT_OF_SCOPE_MESSAGE),
    ],
)
def test_check_grounding_for_all_routes_and_evidence_combinations(
    route: Route,
    data_found: bool,
    policy_found: bool,
    ok: bool,
    message: str | None,
) -> None:
    result = check_grounding(
        route,
        _data_evidence(data_found),
        _policy_evidence(policy_found),
    )

    assert result.ok is ok
    assert result.message == message
