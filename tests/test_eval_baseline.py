from typing import Any

from app.llm.client import LLMResponse
from eval.baseline import answer_baseline, answer_vector_only


class RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return LLMResponse(
            text="A generic answer",
            model=model,
            prompt_tokens=11,
            completion_tokens=4,
            latency_ms=23,
            cost_usd=0.001,
            attempts=1,
        )


class RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.25, 0.75]]


class RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[float], int]] = []

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        self.calls.append((embedding, top_k))
        return [
            {
                "chunk_id": "returns:v1:0",
                "doc_id": "returns",
                "text": "Returns are accepted within 30 days.",
                "distance": 0.12,
            },
            {
                "chunk_id": "refunds:v2:1",
                "doc_id": "refunds",
                "text": "Refunds are issued after inspection.",
                "distance": 0.91,
            },
        ]


def test_baseline_makes_one_call_without_retrieved_context(monkeypatch: Any) -> None:
    client = RecordingClient()
    monkeypatch.setattr("eval.baseline.get_llm_client", lambda: client)
    monkeypatch.setattr("eval.baseline.settings.ANSWER_MODEL", "answer-model")

    result = answer_baseline("Where is my order?")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "answer-model"
    assert call["messages"] == [
        {
            "role": "system",
            "content": (
                "You are a customer support assistant for Kartly, an online store."
            ),
        },
        {"role": "user", "content": "Where is my order?"},
    ]
    prompt = " ".join(message["content"] for message in call["messages"])
    assert "Policy chunks" not in prompt
    assert "computed_facts" not in prompt
    assert "customer_id" not in prompt
    assert result == {
        "answer": "A generic answer",
        "model": "answer-model",
        "latency_ms": 23,
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "cost_usd": 0.001,
        "attempts": 1,
        "cache_hit": False,
    }


def test_vector_only_baseline_retrieves_chunks_and_makes_one_llm_call(
    monkeypatch: Any,
) -> None:
    client = RecordingClient()
    embedder = RecordingEmbedder()
    store = RecordingStore()
    monkeypatch.setattr("eval.baseline.settings.ANSWER_MODEL", "answer-model")

    result = answer_vector_only(
        "What is the return policy?",
        store=store,
        embedder=embedder,
        llm_client=client,
    )

    assert embedder.calls == [["What is the return policy?"]]
    assert store.calls == [([0.25, 0.75], 4)]
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["model"] == "answer-model"
    assert call["messages"][0] == {
        "role": "system",
        "content": (
            "You are a customer support assistant for Kartly. Answer using only "
            "the policy excerpts provided. If they do not contain the answer, "
            "say you cannot answer from the policy documents."
        ),
    }
    prompt = call["messages"][1]["content"]
    assert "Returns are accepted within 30 days." in prompt
    assert "Refunds are issued after inspection." in prompt
    assert "customer_id" not in prompt
    assert "sql" not in prompt.casefold()
    assert call["stage"] == "vector_baseline"
    assert call["temperature"] == 0
    assert call["max_tokens"] == 400
    assert result == {
        "answer": "A generic answer",
        "model": "answer-model",
        "latency_ms": 23,
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "total_tokens": 15,
        "cost_usd": 0.001,
        "attempts": 1,
        "cache_hit": False,
        "chunk_ids": ["returns:v1:0", "refunds:v2:1"],
    }
