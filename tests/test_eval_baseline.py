from typing import Any

from app.llm.client import LLMResponse
from eval.baseline import answer_baseline


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
