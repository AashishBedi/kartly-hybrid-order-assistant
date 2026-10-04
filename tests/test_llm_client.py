import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from app.llm.client import LLMClient, LLMError
from app.observability import finish_request, start_request


BASE_URL = "https://api.groq.test/openai/v1"
MODEL = "test-model"


def llm_response(
    *,
    text: str = "ok",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    finish_reason: str | None = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "choices": [
                {
                    "message": {"content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        },
    )


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    waits: list[float] | None = None,
    *,
    api_key: str = "test-api-key",
    max_retries: int = 2,
    price_in: float = 1.0,
    price_out: float = 2.0,
    cache_dir: Path | str = Path("eval/.cache"),
) -> LLMClient:
    if waits is None:
        waits = []
    return LLMClient(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=1,
        max_retries=max_retries,
        price_in=price_in,
        price_out=price_out,
        transport=httpx.MockTransport(handler),
        sleep=waits.append,
        cache_dir=cache_dir,
    )


def test_eval_cache_miss_then_hit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVAL_CACHE", "1")
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return llm_response(text="cached", prompt_tokens=12, completion_tokens=3)

    client = make_client(handler, cache_dir=tmp_path)
    messages = [{"role": "user", "content": "hello"}]

    miss = client.chat(MODEL, messages)
    hit = client.chat(MODEL, messages)

    assert request_count == 1
    assert miss.cache_hit is False
    assert hit.cache_hit is True
    assert hit.text == miss.text
    assert hit.finish_reason == miss.finish_reason == "stop"
    assert hit.prompt_tokens == miss.prompt_tokens
    assert hit.completion_tokens == miss.completion_tokens
    assert hit.latency_ms == miss.latency_ms
    assert len(list(tmp_path.glob("*.json"))) == 1
    cache_payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cache_payload["finish_reason"] == "stop"


def test_blank_length_response_retries_once_with_double_tokens_and_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVAL_CACHE", "1")
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return llm_response(text=" \n", finish_reason="length")

    client = make_client(handler, cache_dir=tmp_path)

    start_request("blank-test")
    with pytest.raises(LLMError, match="empty content"):
        client.chat(MODEL, [], max_tokens=50)
    metrics = finish_request()

    assert [body["max_tokens"] for body in request_bodies] == [50, 100]
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
    assert metrics["llm_calls"][0]["finish_reason"] == "length"
    assert metrics["llm_calls"][0]["error_category"] == "empty"


def test_blank_length_response_can_recover_on_retry() -> None:
    responses = iter(
        [
            llm_response(text="", finish_reason="length"),
            llm_response(text="recovered", finish_reason="stop"),
        ]
    )
    client = make_client(lambda request: next(responses))

    response = client.chat(MODEL, [], max_tokens=25)

    assert response.text == "recovered"
    assert response.finish_reason == "stop"
    assert response.attempts == 2


def test_blank_cache_entry_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVAL_CACHE", "1")
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return llm_response(text="fresh")

    client = make_client(handler, cache_dir=tmp_path)
    body = {
        "model": MODEL,
        "messages": [],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    cache_path = client._cache_path(body)
    assert cache_path is not None
    cache_path.write_text(
        json.dumps(
            {
                "ok": True,
                "text": "  ",
                "model": MODEL,
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_ms": 1,
                "cost_usd": 0.0,
                "attempts": 1,
            }
        ),
        encoding="utf-8",
    )

    response = client.chat(MODEL, [])

    assert request_count == 1
    assert response.text == "fresh"
    assert response.cache_hit is False


def test_eval_cache_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EVAL_CACHE", raising=False)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return llm_response()

    client = make_client(handler, cache_dir=tmp_path)
    client.chat(MODEL, [])
    client.chat(MODEL, [])

    assert request_count == 2
    assert list(tmp_path.iterdir()) == []


def test_failed_calls_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EVAL_CACHE", "1")
    client = make_client(
        lambda request: httpx.Response(401),
        cache_dir=tmp_path,
    )

    with pytest.raises(LLMError, match="LLM HTTP 401"):
        client.chat(MODEL, [])

    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_success_returns_text_usage_latency_and_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([10.0, 10.025])
    monkeypatch.setattr(
        "app.llm.client.time",
        SimpleNamespace(perf_counter=lambda: next(clock)),
    )
    client = make_client(lambda request: llm_response())

    response = client.chat(MODEL, [{"role": "user", "content": "hello"}])

    assert response.text == "ok"
    assert response.finish_reason == "stop"
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 20
    assert response.latency_ms == 25
    assert response.latency_ms > 0
    assert response.cost_usd == pytest.approx(0.00014)


def test_json_mode_adds_response_format() -> None:
    request_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return llm_response()

    client = make_client(handler)

    client.chat(MODEL, [], json_mode=True)

    assert request_bodies[0]["response_format"] == {"type": "json_object"}


def test_retries_two_server_errors_then_succeeds() -> None:
    responses = iter(
        [
            httpx.Response(500),
            httpx.Response(500),
            llm_response(),
        ]
    )
    waits: list[float] = []
    client = make_client(lambda request: next(responses), waits)

    response = client.chat(MODEL, [])

    assert response.attempts == 3
    assert len(waits) == 2
    assert waits[0] < waits[1]


def test_retry_after_header_controls_429_wait() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}),
            llm_response(),
        ]
    )
    waits: list[float] = []
    client = make_client(lambda request: next(responses), waits)

    response = client.chat(MODEL, [])

    assert response.attempts == 2
    assert waits == pytest.approx([2.0])


def test_timeout_exhausts_max_retries_plus_one_attempt() -> None:
    request_count = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client = make_client(handler, waits, max_retries=2)

    with pytest.raises(LLMError, match="LLM request timed out"):
        client.chat(MODEL, [])

    assert request_count == 3
    assert len(waits) == 2


def test_unauthorized_is_not_retried() -> None:
    request_count = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(401)

    client = make_client(handler, waits)

    with pytest.raises(LLMError, match="LLM HTTP 401"):
        client.chat(MODEL, [])

    assert request_count == 1
    assert waits == []


def test_missing_choices_raises_llm_error() -> None:
    client = make_client(lambda request: httpx.Response(200, json={"model": MODEL}))

    with pytest.raises(LLMError, match="Malformed LLM response"):
        client.chat(MODEL, [])


def test_empty_api_key_fails_before_transport_is_called() -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return llm_response()

    client = make_client(handler, api_key="")

    with pytest.raises(LLMError, match="GROQ_API_KEY is not set"):
        client.chat(MODEL, [])

    assert request_count == 0


def test_api_key_is_absent_from_errors_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = "secret-key-that-must-not-leak"
    client = make_client(
        lambda request: httpx.Response(401),
        api_key=api_key,
    )
    start_request("secret-check")

    with caplog.at_level(logging.INFO, logger="app.observability"):
        with pytest.raises(LLMError) as exc_info:
            client.chat(MODEL, [], stage="router")
        finish_request()

    assert api_key not in str(exc_info.value)
    assert api_key not in caplog.text


def test_metrics_record_one_entry_per_call_and_totals() -> None:
    success_client = make_client(
        lambda request: llm_response(prompt_tokens=30, completion_tokens=10),
        price_in=1.0,
        price_out=2.0,
    )
    failure_client = make_client(lambda request: httpx.Response(401))
    start_request("metrics-test")

    response = success_client.chat(MODEL, [], stage="router")
    with pytest.raises(LLMError):
        failure_client.chat(MODEL, [], stage="answer")
    metrics = finish_request()

    assert len(metrics["llm_calls"]) == 2
    assert metrics["llm_calls"][0]["stage"] == "router"
    assert metrics["llm_calls"][0]["ok"] is True
    assert metrics["llm_calls"][0]["finish_reason"] == "stop"
    assert metrics["llm_calls"][1]["stage"] == "answer"
    assert metrics["llm_calls"][1]["ok"] is False
    assert metrics["llm_calls"][1]["finish_reason"] is None
    assert metrics["llm_calls"][1]["error_category"] == "http_401"
    assert metrics["total_tokens"] == 40
    assert metrics["total_cost_usd"] == pytest.approx(response.cost_usd)
