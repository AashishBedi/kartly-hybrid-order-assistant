import json
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from app.config import settings


logger = logging.getLogger("app.observability")
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))


@dataclass
class _RequestMetrics:
    request_id: str
    started_at: float
    llm_calls: list[dict[str, Any]] = field(default_factory=list)


_current_metrics: ContextVar[_RequestMetrics | None] = ContextVar(
    "request_metrics",
    default=None,
)


def start_request(request_id: str) -> None:
    _current_metrics.set(
        _RequestMetrics(request_id=request_id, started_at=time.perf_counter())
    )


def record_llm_call(
    stage: str,
    response: Any | None = None,
    *,
    error: Exception | str | None = None,
    model: str = "",
    latency_ms: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    attempts: int = 0,
) -> None:
    metrics = _current_metrics.get()
    if metrics is None:
        return

    if response is not None:
        model = response.model
        latency_ms = response.latency_ms
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.completion_tokens
        cost_usd = response.cost_usd
        attempts = response.attempts

    metrics.llm_calls.append(
        {
            "stage": stage,
            "model": model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "attempts": attempts,
            "ok": response is not None and error is None,
        }
    )


def finish_request() -> dict[str, Any]:
    metrics = _current_metrics.get()
    if metrics is None:
        raise RuntimeError("No request metrics active")

    result = {
        "request_id": metrics.request_id,
        "total_latency_ms": round((time.perf_counter() - metrics.started_at) * 1000),
        "llm_calls": list(metrics.llm_calls),
        "total_tokens": sum(
            call["prompt_tokens"] + call["completion_tokens"]
            for call in metrics.llm_calls
        ),
        "total_cost_usd": sum(call["cost_usd"] for call in metrics.llm_calls),
    }
    _current_metrics.set(None)
    logger.info(json.dumps(result, separators=(",", ":")))
    return result
