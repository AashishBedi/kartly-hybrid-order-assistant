import hashlib
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

from app.observability import record_llm_call


class LLMError(Exception):
    """A safe, short description of an LLM request failure."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    attempts: int
    finish_reason: str | None = None
    cache_hit: bool = False


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float,
        max_retries: int,
        price_in: float,
        price_out: float,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        cache_dir: Path | str = Path("eval/.cache"),
        cache_enabled: bool | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._price_in = price_in
        self._price_out = price_out
        self._transport = transport
        self._sleep = sleep
        self._cache_dir = Path(cache_dir)
        self._cache_enabled = (
            os.getenv("EVAL_CACHE") == "1"
            if cache_enabled is None
            else cache_enabled
        )

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stage: str | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        attempts = 0
        stage_name = stage or "llm"

        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        cache_path = self._cache_path(body)
        if cache_path is not None:
            cached = self._read_cache(cache_path, model)
            if cached is not None:
                record_llm_call(stage_name, cached)
                return cached

        if not self._api_key:
            error = LLMError("GROQ_API_KEY is not set")
            self._record_failure(
                stage_name, model, started, attempts, error, "transport"
            )
            raise error

        empty_length_retried = False
        retry_count = 0

        with httpx.Client(
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            while True:
                attempts += 1
                try:
                    response = client.post(
                        f"{self._base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json=body,
                    )
                except httpx.TimeoutException as exc:
                    if retry_count >= self._max_retries:
                        error = LLMError("LLM request timed out")
                        self._record_failure(
                            stage_name, model, started, attempts, error, "timeout"
                        )
                        raise error from exc
                    retry_count += 1
                    self._sleep(self._backoff_delay(attempts))
                    continue
                except httpx.TransportError as exc:
                    if retry_count >= self._max_retries:
                        error = LLMError("LLM connection failed")
                        self._record_failure(
                            stage_name, model, started, attempts, error, "transport"
                        )
                        raise error from exc
                    retry_count += 1
                    self._sleep(self._backoff_delay(attempts))
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    if retry_count >= self._max_retries:
                        error = LLMError(f"LLM HTTP {response.status_code}")
                        self._record_failure(
                            stage_name,
                            model,
                            started,
                            attempts,
                            error,
                            f"http_{response.status_code}",
                        )
                        raise error
                    retry_count += 1
                    retry_after = (
                        response.headers.get("Retry-After")
                        if response.status_code == 429
                        else None
                    )
                    self._sleep(self._retry_delay(attempts, retry_after))
                    continue

                if 400 <= response.status_code < 500:
                    error = LLMError(f"LLM HTTP {response.status_code}")
                    self._record_failure(
                        stage_name,
                        model,
                        started,
                        attempts,
                        error,
                        f"http_{response.status_code}",
                    )
                    raise error

                try:
                    payload = response.json()
                except ValueError as exc:
                    error = LLMError("Malformed LLM response")
                    self._record_failure(
                        stage_name, model, started, attempts, error, "parse"
                    )
                    raise error from exc

                try:
                    text, finish_reason = self._extract_choice(payload)
                except LLMError as error:
                    self._record_failure(
                        stage_name, model, started, attempts, error, "parse"
                    )
                    raise

                if text.strip():
                    break
                if finish_reason == "length" and not empty_length_retried:
                    empty_length_retried = True
                    body["max_tokens"] = max_tokens * 2
                    continue

                error = LLMError("LLM returned empty content")
                self._record_failure(
                    stage_name,
                    model,
                    started,
                    attempts,
                    error,
                    "empty",
                    finish_reason=finish_reason,
                )
                raise error

        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            usage = {}

        prompt_tokens = self._token_count(usage.get("prompt_tokens"))
        completion_tokens = self._token_count(usage.get("completion_tokens"))
        cost_usd = (
            prompt_tokens / 1_000_000 * self._price_in
            + completion_tokens / 1_000_000 * self._price_out
        )
        response_model = payload.get("model")
        if not isinstance(response_model, str) or not response_model:
            response_model = model

        result = LLMResponse(
            text=text,
            model=response_model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
            cost_usd=cost_usd,
            attempts=attempts,
        )
        if cache_path is not None:
            self._write_cache(cache_path, result)
        record_llm_call(stage_name, result)
        return result

    def _cache_path(self, body: Mapping[str, Any]) -> Path | None:
        if not self._cache_enabled:
            return None
        serialized = json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(serialized).hexdigest()
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, path: Path, requested_model: str) -> LLMResponse | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            return None
        try:
            text = payload["text"]
            model = payload.get("model", requested_model)
            finish_reason = payload.get("finish_reason")
            prompt_tokens = payload["prompt_tokens"]
            completion_tokens = payload["completion_tokens"]
            latency_ms = payload["latency_ms"]
            cost_usd = payload["cost_usd"]
            attempts = payload["attempts"]
        except KeyError:
            return None
        if not isinstance(text, str) or not text.strip() or not isinstance(model, str):
            return None
        if finish_reason is not None and not isinstance(finish_reason, str):
            return None
        if not all(
            isinstance(value, int) and value >= 0
            for value in (prompt_tokens, completion_tokens, latency_ms, attempts)
        ):
            return None
        if not isinstance(cost_usd, (int, float)) or cost_usd < 0:
            return None
        return LLMResponse(
            text=text,
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=float(cost_usd),
            attempts=attempts,
            cache_hit=True,
        )

    @staticmethod
    def _write_cache(path: Path, response: LLMResponse) -> None:
        if not response.text.strip():
            return
        payload = {
            "ok": True,
            "text": response.text,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "attempts": response.attempts,
        }
        temporary_path: Path | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(path)
        except OSError:
            # The eval cache is an optimization and must never break a request.
            try:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _record_failure(
        stage: str,
        model: str,
        started: float,
        attempts: int,
        error: LLMError,
        error_category: str,
        finish_reason: str | None = None,
    ) -> None:
        record_llm_call(
            stage,
            error=error,
            model=model,
            latency_ms=round((time.perf_counter() - started) * 1000),
            attempts=attempts,
            error_category=error_category,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_choice(payload: Any) -> tuple[str, str | None]:
        if not isinstance(payload, Mapping):
            raise LLMError("Malformed LLM response")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("Malformed LLM response")
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMError("Malformed LLM response")
        message = first_choice.get("message")
        if not isinstance(message, Mapping) or "content" not in message:
            raise LLMError("Malformed LLM response")
        content = message["content"]
        if not isinstance(content, str):
            raise LLMError("Malformed LLM response")
        finish_reason = first_choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise LLMError("Malformed LLM response")
        return content, finish_reason

    @staticmethod
    def _token_count(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.1)

    @classmethod
    def _retry_delay(cls, attempt: int, retry_after: str | None) -> float:
        parsed_retry_after = cls._parse_retry_after(retry_after)
        if parsed_retry_after is not None:
            return min(parsed_retry_after, 5.0)
        return cls._backoff_delay(attempt)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
