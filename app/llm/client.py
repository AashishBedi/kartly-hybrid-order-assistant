import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx


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
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max(0, max_retries)
        self._price_in = price_in
        self._price_out = price_out
        self._transport = transport
        self._sleep = sleep

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> LLMResponse:
        if not self._api_key:
            raise LLMError("GROQ_API_KEY is not set")

        body: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        attempts = 0
        response: httpx.Response | None = None

        with httpx.Client(
            timeout=self._timeout,
            transport=self._transport,
        ) as client:
            while attempts <= self._max_retries:
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
                    if attempts > self._max_retries:
                        raise LLMError("LLM request timed out") from exc
                    self._sleep(self._backoff_delay(attempts))
                    continue
                except httpx.NetworkError as exc:
                    if attempts > self._max_retries:
                        raise LLMError("LLM connection failed") from exc
                    self._sleep(self._backoff_delay(attempts))
                    continue

                if response.status_code == 429 or response.status_code >= 500:
                    if attempts > self._max_retries:
                        raise LLMError(f"LLM HTTP {response.status_code}")
                    retry_after = (
                        response.headers.get("Retry-After")
                        if response.status_code == 429
                        else None
                    )
                    self._sleep(self._retry_delay(attempts, retry_after))
                    continue

                if 400 <= response.status_code < 500:
                    raise LLMError(f"LLM HTTP {response.status_code}")

                break

        if response is None:
            raise LLMError("LLM request failed")

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError("Malformed LLM response") from exc

        text = self._extract_content(payload)
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

        return LLMResponse(
            text=text,
            model=response_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round((time.perf_counter() - started) * 1000),
            cost_usd=cost_usd,
            attempts=attempts,
        )

    @staticmethod
    def _extract_content(payload: Any) -> str:
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
        return content

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
