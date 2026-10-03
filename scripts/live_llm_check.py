import logging
import sys
from uuid import uuid4

from app.config import settings
from app.llm.client import LLMClient, LLMError
from app.observability import finish_request, start_request


def main() -> None:
    if not settings.GROQ_API_KEY:
        print("GROQ_API_KEY is not set. Add it to your local .env and try again.")
        raise SystemExit(1)

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )
    client = LLMClient(
        api_key=settings.GROQ_API_KEY,
        base_url=settings.GROQ_BASE_URL,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=settings.LLM_MAX_RETRIES,
        price_in=settings.PRICE_INPUT_PER_MTOK,
        price_out=settings.PRICE_OUTPUT_PER_MTOK,
    )
    start_request(f"live-llm-check-{uuid4()}")

    try:
        response = client.chat(
            settings.ROUTER_MODEL,
            [{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0,
            max_tokens=5,
            stage="router",
        )
    except LLMError as exc:
        finish_request()
        print(f"Live LLM check failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"text: {response.text}")
    print(
        "tokens: "
        f"prompt={response.prompt_tokens}, "
        f"completion={response.completion_tokens}"
    )
    print(f"latency_ms: {response.latency_ms}")
    print(f"cost_usd: {response.cost_usd:.10f}")
    finish_request()


if __name__ == "__main__":
    main()
