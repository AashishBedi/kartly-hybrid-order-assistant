"""Run Kartly's hybrid and LLM-only systems and save unjudged results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings
from app.db.connection import get_readonly_connection
from app.deps import get_embedder, get_llm_client, get_store
from app.llm.client import LLMClient
from app.pipeline.ask import handle_ask
from eval.baseline import answer_baseline
from eval.resolve import resolve_case


EVAL_DIR = Path(__file__).resolve().parent
CASES_PATH = EVAL_DIR / "cases.json"
RESULTS_DIR = EVAL_DIR / "results"


class EvaluationInterrupted(KeyboardInterrupt):
    """An interrupted run whose partial output can be resumed."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        super().__init__(str(output_path))


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load the checked-in evaluation cases without modifying them."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(
        isinstance(case, dict) for case in payload
    ):
        raise ValueError("Evaluation cases must be a JSON array of objects")
    return payload


def run_evaluation(
    *,
    limit: int | None = None,
    ids: set[str] | None = None,
    category: str | None = None,
    resume: Path | None = None,
    sleep_seconds: float = 1.0,
    cases_path: Path = CASES_PATH,
    results_dir: Path = RESULTS_DIR,
    sleeper: Callable[[float], None] = time.sleep,
) -> Path:
    """Evaluate cases with caching enabled only for the duration of the run."""
    previous_cache_setting = os.environ.get("EVAL_CACHE")
    os.environ["EVAL_CACHE"] = "1"
    cache_clear = getattr(get_llm_client, "cache_clear", None)
    if callable(cache_clear):
        cache_clear()
    try:
        return _run_evaluation(
            limit=limit,
            ids=ids,
            category=category,
            resume=resume,
            sleep_seconds=sleep_seconds,
            cases_path=cases_path,
            results_dir=results_dir,
            sleeper=sleeper,
        )
    finally:
        if callable(cache_clear):
            cache_clear()
        if previous_cache_setting is None:
            os.environ.pop("EVAL_CACHE", None)
        else:
            os.environ["EVAL_CACHE"] = previous_cache_setting


def _run_evaluation(
    *,
    limit: int | None,
    ids: set[str] | None,
    category: str | None,
    resume: Path | None,
    sleep_seconds: float,
    cases_path: Path,
    results_dir: Path,
    sleeper: Callable[[float], None],
) -> Path:
    """Evaluate selected cases and append one raw JSON object per case."""

    cases = _select_cases(
        load_cases(cases_path),
        ids=ids,
        category=category,
        limit=limit,
    )
    completed_ids: set[str] = set()
    if resume is not None:
        output_path = resume.resolve()
        completed_ids = _read_completed_ids(output_path)
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output_path = results_dir / f"raw_{timestamp}.jsonl"

    pending = [case for case in cases if str(case["id"]) not in completed_ids]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        output_path.touch(exist_ok=True)
        return output_path
    output_path.touch(exist_ok=True)

    connection = None
    try:
        store = get_store()
        embedder = get_embedder()
        llm_client = get_llm_client()
        connection = get_readonly_connection()
        evaluation_date = date.today()

        with output_path.open("a", encoding="utf-8", newline="\n") as output:
            for index, case in enumerate(pending, start=1):
                total = len(pending)
                case_id = str(case["id"])
                print(
                    f"[{index}/{total}] {case_id} starting",
                    file=sys.stderr,
                    flush=True,
                )
                timings: dict[str, float] = {}
                try:
                    record = _evaluate_case(
                        case,
                        connection=connection,
                        store=store,
                        embedder=embedder,
                        llm_client=llm_client,
                        evaluation_date=evaluation_date,
                        timings=timings,
                    )
                except Exception as exc:  # Keep an unexpected case failure isolated.
                    record = _empty_record(case)
                    record["error"] = f"runner: {_safe_error(exc)}"
                record = _redact_api_key(record)
                output.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output.flush()
                _print_completed_progress(
                    index=index,
                    total=total,
                    case=case,
                    record=record,
                    timings=timings,
                )
                if index < total and sleep_seconds > 0:
                    sleeper(sleep_seconds)
    except KeyboardInterrupt as exc:
        raise EvaluationInterrupted(output_path) from exc
    finally:
        if connection is not None:
            connection.close()

    return output_path


def _evaluate_case(
    case: dict[str, Any],
    *,
    connection: Any,
    store: Any,
    embedder: Any,
    llm_client: LLMClient,
    evaluation_date: date,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    record = _empty_record(case)
    errors: list[str] = []

    try:
        record["resolved_checks"] = resolve_case(
            case,
            connection,
            evaluation_date,
        )
    except Exception as exc:
        errors.append(f"resolve: {_safe_error(exc)}")

    captured_metrics: list[dict[str, Any]] = []
    hybrid_started = time.perf_counter()
    try:
        hybrid = handle_ask(
            customer_id=int(case["customer_id"]),
            question=str(case["question"]),
            store=store,
            embedder=embedder,
            llm_client=llm_client,
            conn=connection,
            today=evaluation_date,
            router_model=settings.ROUTER_MODEL,
            answer_model=settings.ANSWER_MODEL,
            top_k=settings.RETRIEVAL_TOP_K,
            max_distance=settings.RELEVANCE_MAX_DISTANCE,
            metrics_sink=captured_metrics.append,
        )
        sources = hybrid.get("sources", {"sql": [], "chunks": []})
        record.update(
            {
                "hybrid_answer": hybrid.get("answer"),
                "degraded": hybrid.get("degraded"),
                "route": hybrid.get("route"),
                "tools_used": list(
                    dict.fromkeys(
                        source["tool"]
                        for source in sources.get("sql", [])
                        if source.get("tool")
                    )
                ),
                "evidence_sources": sources,
                "hybrid_blocked": hybrid.get("blocked"),
                "hybrid_grounded": hybrid.get("grounded"),
                "hybrid_request_id": hybrid.get("request_id"),
            }
        )
    except Exception as exc:
        errors.append(f"hybrid: {_safe_error(exc)}")
    finally:
        if timings is not None:
            timings["hybrid"] = time.perf_counter() - hybrid_started

    if captured_metrics:
        record["hybrid_metrics"] = captured_metrics[-1]
        record["hybrid_cache_hits"] = [
            {
                "stage": call.get("stage"),
                "cache_hit": bool(call.get("cache_hit", False)),
            }
            for call in captured_metrics[-1].get("llm_calls", [])
        ]

    baseline_started = time.perf_counter()
    try:
        baseline = answer_baseline(
            str(case["question"]),
            llm_client=llm_client,
        )
        record["baseline_answer"] = baseline["answer"]
        record["baseline_cache_hit"] = bool(baseline.get("cache_hit", False))
        record["baseline_metrics"] = {
            key: value
            for key, value in baseline.items()
            if key not in {"answer", "cache_hit"}
        }
    except Exception as exc:
        errors.append(f"baseline: {_safe_error(exc)}")
    finally:
        if timings is not None:
            timings["baseline"] = time.perf_counter() - baseline_started

    record["error"] = "; ".join(errors) if errors else None
    return record


def _print_completed_progress(
    *,
    index: int,
    total: int,
    case: dict[str, Any],
    record: dict[str, Any],
    timings: dict[str, float],
) -> None:
    hybrid_status = "error" if _stage_has_error(record, "hybrid") else "ok"
    if hybrid_status == "ok" and record.get("degraded"):
        hybrid_status = "degraded"
    baseline_status = "error" if _stage_has_error(record, "baseline") else "ok"
    cache_hits = sum(
        bool(item.get("cache_hit"))
        for item in record.get("hybrid_cache_hits", [])
    ) + int(bool(record.get("baseline_cache_hit")))
    print(
        f"[{index}/{total}] {case['id']} {case['category']} "
        f"hybrid={hybrid_status} {timings.get('hybrid', 0.0):.2f}s "
        f"baseline={baseline_status} {timings.get('baseline', 0.0):.2f}s "
        f"cache_hits={cache_hits}",
        file=sys.stderr,
        flush=True,
    )


def _stage_has_error(record: dict[str, Any], stage: str) -> bool:
    error = record.get("error")
    if not error:
        return False
    parts = [part.strip() for part in str(error).split(";")]
    return any(
        part.startswith(f"{stage}:") or part.startswith("runner:")
        for part in parts
    )


def _empty_record(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case.get("id"),
        "category": case.get("category"),
        "question": case.get("question"),
        "customer_id": case.get("customer_id"),
        "resolved_checks": [],
        "hybrid_answer": None,
        "degraded": None,
        "route": None,
        "tools_used": [],
        "evidence_sources": {"sql": [], "chunks": []},
        "hybrid_blocked": None,
        "hybrid_grounded": None,
        "hybrid_request_id": None,
        "hybrid_metrics": {},
        "hybrid_cache_hits": [],
        "baseline_answer": None,
        "baseline_metrics": {},
        "baseline_cache_hit": None,
        "error": None,
    }


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    ids: set[str] | None,
    category: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        case
        for case in cases
        if (ids is None or str(case["id"]) in ids)
        and (category is None or case.get("category") == category)
    ]
    return selected if limit is None else selected[:limit]


def _read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc
            case_id = record.get("case_id", record.get("id"))
            if case_id is not None:
                completed.add(str(case_id))
    return completed


def _safe_error(error: Exception) -> str:
    text = str(error) or error.__class__.__name__
    if settings.GROQ_API_KEY:
        text = text.replace(settings.GROQ_API_KEY, "[REDACTED]")
    return text


def _redact_api_key(value: Any) -> Any:
    """Remove the configured API key from every string written to disk."""
    api_key = settings.GROQ_API_KEY
    if not api_key:
        return value
    if isinstance(value, str):
        return value.replace(api_key, "[REDACTED]")
    if isinstance(value, list):
        return [_redact_api_key(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_api_key(item) for key, item in value.items()}
    return value


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save raw Kartly hybrid and baseline evaluation results."
    )
    parser.add_argument("--limit", type=_non_negative_int)
    parser.add_argument("--ids", help="Comma-separated case IDs")
    parser.add_argument("--category")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--sleep", type=_non_negative_float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_ids = None
    if args.ids is not None:
        selected_ids = {item.strip() for item in args.ids.split(",") if item.strip()}
    try:
        output_path = run_evaluation(
            limit=args.limit,
            ids=selected_ids,
            category=args.category,
            resume=args.resume,
            sleep_seconds=args.sleep,
        )
    except EvaluationInterrupted as exc:
        resume_command = _resume_command(args, exc.output_path)
        print(f"Raw results: {exc.output_path}", file=sys.stderr, flush=True)
        print(f"Resume: {resume_command}", file=sys.stderr, flush=True)
        return 130
    print(output_path)
    return 0


def _resume_command(args: argparse.Namespace, output_path: Path) -> str:
    command = [
        sys.executable,
        "-m",
        "eval.run_eval",
        "--resume",
        str(output_path),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.ids is not None:
        command.extend(["--ids", args.ids])
    if args.category is not None:
        command.extend(["--category", args.category])
    command.extend(["--sleep", str(args.sleep)])
    return subprocess.list2cmdline(command)


if __name__ == "__main__":
    raise SystemExit(main())
