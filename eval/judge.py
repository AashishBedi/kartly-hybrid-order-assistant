"""Deterministically judge raw Kartly evaluation results.

An "unsupported claim" is an exact answer span asserting a concrete order ID,
affirmative order status, date, monetary amount, day count, percentage, or
email address that cannot be matched to the case's customer-scoped DB rows,
cited policy text, resolved checks, or question. ``shipped`` and ``delivered``
are supported when the order has the corresponding lifecycle date, and
negated status wording is not treated as a claim. For out-of-scope,
unanswerable, and adversarial cases, a non-refusal declarative answer is also
unsupported; when it contains none of the concrete token types above, its
first exact sentence is flagged. The detector intentionally ignores other
prose, so a missing flag is not proof that every statement is grounded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from app.data.access import run_tool
from app.db.connection import get_readonly_connection
from app.pipeline.data_evidence import compute_order_facts
from app.rag.store import PolicyStore
from eval.resolve import check_answer


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SYSTEMS = ("hybrid", "baseline")
REFUSAL_CATEGORIES = {"out_of_scope", "unanswerable", "adversarial"}
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u00ad"
_ORDER_ID_RE = re.compile(
    r"(?i)(?<!\w)(?:order\s*(?:number\s*)?#?\s*|#)(\d+)\b"
)
_STATUS_RE = re.compile(
    r"(?i)\b(?:is|was|were|has\s+been|status(?:\s+is|:)?|shows?|has\s+status)"
    r"\s+(placed|shipped|delivered|cancelled|returned)\b"
)
_DATE_RE = re.compile(
    rf"(?<!\d)(\d{{4}}[-{_DASHES}]\d{{1,2}}[-{_DASHES}]\d{{1,2}})(?!\d)"
)
_CURRENCY_RE = re.compile(
    r"(?i)(?<!\w)(?:USD\s*|[$€£₹]\s*)"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"|(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*"
    r"(?:USD|dollars?|rupees?|euros?|pounds?)\b"
)
_CONTEXT_AMOUNT_RE = re.compile(
    r"(?i)\b(?:total|charged|cost|price|amount)\b[^.\n]{0,80}?"
    r"(?:\b(?:is|was|came\s+to|of)\b|:)\s*(?:USD\s*|[$€£₹]\s*)?"
    r"(\d[\d,]*(?:\.\d+)?)"
)
_DAY_RE = re.compile(
    rf"(?i)(?<!\w)(\d+(?:\.\d+)?)\s*[-{_DASHES}]?\s*"
    r"(?:business\s+)?days?\b"
)
_PERCENT_RE = re.compile(r"(?i)(?<!\w)(\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_DECLARATIVE_RE = re.compile(
    r"(?i)\b(?:is|are|was|were|accepts?|accepted|offers?|ships?|costs?|takes?|"
    r"allows?|allowed|covers?|covered|will)\b"
)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?", re.MULTILINE)
_TEXT_TRANSLATION = str.maketrans(
    {
        **{
            ord(character): "-"
            for character in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
        },
        **{ord(character): "'" for character in "\u2018\u2019\u201a\u201b"},
        **{ord(character): '"' for character in "\u201c\u201d\u201e\u201f"},
    }
)


def normalize_text(text: str) -> str:
    """Return a comparison-only canonical form for Unicode text."""
    normalized = unicodedata.normalize("NFKC", text).translate(_TEXT_TRANSLATION)
    normalized = "".join(
        ""
        if unicodedata.category(character) == "Cf"
        or 0xFE00 <= ord(character) <= 0xFE0F
        or 0xE0100 <= ord(character) <= 0xE01EF
        else " "
        if character != " " and character.isspace()
        else character
        for character in normalized
    )
    return re.sub(r" +", " ", normalized)


@dataclass(frozen=True)
class Claim:
    kind: str
    value: str
    token: str

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.value


class EvidenceContext:
    """Lazily reconstruct evidence identified by a raw result record."""

    def __init__(
        self,
        connection: Any | None = None,
        store: PolicyStore | None = None,
        evaluation_date: date | None = None,
    ) -> None:
        self.connection = connection
        self.store = store
        self.evaluation_date = evaluation_date or date.today()
        self._owns_connection = False

    def close(self) -> None:
        if self._owns_connection and self.connection is not None:
            self.connection.close()

    def collect(self, record: dict[str, Any]) -> tuple[set[tuple[str, str]], bool]:
        allowed = _claims_from_resolved_checks(record.get("resolved_checks", []))
        allowed.update(
            claim.key
            for claim in extract_claims(str(record.get("question", "")))
            if claim.kind == "order_id"
        )
        complete = True
        sources = record.get("evidence_sources") or {}

        for source in sources.get("sql", []):
            try:
                rows = self._load_sql_rows(record, source)
            except Exception:
                complete = False
                continue
            allowed.update(_claims_from_rows(rows))
            if source.get("tool") == "get_order" and rows:
                allowed.update(
                    _claims_from_computed_facts(
                        compute_order_facts(rows, self.evaluation_date)
                    )
                )

        chunk_ids = [str(value) for value in sources.get("chunks", [])]
        if chunk_ids:
            try:
                texts = self._load_chunks(chunk_ids)
            except Exception:
                complete = False
            else:
                if set(texts) != set(chunk_ids):
                    complete = False
                for text in texts.values():
                    allowed.update(claim.key for claim in extract_claims(text))

        return allowed, complete

    def _load_sql_rows(
        self,
        record: dict[str, Any],
        source: dict[str, Any],
    ) -> list[dict[str, Any]]:
        customer_id = int(record["customer_id"])
        params = source.get("params", [])
        tool = source.get("tool")
        if not params or int(params[0 if tool != "get_order" else 1]) != customer_id:
            raise ValueError("SQL source customer does not match the case")

        if tool == "get_order" and len(params) == 2:
            args = {"order_id": int(params[0])}
        elif tool == "list_orders" and len(params) == 2:
            args = {"limit": int(params[1])}
        elif tool == "list_orders" and len(params) == 3:
            args = {"status": str(params[1]), "limit": int(params[2])}
        elif tool == "count_orders_by_status" and len(params) == 1:
            args = {}
        else:
            raise ValueError("Unknown SQL evidence shape")

        if self.connection is None:
            self.connection = get_readonly_connection()
            self._owns_connection = True
        return run_tool(tool, args, customer_id, conn=self.connection).rows

    def _load_chunks(self, chunk_ids: list[str]) -> dict[str, str]:
        if self.store is None:
            self.store = PolicyStore()
        return self.store.get_chunks(chunk_ids)


def extract_claims(text: str) -> list[Claim]:
    """Extract the deliberately small set of concrete claim spans."""
    text = normalize_text(text)
    claims: list[Claim] = []
    occupied: set[tuple[int, int, str]] = set()

    def add_matches(
        pattern: re.Pattern[str],
        kind: str,
        canonicalizer: Any,
        group: int = 0,
        token_group: int = 0,
    ) -> None:
        for match in pattern.finditer(text):
            token = match.group(token_group).strip()
            raw_value = match.group(group)
            if raw_value is None:
                raw_value = next(
                    value for value in match.groups() if value is not None
                )
            value = canonicalizer(raw_value)
            marker = (match.start(), match.end(), kind)
            if value is not None and marker not in occupied:
                occupied.add(marker)
                claims.append(Claim(kind=kind, value=value, token=token))

    add_matches(_ORDER_ID_RE, "order_id", lambda value: str(int(value)), group=1)
    for match in _STATUS_RE.finditer(text):
        if _status_is_negated(text, match):
            continue
        token = match.group(1).strip()
        claims.append(Claim(kind="status", value=token.casefold(), token=token))
    add_matches(_DATE_RE, "date", _canonical_date, group=1)
    add_matches(_CURRENCY_RE, "amount", _canonical_number)
    add_matches(
        _CONTEXT_AMOUNT_RE,
        "amount",
        _canonical_number,
        group=1,
        token_group=1,
    )
    add_matches(_DAY_RE, "day_count", _canonical_number, group=1)
    add_matches(_PERCENT_RE, "percentage", _canonical_number, group=1)
    add_matches(_EMAIL_RE, "email", lambda value: value.casefold())

    claims.sort(key=lambda claim: text.find(claim.token))
    return _deduplicate_claims(claims)


def judge_file(
    raw_path: Path,
    *,
    results_dir: Path = RESULTS_DIR,
    connection: Any | None = None,
    store: PolicyStore | None = None,
    timestamp: str | None = None,
    exclude_ids: Iterable[str] | None = None,
    exclude_reason: str | None = None,
) -> tuple[Path, Path]:
    """Judge one raw JSONL file and write JSON plus a manual-review CSV."""
    records = _read_jsonl(raw_path)
    excluded_case_ids = _normalize_case_ids(exclude_ids or [])
    evidence = EvidenceContext(
        connection=connection,
        store=store,
        evaluation_date=_evaluation_date_from_path(raw_path),
    )
    verdicts: list[dict[str, Any]] = []
    try:
        for record in records:
            allowed_claims, evidence_complete = evidence.collect(record)
            systems = {
                system: _judge_system(
                    record,
                    system,
                    allowed_claims=allowed_claims,
                    evidence_complete=evidence_complete,
                )
                for system in SYSTEMS
            }
            verdicts.append(
                {
                    "case_id": record.get("case_id", record.get("id")),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "systems": systems,
                }
            )
    finally:
        evidence.close()

    generated_at = timestamp or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    judged_path = results_dir / f"judged_{generated_at}.json"
    review_path = results_dir / f"review_{generated_at}.csv"
    payload = {
        "source_raw_file": str(raw_path),
        "generated_at_utc": generated_at,
        "cases": verdicts,
        "summary": _build_summary(records, verdicts),
    }
    if excluded_case_ids:
        excluded = set(excluded_case_ids)
        adjusted_pairs = [
            (record, verdict)
            for record, verdict in zip(records, verdicts)
            if str(record.get("case_id", record.get("id"))) not in excluded
        ]
        adjusted_records = [record for record, _ in adjusted_pairs]
        adjusted_verdicts = [verdict for _, verdict in adjusted_pairs]
        payload["adjusted_summary"] = {
            "excluded_case_ids": excluded_case_ids,
            "exclude_reason": exclude_reason,
            **_build_summary(adjusted_records, adjusted_verdicts),
        }
    judged_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_review_csv(review_path, records, verdicts)
    return judged_path, review_path


def _judge_system(
    record: dict[str, Any],
    system: str,
    *,
    allowed_claims: set[tuple[str, str]],
    evidence_complete: bool,
) -> dict[str, Any]:
    answer = str(record.get(f"{system}_answer") or "")
    comparison_answer = normalize_text(answer)
    error = _system_error(record.get("error"), system)
    check_results = []
    for index, check in enumerate(record.get("resolved_checks", [])):
        comparison_check = {
            key: (
                [
                    normalize_text(value) if isinstance(value, str) else value
                    for value in item
                ]
                if isinstance(item, list)
                else normalize_text(item)
                if isinstance(item, str)
                else item
            )
            for key, item in check.items()
        }
        passed = bool(check_answer(comparison_answer, comparison_check))
        check_results.append(
            {
                "index": index,
                "kind": check["kind"],
                "passed": passed,
            }
        )
    case_pass = error is None and all(result["passed"] for result in check_results)
    flagged_tokens = [] if error else _unsupported_tokens(
        comparison_answer,
        category=str(record.get("category", "")),
        checks=record.get("resolved_checks", []),
        check_results=check_results,
        allowed_claims=allowed_claims,
        evidence_complete=evidence_complete,
    )
    return {
        "answer": answer,
        "checks": check_results,
        "case_pass": case_pass,
        "unsupported_claim": bool(flagged_tokens),
        "flagged_tokens": flagged_tokens,
        "error": error,
    }


def _unsupported_tokens(
    answer: str,
    *,
    category: str,
    checks: list[dict[str, Any]],
    check_results: list[dict[str, Any]],
    allowed_claims: set[tuple[str, str]],
    evidence_complete: bool,
) -> list[str]:
    answer = normalize_text(answer)
    claims = extract_claims(answer)
    flagged = [
        claim.token
        for claim in claims
        if evidence_complete and claim.key not in allowed_claims
    ]

    refusal_passed = any(
        check.get("kind") == "refusal" and result["passed"]
        for check, result in zip(checks, check_results)
    )
    if (
        category in REFUSAL_CATEGORIES
        and answer.strip()
        and not refusal_passed
        and _DECLARATIVE_RE.search(answer)
        and not flagged
    ):
        span = _first_declarative_sentence(answer)
        if span:
            flagged.append(span)
    return _deduplicate_strings(flagged)


def _claims_from_resolved_checks(
    checks: Iterable[dict[str, Any]],
) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    for check in checks:
        kind = check.get("kind")
        values = check.get("values", [])
        if "value" in check:
            values = [check["value"], *values]
        if kind == "order_status":
            allowed.update(
                ("status", normalize_text(str(value)).casefold())
                for value in values
            )
        elif kind == "order_total":
            allowed.update(
                ("amount", canonical)
                for value in values
                if (canonical := _canonical_number(str(value))) is not None
            )
        elif kind == "order_ids_listed":
            allowed.update(("order_id", str(int(value))) for value in values)
        elif kind in {"policy_number", "order_count"}:
            for value in values:
                canonical = _canonical_number(str(value))
                if canonical is not None:
                    allowed.add(("day_count", canonical))
                    allowed.add(("percentage", canonical))
    return allowed


def _claims_from_rows(rows: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            if key in {"id", "order_id"}:
                allowed.add(("order_id", str(int(value))))
            elif key == "status":
                allowed.add(("status", normalize_text(str(value)).casefold()))
            elif key.endswith("_date"):
                allowed.add(("date", _canonical_date(str(value))))
                lifecycle_status = {
                    "shipped_date": "shipped",
                    "delivered_date": "delivered",
                }.get(key)
                if lifecycle_status:
                    allowed.add(("status", lifecycle_status))
            elif key in {"total", "unit_price"}:
                canonical = _canonical_number(str(value))
                if canonical is not None:
                    allowed.add(("amount", canonical))
            elif key == "email":
                allowed.add(("email", normalize_text(str(value)).casefold()))
    return allowed


def _status_is_negated(text: str, match: re.Match[str]) -> bool:
    clause_start = max(
        text.rfind(separator, 0, match.start())
        for separator in (".", "!", "?", ";", ",", "\n")
    )
    prefix = text[clause_start + 1 : match.start()]
    return re.search(r"(?i)\b(?:neither|no|none)\b", prefix) is not None


def _claims_from_computed_facts(facts: dict[str, Any]) -> set[tuple[str, str]]:
    allowed: set[tuple[str, str]] = set()
    for key in ("delivered_date", "return_window_end"):
        if facts.get(key):
            allowed.add(("date", _canonical_date(str(facts[key]))))
    if facts.get("days_since_delivery") is not None:
        canonical = _canonical_number(str(facts["days_since_delivery"]))
        if canonical is not None:
            allowed.add(("day_count", canonical))
    for warranty in facts.get("warranty_per_item", []):
        if warranty.get("warranty_end_date"):
            allowed.add(
                ("date", _canonical_date(str(warranty["warranty_end_date"])))
            )
    return allowed


def _build_summary(
    records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> dict[str, Any]:
    categories = sorted({str(record.get("category", "")) for record in records})
    return {
        "overall": {
            system: _summarize(records, verdicts, system)
            for system in SYSTEMS
        },
        "categories": {
            category: {
                system: _summarize(records, verdicts, system, category=category)
                for system in SYSTEMS
            }
            for category in categories
        },
    }


def _summarize(
    records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
    system: str,
    *,
    category: str | None = None,
) -> dict[str, Any]:
    pairs = [
        (record, verdict)
        for record, verdict in zip(records, verdicts)
        if category is None or str(record.get("category", "")) == category
    ]
    total = len(pairs)
    system_verdicts = [verdict["systems"][system] for _, verdict in pairs]
    passes = sum(item["case_pass"] for item in system_verdicts)
    unsupported = sum(item["unsupported_claim"] for item in system_verdicts)
    errors = sum(item["error"] is not None for item in system_verdicts)
    degraded = (
        sum(bool(record.get("degraded")) for record, _ in pairs)
        if system == "hybrid"
        else 0
    )
    latencies = [
        latency
        for record, _ in pairs
        if (latency := _latency(record, system)) is not None
    ]
    costs = [_cost(record, system) for record, _ in pairs]
    cache_flags = [
        flag
        for record, _ in pairs
        for flag in _cache_flags(record, system)
    ]
    return {
        "passes": f"{passes}/{total}",
        "unsupported_claim_rate": f"{unsupported}/{total}",
        "errors": f"{errors}/{total}",
        "degraded_count": degraded,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "mean_cost_per_case": sum(costs) / total if total else 0.0,
        "cache_hits": f"{sum(cache_flags)}/{len(cache_flags)}",
    }


def _latency(record: dict[str, Any], system: str) -> float | None:
    metrics = record.get(f"{system}_metrics") or {}
    key = "total_latency_ms" if system == "hybrid" else "latency_ms"
    value = metrics.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _cost(record: dict[str, Any], system: str) -> float:
    metrics = record.get(f"{system}_metrics") or {}
    key = "total_cost_usd" if system == "hybrid" else "cost_usd"
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _cache_flags(record: dict[str, Any], system: str) -> list[bool]:
    if system == "hybrid":
        explicit = record.get("hybrid_cache_hits")
        if isinstance(explicit, list):
            return [bool(item.get("cache_hit")) for item in explicit]
        calls = (record.get("hybrid_metrics") or {}).get("llm_calls", [])
        return [bool(call.get("cache_hit")) for call in calls]
    value = record.get("baseline_cache_hit")
    return [] if value is None else [bool(value)]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _system_error(error: Any, system: str) -> str | None:
    if not error:
        return None
    parts = [part.strip() for part in str(error).split(";") if part.strip()]
    relevant = [
        part
        for part in parts
        if part.startswith(f"{system}:")
        or part.startswith("resolve:")
        or part.startswith("runner:")
        or ":" not in part
    ]
    return "; ".join(relevant) if relevant else None


def _write_review_csv(
    path: Path,
    records: list[dict[str, Any]],
    verdicts: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "id",
        "category",
        "system",
        "auto_verdict",
        "failed_checks",
        "flagged_tokens",
        "answer_excerpt",
        "manual_verdict",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record, verdict in zip(records, verdicts):
            for system in SYSTEMS:
                result = verdict["systems"][system]
                failed = [
                    f"{item['index']}:{item['kind']}"
                    for item in result["checks"]
                    if not item["passed"]
                ]
                auto_verdict = (
                    "error"
                    if result["error"] is not None
                    else "pass" if result["case_pass"] else "fail"
                )
                writer.writerow(
                    {
                        "id": verdict["case_id"],
                        "category": verdict["category"],
                        "system": system,
                        "auto_verdict": auto_verdict,
                        "failed_checks": json.dumps(failed),
                        "flagged_tokens": json.dumps(
                            result["flagged_tokens"], ensure_ascii=False
                        ),
                        "answer_excerpt": result["answer"][:300],
                        "manual_verdict": "",
                        "notes": "",
                    }
                )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
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
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def _evaluation_date_from_path(path: Path) -> date:
    match = re.match(r"raw_(\d{8})T", path.name)
    if match is None:
        return date.today()
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def _canonical_date(value: str) -> str:
    return re.sub(rf"[-{_DASHES}]", "-", normalize_text(value))


def _canonical_number(value: str) -> str | None:
    cleaned = re.sub(
        r"(?i)(?:USD|dollars?|rupees?|euros?|pounds?|[$€£₹])",
        "",
        normalize_text(value),
    )
    try:
        number = Decimal(cleaned.replace(",", "").strip())
    except InvalidOperation:
        return None
    return format(number.normalize(), "f")


def _first_declarative_sentence(answer: str) -> str:
    answer = normalize_text(answer)
    for match in _SENTENCE_RE.finditer(answer):
        sentence = match.group(0).strip()
        if sentence and _DECLARATIVE_RE.search(sentence):
            return sentence
    return ""


def _deduplicate_claims(claims: list[Claim]) -> list[Claim]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for claim in claims:
        marker = (claim.kind, claim.value, claim.token)
        if marker not in seen:
            seen.add(marker)
            result.append(claim)
    return result


def _deduplicate_strings(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _normalize_case_ids(values: Iterable[str]) -> list[str]:
    return list(
        dict.fromkeys(
            case_id
            for value in values
            for item in str(value).split(",")
            if (case_id := item.strip())
        )
    )


def _format_summary_table(title: str, summary: dict[str, Any]) -> str:
    headers = (
        "category",
        "hybrid passes",
        "hybrid unsupported",
        "baseline passes",
        "baseline unsupported",
    )
    rows = [
        (
            "overall",
            summary["overall"]["hybrid"]["passes"],
            summary["overall"]["hybrid"]["unsupported_claim_rate"],
            summary["overall"]["baseline"]["passes"],
            summary["overall"]["baseline"]["unsupported_claim_rate"],
        ),
        *(
            (
                category,
                systems["hybrid"]["passes"],
                systems["hybrid"]["unsupported_claim_rate"],
                systems["baseline"]["passes"],
                systems["baseline"]["unsupported_claim_rate"],
            )
            for category, systems in summary["categories"].items()
        ),
    ]
    widths = [
        max(len(str(value)) for value in (header, *(row[index] for row in rows)))
        for index, header in enumerate(headers)
    ]

    def format_row(row: Sequence[str]) -> str:
        return " | ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ).rstrip()

    divider = "-+-".join("-" * width for width in widths)
    return "\n".join(
        (title, format_row(headers), divider, *(format_row(row) for row in rows))
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge raw Kartly evaluation results without model calls."
    )
    parser.add_argument("raw_file", type=Path)
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated case IDs to omit from an adjusted summary.",
    )
    parser.add_argument(
        "--exclude-reason",
        help="Reason recorded alongside the adjusted summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    excluded_case_ids = _normalize_case_ids([args.exclude])
    judged_path, review_path = judge_file(
        args.raw_file,
        exclude_ids=excluded_case_ids,
        exclude_reason=args.exclude_reason,
    )
    print(judged_path)
    print(review_path)
    payload = json.loads(judged_path.read_text(encoding="utf-8"))
    print(_format_summary_table("Full summary", payload["summary"]))
    if "adjusted_summary" in payload:
        print(
            _format_summary_table(
                "Adjusted summary", payload["adjusted_summary"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
