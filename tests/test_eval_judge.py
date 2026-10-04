import csv
import json
from pathlib import Path
from typing import Any

import pytest

from eval.judge import extract_claims, judge_file


def metrics(
    latency_ms: int,
    cost_usd: float,
    *,
    hybrid: bool,
) -> dict[str, Any]:
    if hybrid:
        return {
            "total_latency_ms": latency_ms,
            "total_cost_usd": cost_usd,
            "llm_calls": [],
        }
    return {
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
    }


def raw_case(
    case_id: str,
    category: str,
    question: str,
    checks: list[dict[str, Any]],
    hybrid_answer: str,
    baseline_answer: str,
    index: int,
    *,
    degraded: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "category": category,
        "question": question,
        "customer_id": 1,
        "resolved_checks": checks,
        "hybrid_answer": hybrid_answer,
        "degraded": degraded,
        "route": category,
        "tools_used": [],
        "evidence_sources": {"sql": [], "chunks": []},
        "hybrid_metrics": metrics(index * 10, index / 10, hybrid=True),
        "baseline_answer": baseline_answer,
        "baseline_metrics": metrics(index * 10 + 1, index / 100, hybrid=False),
        "hybrid_cache_hits": [
            {"stage": "answer", "cache_hit": index % 2 == 1}
        ],
        "baseline_cache_hit": index == 2,
        "error": error,
    }


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    records = [
        raw_case(
            "S1",
            "data",
            "Where is order 1?",
            [{"kind": "order_status", "value": "delivered"}],
            "Order 1 is delivered.",
            "Order 1 is shipped.",
            1,
        ),
        raw_case(
            "S2",
            "policy",
            "How many days do returns take?",
            [{"kind": "policy_number", "value": "30"}],
            "Returns take 30 days.",
            "Returns take 60 days.",
            2,
            degraded=True,
        ),
        raw_case(
            "S3",
            "out_of_scope",
            "What is the answer?",
            [{"kind": "refusal", "phrases": ["can't help"]}],
            "I can't help with that.",
            "The answer is 42%.",
            3,
        ),
        raw_case(
            "S4",
            "data",
            "Give me a number.",
            [{"kind": "policy_number", "value": "30"}],
            "30 days.",
            "30 days.",
            4,
            error="hybrid: timed out; baseline: unavailable",
        ),
    ]
    path = tmp_path / "raw_synthetic.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_judge_records_pass_fail_unsupported_error_and_summary(
    synthetic_raw: Path,
    tmp_path: Path,
) -> None:
    judged_path, review_path = judge_file(
        synthetic_raw,
        results_dir=tmp_path / "results",
        timestamp="20261004T120000_000000Z",
    )

    judged = json.loads(judged_path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in judged["cases"]}

    assert cases["S1"]["systems"]["hybrid"]["case_pass"] is True
    assert cases["S1"]["systems"]["baseline"]["case_pass"] is False
    assert cases["S1"]["systems"]["baseline"]["flagged_tokens"] == [
        "shipped"
    ]
    assert cases["S2"]["systems"]["baseline"]["flagged_tokens"] == [
        "60 days"
    ]
    assert cases["S3"]["systems"]["baseline"]["flagged_tokens"] == [
        "42%"
    ]
    assert cases["S4"]["systems"]["hybrid"]["case_pass"] is False
    assert cases["S4"]["systems"]["hybrid"]["error"] == "hybrid: timed out"
    assert cases["S4"]["systems"]["baseline"]["error"] == (
        "baseline: unavailable"
    )

    hybrid = judged["summary"]["overall"]["hybrid"]
    baseline = judged["summary"]["overall"]["baseline"]
    assert hybrid["passes"] == "3/4"
    assert hybrid["unsupported_claim_rate"] == "0/4"
    assert hybrid["errors"] == "1/4"
    assert hybrid["degraded_count"] == 1
    assert hybrid["latency_ms"] == {"p50": 25.0, "p95": 38.5}
    assert hybrid["mean_cost_per_case"] == pytest.approx(0.25)
    assert hybrid["cache_hits"] == "2/4"
    assert baseline["passes"] == "0/4"
    assert baseline["unsupported_claim_rate"] == "3/4"
    assert baseline["errors"] == "1/4"
    assert baseline["cache_hits"] == "1/4"
    assert judged["summary"]["categories"]["data"]["hybrid"]["passes"] == (
        "1/2"
    )

    with review_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 8
    assert rows[0]["id"] == "S1"
    assert rows[0]["system"] == "hybrid"
    assert rows[0]["auto_verdict"] == "pass"
    assert rows[0]["manual_verdict"] == ""
    assert rows[0]["notes"] == ""
    error_row = next(
        row for row in rows if row["id"] == "S4" and row["system"] == "hybrid"
    )
    assert error_row["auto_verdict"] == "error"


def test_claim_extraction_returns_exact_tokens() -> None:
    answer = (
        "Order #17 was delivered on 2026-10-02. Its total charged for order 17 "
        "was 1,299.50, due in 30-day with a 5% fee."
    )

    claims = extract_claims(answer)

    assert [claim.token for claim in claims] == [
        "Order #17",
        "delivered",
        "2026-10-02",
        "order 17",
        "1,299.50",
        "30-day",
        "5%",
    ]
