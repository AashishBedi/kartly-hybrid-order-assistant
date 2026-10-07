import csv
import json
from pathlib import Path
from typing import Any

import pytest

from eval import judge
from eval.judge import (
    _claims_from_rows,
    build_parser,
    extract_claims,
    judge_file,
    normalize_text,
)


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
        "baseline_sources": {"chunks": []},
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


def test_adjusted_summary_excludes_ids_for_both_systems_without_changing_full(
    synthetic_raw: Path,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    full_path, _ = judge_file(
        synthetic_raw,
        results_dir=results_dir,
        timestamp="20261004T120000_000000Z",
    )
    adjusted_path, _ = judge_file(
        synthetic_raw,
        results_dir=results_dir,
        timestamp="20261004T120001_000000Z",
        exclude_ids=["S1", "S3"],
        exclude_reason="Known ambiguous prompts",
    )

    full = json.loads(full_path.read_text(encoding="utf-8"))
    adjusted = json.loads(adjusted_path.read_text(encoding="utf-8"))

    assert adjusted["summary"] == full["summary"]
    adjusted_summary = adjusted["adjusted_summary"]
    assert adjusted_summary["excluded_case_ids"] == ["S1", "S3"]
    assert adjusted_summary["exclude_reason"] == "Known ambiguous prompts"
    assert adjusted_summary["overall"]["hybrid"]["passes"] == "1/2"
    assert adjusted_summary["overall"]["hybrid"][
        "unsupported_claim_rate"
    ] == "0/2"
    assert adjusted_summary["overall"]["baseline"]["passes"] == "0/2"
    assert adjusted_summary["overall"]["baseline"][
        "unsupported_claim_rate"
    ] == "1/2"


def test_exclude_cli_accepts_comma_separated_case_ids() -> None:
    args = build_parser().parse_args(
        ["raw.jsonl", "--exclude", "U05,D03", "--exclude-reason", "Review"]
    )

    assert args.exclude == "U05,D03"
    assert args.exclude_reason == "Review"


def test_cli_prints_full_and_adjusted_summary_tables(
    synthetic_raw: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    judged_path, review_path = judge_file(
        synthetic_raw,
        results_dir=tmp_path / "results",
        timestamp="20261004T120002_000000Z",
        exclude_ids=["S1"],
        exclude_reason="Review",
    )

    def fake_judge_file(
        raw_path: Path,
        **kwargs: Any,
    ) -> tuple[Path, Path]:
        assert raw_path == synthetic_raw
        assert kwargs["exclude_ids"] == ["S1"]
        assert kwargs["exclude_reason"] == "Review"
        return judged_path, review_path

    monkeypatch.setattr(judge, "judge_file", fake_judge_file)

    assert judge.main(
        [
            str(synthetic_raw),
            "--exclude",
            "S1",
            "--exclude-reason",
            "Review",
        ]
    ) == 0

    output = capsys.readouterr().out
    assert "Full summary\n" in output
    assert "Adjusted summary\n" in output
    assert "category" in output
    assert "hybrid passes" in output
    assert "baseline unsupported" in output
    assert "overall" in output
    assert "data" in output


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
    assert normalize_text(answer) == answer


def test_unicode_date_matches_ascii_db_date() -> None:
    claims = extract_claims("Delivered on 2026\u201110\u201102.")
    date_claim = next(claim for claim in claims if claim.kind == "date")

    assert date_claim.key in _claims_from_rows(
        [{"delivered_date": "2026-10-02"}]
    )


def test_narrow_no_break_space_order_id_tokenizes_normally() -> None:
    claims = extract_claims("Order\u202f#131 was delivered.")

    assert (claims[0].kind, claims[0].value, claims[0].token) == (
        "order_id",
        "131",
        "Order #131",
    )


def test_normalize_text_canonicalizes_unicode_punctuation_and_spacing() -> None:
    text = (
        "\u201cA\u201d\u00a0\u202f\u2009B\u200b"
        "\u2010\u2011\u2012\u2013\u2014\u2015\u2212C\u2018D\u2019"
    )

    assert normalize_text(text) == '"A" B-------C\'D\''


def test_lifecycle_dates_support_matching_status_words() -> None:
    allowed = _claims_from_rows(
        [
            {
                "status": "delivered",
                "shipped_date": "2026-10-01",
                "delivered_date": "2026-10-02",
            }
        ]
    )

    assert ("status", "shipped") in allowed
    assert ("status", "delivered") in allowed


@pytest.mark.parametrize(
    "answer",
    [
        "Neither item was shipped or delivered.",
        "No item was shipped.",
        "Order #12 wasn't delivered.",
        "Order #12 has not been delivered.",
    ],
)
def test_negated_status_words_are_not_claims(answer: str) -> None:
    assert not [claim for claim in extract_claims(answer) if claim.kind == "status"]


def test_generic_no_information_refusal_passes_but_concrete_answer_fails(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_refusal.jsonl"
    record = raw_case(
        "U1",
        "unanswerable",
        "Which countries does Kartly ship to?",
        [{"kind": "refusal", "phrases": []}],
        (
            "I'm sorry, but the provided information doesn't include details "
            "about the countries Kartly ships to internationally."
        ),
        "The capital of France is Paris.",
        1,
    )
    raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    judged_path, _ = judge_file(
        raw_path,
        results_dir=tmp_path / "results",
        timestamp="20261004T130000_000000Z",
    )
    case = json.loads(judged_path.read_text(encoding="utf-8"))["cases"][0]

    assert case["systems"]["hybrid"]["case_pass"] is True
    assert case["systems"]["hybrid"]["unsupported_claim"] is False
    assert case["systems"]["baseline"]["case_pass"] is False
    assert case["systems"]["baseline"]["unsupported_claim"] is True


def test_baseline_claims_use_vector_baseline_chunks_only(tmp_path: Path) -> None:
    record = raw_case(
        "P1",
        "policy",
        "How long is the special return window?",
        [],
        "The special return window is 45 days.",
        "The special return window is 45 days.",
        1,
    )
    record["baseline_sources"] = {"chunks": ["returns:v1:0"]}
    raw_path = tmp_path / "raw_vector_baseline.jsonl"
    raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    class FakeStore:
        def get_chunks(self, chunk_ids: list[str]) -> dict[str, str]:
            assert chunk_ids == ["returns:v1:0"]
            return {
                "returns:v1:0": "The special return window is 45 days."
            }

    judged_path, _ = judge_file(
        raw_path,
        results_dir=tmp_path / "results",
        store=FakeStore(),
        timestamp="20261004T140000_000000Z",
    )
    systems = json.loads(judged_path.read_text(encoding="utf-8"))["cases"][0][
        "systems"
    ]

    assert systems["hybrid"]["flagged_tokens"] == ["45 days"]
    assert systems["baseline"]["flagged_tokens"] == []
