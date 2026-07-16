"""
Tests for the batch validation runner and KPI computation.

Verifies:
  - All 24 scenarios process without error
  - Scenario labels (has_issue / issue_type) are self-consistent
  - KPI computation produces well-formed, bounded output
  - Routing accuracy metrics (false positive/negative) are computable
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest

from core.batch_runner import run_batch, BatchRecord
from core.evaluator import EvaluationHarness
from core.kpi import compute_kpis
from mortgage.batch_scenarios import SCENARIOS, scenario_counts


@pytest.fixture
def harness(tmp_path: Path) -> EvaluationHarness:
    return EvaluationHarness(audit_log_path=tmp_path / "batch_audit.jsonl")


class TestScenarios:
    def test_at_least_20_scenarios(self) -> None:
        assert len(SCENARIOS) >= 20

    def test_unique_scenario_ids(self) -> None:
        ids = [s["id"] for s in SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_unique_loan_ids(self) -> None:
        loan_ids = [s["loan_id"] for s in SCENARIOS]
        assert len(loan_ids) == len(set(loan_ids))

    def test_all_five_use_cases_represented(self) -> None:
        use_cases = {s["use_case"] for s in SCENARIOS}
        assert use_cases == {
            "income_verification",
            "closing_disclosure",
            "appraisal_comparison",
            "credit_decision",
            "document_lineage",
        }

    def test_issue_categories_present(self) -> None:
        counts = scenario_counts()
        for category in (
            "clean",
            "income_mismatch",
            "document_version_conflict",
            "cd_imbalance",
            "compliance_violation",
        ):
            assert counts.get(category, 0) > 0, f"missing scenarios for {category}"

    def test_has_issue_matches_issue_type(self) -> None:
        for s in SCENARIOS:
            expected = s["issue_type"] != "clean"
            assert s["has_issue"] == expected


class TestBatchRunner:
    def test_all_scenarios_process(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        assert len(records) == len(SCENARIOS)

    def test_records_are_valid(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        for r in records:
            assert isinstance(r, BatchRecord)
            assert r.result["routing_decision"] in ("AUTO_APPROVE", "HUMAN_REVIEW", "REJECT")
            assert 0.0 <= r.result["combined_confidence"] <= 1.0
            assert r.elapsed_seconds >= 0.0

    def test_audit_log_has_all_records(self, harness: EvaluationHarness) -> None:
        run_batch(harness)
        assert len(harness.audit_logger.all_records()) == len(SCENARIOS)


class TestKPIs:
    def test_kpi_structure(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        kpis = compute_kpis(records)
        for key in (
            "routing",
            "processing_time",
            "error_rate",
            "compliance_score",
            "cost_savings",
            "routing_accuracy",
        ):
            assert key in kpis

    def test_routing_rates_sum_to_one(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        kpis = compute_kpis(records)
        routing = kpis["routing"]
        total = (
            routing["auto_approve_rate"]
            + routing["human_review_rate"]
            + routing["reject_rate"]
        )
        assert total == pytest.approx(1.0, abs=0.001)

    def test_false_positive_negative_rates_bounded(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        kpis = compute_kpis(records)
        acc = kpis["routing_accuracy"]
        assert 0.0 <= acc["false_positive_rate"] <= 1.0
        assert 0.0 <= acc["false_negative_rate"] <= 1.0

    def test_clean_scenarios_mostly_auto_approve(self, harness: EvaluationHarness) -> None:
        """Clean scenarios should have a low false positive rate."""
        records = run_batch(harness)
        kpis = compute_kpis(records)
        assert kpis["routing_accuracy"]["clean_pass_rate"] >= 0.5

    def test_flawed_scenarios_mostly_caught(self, harness: EvaluationHarness) -> None:
        """Scenarios with an injected issue should be mostly flagged, not auto-approved."""
        records = run_batch(harness)
        kpis = compute_kpis(records)
        assert kpis["routing_accuracy"]["catch_rate"] >= 0.5

    def test_cost_savings_non_negative(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        kpis = compute_kpis(records)
        assert kpis["cost_savings"]["savings_per_loan_usd"] > 0

    def test_processing_time_reduction_positive(self, harness: EvaluationHarness) -> None:
        records = run_batch(harness)
        kpis = compute_kpis(records)
        assert kpis["processing_time"]["reduction_pct"] > 0.9

    def test_empty_records_returns_empty_dict(self) -> None:
        assert compute_kpis([]) == {}
