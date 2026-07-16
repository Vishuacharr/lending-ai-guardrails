"""
Batch validation runner for the evaluation harness.

Runs every scenario in mortgage/batch_scenarios.py through the
EvaluationHarness, measures wall-clock time per evaluation, and tags each
result with its ground-truth issue label so downstream KPI computation can
measure real false positive / false negative rates instead of guessing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from core.evaluator import EvaluationHarness
from mortgage.batch_scenarios import SCENARIOS, Scenario


class BatchRecord:
    """One scenario's evaluation result plus timing and ground-truth label."""

    def __init__(self, scenario: Scenario, result_dict: dict[str, Any], elapsed_seconds: float) -> None:
        self.scenario_id = scenario["id"]
        self.loan_id = scenario["loan_id"]
        self.use_case = scenario["use_case"]
        self.issue_type = scenario["issue_type"]
        self.has_issue = scenario["has_issue"]
        self.elapsed_seconds = elapsed_seconds
        self.result = result_dict

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "loan_id": self.loan_id,
            "use_case": self.use_case,
            "issue_type": self.issue_type,
            "has_issue": self.has_issue,
            "elapsed_seconds": self.elapsed_seconds,
            **self.result,
        }


def run_batch(harness: EvaluationHarness, scenarios: list[Scenario] | None = None) -> list[BatchRecord]:
    """
    Run every scenario through the harness and return timed, labeled records.

    Args:
        harness: An EvaluationHarness instance (audit log wired up by caller).
        scenarios: Optional override list; defaults to the full 24-scenario set.

    Returns:
        List of BatchRecord, one per scenario, in input order.
    """
    scenarios = scenarios if scenarios is not None else SCENARIOS
    records: list[BatchRecord] = []

    for scenario in scenarios:
        start = time.perf_counter()
        result = harness.evaluate(
            use_case=scenario["use_case"],
            loan_application_id=scenario["loan_id"],
            ai_output=scenario["ai_output"],
            source_context=scenario["source_context"],
            query=scenario["query"],
            extracted_fields=scenario["extracted_fields"],
            ground_truth_fields=scenario["ground_truth_fields"],
            metadata={**scenario["metadata"], "scenario_id": scenario["id"], "issue_type": scenario["issue_type"]},
        )
        elapsed = time.perf_counter() - start
        records.append(BatchRecord(scenario, result.to_dict(), elapsed))

    return records


def run_batch_headless(audit_log_path: str | Path | None = None) -> tuple[list[BatchRecord], EvaluationHarness]:
    """Convenience entry point: build a fresh harness and run the full batch."""
    harness = EvaluationHarness(audit_log_path=audit_log_path)
    records = run_batch(harness)
    return records, harness
