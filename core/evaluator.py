"""
Main evaluation engine for the mortgage AI evaluation harness.

EvaluationHarness wires together metrics, confidence scoring, routing,
and audit logging into a single evaluate() call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.audit_logger import AuditLogger, AuditRecord
from core.confidence import compute_confidence
from core.metrics import (
    compute_composite_score,
    score_accuracy,
    score_faithfulness,
    score_relevance,
)
from core.router import RouteResult, route


class EvaluationResult:
    """Full result of one evaluation pass."""

    def __init__(
        self,
        faithfulness: float,
        relevance: float,
        accuracy: float,
        composite: float,
        confidence: Any,
        route_result: RouteResult,
        audit_record: AuditRecord,
    ) -> None:
        self.faithfulness = faithfulness
        self.relevance = relevance
        self.accuracy = accuracy
        self.composite = composite
        self.confidence = confidence
        self.route_result = route_result
        self.audit_record = audit_record

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"composite={self.composite:.3f}, "
            f"confidence={self.confidence.combined:.3f}, "
            f"decision={self.route_result.decision.value})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "accuracy": self.accuracy,
            "composite_score": self.composite,
            "intent_confidence": self.confidence.intent_confidence,
            "transaction_confidence": self.confidence.transaction_confidence,
            "combined_confidence": self.confidence.combined,
            "routing_decision": self.route_result.decision.value,
            "routing_reason": self.route_result.reason,
            "flags": self.confidence.intent_flags + self.confidence.transaction_flags,
        }


class EvaluationHarness:
    """
    Central orchestrator for mortgage AI evaluation.

    Usage:
        harness = EvaluationHarness(audit_log_path="audit/eval.jsonl")
        result = harness.evaluate(
            use_case="income_verification",
            loan_application_id="LA-001",
            ai_output="Gross income extracted: $120,000 annually.",
            source_context="W-2 Box 1: $120,000 wages...",
            query="Extract gross income from the W-2 document.",
            extracted_fields={"gross_income": 120000},
            ground_truth_fields={"gross_income": 120000},
        )
        print(result.route_result.decision)  # AUTO_APPROVE
    """

    def __init__(
        self,
        audit_log_path: str | Path | None = None,
        metric_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        high_threshold: float = 0.85,
        low_threshold: float = 0.60,
    ) -> None:
        """
        Args:
            audit_log_path: Where to write the JSONL audit trail.
            metric_weights: (faithfulness, relevance, accuracy) weights.
            high_threshold: Auto-approve cutoff.
            low_threshold: Reject cutoff.
        """
        self.audit_logger = AuditLogger(log_path=audit_log_path)
        self.metric_weights = metric_weights
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def evaluate(
        self,
        use_case: str,
        loan_application_id: str,
        ai_output: str,
        source_context: str,
        query: str,
        extracted_fields: dict[str, Any],
        ground_truth_fields: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """
        Run the full evaluation pipeline for one AI output.

        Args:
            use_case: Mortgage use-case identifier (e.g. "income_verification").
            loan_application_id: Unique loan ID for the audit trail.
            ai_output: The text output produced by the AI system.
            source_context: The source document(s) the AI drew from.
            query: The original task or query posed to the AI.
            extracted_fields: Key-value fields the AI extracted.
            ground_truth_fields: Expected correct field values.
            metadata: Optional extra context stored in the audit record.

        Returns:
            EvaluationResult with scores, confidence, routing, and audit record.
        """
        # Step 1: Compute metrics
        faith = score_faithfulness(ai_output, source_context)
        relev = score_relevance(ai_output, query)
        accur = score_accuracy(extracted_fields, ground_truth_fields)
        composite = compute_composite_score(faith, relev, accur, self.metric_weights)

        # Step 2: Confidence scoring
        confidence = compute_confidence(use_case, extracted_fields)

        # Step 3: Route the decision
        all_flags = confidence.intent_flags + confidence.transaction_flags
        route_result = route(
            confidence.combined,
            flags=all_flags,
            high_threshold=self.high_threshold,
            low_threshold=self.low_threshold,
        )

        # Step 4: Audit log
        record = AuditRecord(
            use_case=use_case,
            loan_application_id=loan_application_id,
            ai_output_summary=ai_output[:300],
            faithfulness_score=faith,
            relevance_score=relev,
            accuracy_score=accur,
            composite_score=composite,
            intent_confidence=confidence.intent_confidence,
            transaction_confidence=confidence.transaction_confidence,
            combined_confidence=confidence.combined,
            routing_decision=route_result.decision.value,
            routing_reason=route_result.reason,
            flags=all_flags,
            metadata=metadata or {},
        )
        self.audit_logger.log(record)

        return EvaluationResult(
            faithfulness=faith,
            relevance=relev,
            accuracy=accur,
            composite=composite,
            confidence=confidence,
            route_result=route_result,
            audit_record=record,
        )

    def summary(self) -> dict[str, Any]:
        """Return aggregate stats from the audit logger."""
        return self.audit_logger.summary_stats()

    def export_report(self, path: str | Path) -> Path:
        """Export the full audit log to a JSON file."""
        return self.audit_logger.export_json(path)
