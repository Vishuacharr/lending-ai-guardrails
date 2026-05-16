"""
Immutable audit logger for mortgage AI evaluation decisions.

Every decision is appended to an in-memory log and optionally persisted
to a JSONL file. The log is append-only — no record is ever modified or
deleted, making it suitable for CFPB audit requirements.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AuditRecord(BaseModel):
    """A single immutable audit record for one evaluation decision."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    use_case: str
    loan_application_id: str
    ai_output_summary: str
    faithfulness_score: float
    relevance_score: float
    accuracy_score: float
    composite_score: float
    intent_confidence: float
    transaction_confidence: float
    combined_confidence: float
    routing_decision: str
    routing_reason: str
    flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}  # Enforce immutability after creation


class AuditLogger:
    """
    Append-only audit logger.

    Records are held in memory and optionally flushed to a JSONL file.
    Each line in the JSONL file is a complete, self-contained AuditRecord.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        """
        Args:
            log_path: Path to the JSONL audit file. If None, logs are
                      kept in memory only (useful for testing).
        """
        self._records: list[AuditRecord] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: AuditRecord) -> AuditRecord:
        """
        Append a record to the audit trail.

        Args:
            record: The AuditRecord to persist.

        Returns:
            The same record (for chaining).
        """
        self._records.append(record)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
        return record

    def all_records(self) -> list[AuditRecord]:
        """Return all records in chronological order (copy)."""
        return list(self._records)

    def records_for_loan(self, loan_id: str) -> list[AuditRecord]:
        """Return all records for a specific loan application."""
        return [r for r in self._records if r.loan_application_id == loan_id]

    def summary_stats(self) -> dict[str, Any]:
        """Aggregate stats across all logged decisions."""
        if not self._records:
            return {}
        decisions = [r.routing_decision for r in self._records]
        return {
            "total_evaluations": len(self._records),
            "auto_approve_count": decisions.count("AUTO_APPROVE"),
            "human_review_count": decisions.count("HUMAN_REVIEW"),
            "reject_count": decisions.count("REJECT"),
            "auto_approve_rate": round(decisions.count("AUTO_APPROVE") / len(decisions), 4),
            "human_review_rate": round(decisions.count("HUMAN_REVIEW") / len(decisions), 4),
            "reject_rate": round(decisions.count("REJECT") / len(decisions), 4),
            "avg_faithfulness": round(
                sum(r.faithfulness_score for r in self._records) / len(self._records), 4
            ),
            "avg_relevance": round(
                sum(r.relevance_score for r in self._records) / len(self._records), 4
            ),
            "avg_accuracy": round(
                sum(r.accuracy_score for r in self._records) / len(self._records), 4
            ),
            "avg_confidence": round(
                sum(r.combined_confidence for r in self._records) / len(self._records), 4
            ),
        }

    def export_json(self, path: str | Path) -> Path:
        """Export all records to a single JSON file (pretty-printed)."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.summary_stats(),
            "records": [r.model_dump() for r in self._records],
        }
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return out
