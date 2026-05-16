"""
Two-level confidence scoring for mortgage AI decisions.

Level 1 — Intent confidence: did the AI understand what it was asked to do?
Level 2 — Transaction confidence: is the output valid given mortgage rules?

Both levels are rule-based; no LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_INTENT_KEYS = {
    "income_verification": {"gross_income", "monthly_income"},
    "closing_disclosure": {"loan_amount", "closing_costs", "cash_to_close"},
    "appraisal_comparison": {"appraised_value", "property_address"},
    "credit_decision": {"decision", "reason_codes"},
    "document_lineage": {"authoritative_version", "document_id"},
}

MORTGAGE_BOUNDS = {
    "loan_amount": (10_000, 5_000_000),
    "appraised_value": (10_000, 10_000_000),
    "gross_income": (0, 10_000_000),
    "monthly_income": (0, 1_000_000),
    "dti_ratio": (0.0, 1.0),
    "ltv_ratio": (0.0, 2.0),
    "credit_score": (300, 850),
    "closing_costs": (0, 200_000),
    "interest_rate": (0.001, 0.30),
}


@dataclass
class ConfidenceResult:
    """Holds both confidence levels and a combined score."""

    intent_confidence: float
    transaction_confidence: float
    combined: float
    intent_flags: list[str]
    transaction_flags: list[str]

    def __repr__(self) -> str:
        return (
            f"ConfidenceResult(combined={self.combined:.3f}, "
            f"intent={self.intent_confidence:.3f}, "
            f"transaction={self.transaction_confidence:.3f})"
        )


def score_intent_confidence(
    use_case: str,
    extracted_fields: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Check whether all expected fields for this use-case are present.

    Returns (score, list_of_missing_field_warnings).
    """
    required = REQUIRED_INTENT_KEYS.get(use_case, set())
    if not required:
        return 1.0, []

    present = {k.lower() for k in extracted_fields}
    missing = required - present
    flags = [f"Missing expected field: {f}" for f in sorted(missing)]
    score = 1.0 - (len(missing) / len(required))
    return round(score, 4), flags


def score_transaction_confidence(
    extracted_fields: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Validate numeric fields against known mortgage plausibility bounds.

    Returns (score, list_of_violation_warnings).
    """
    violations: list[str] = []
    checked = 0
    passed = 0

    for field, value in extracted_fields.items():
        key = field.lower()
        if key not in MORTGAGE_BOUNDS:
            continue
        checked += 1
        lo, hi = MORTGAGE_BOUNDS[key]
        try:
            val = float(value)
            if lo <= val <= hi:
                passed += 1
            else:
                violations.append(
                    f"{field}={val} out of plausible range [{lo}, {hi}]"
                )
        except (ValueError, TypeError):
            violations.append(f"{field} has non-numeric value: {value!r}")

    score = (passed / checked) if checked > 0 else 1.0
    return round(score, 4), violations


def compute_confidence(
    use_case: str,
    extracted_fields: dict[str, Any],
    intent_weight: float = 0.5,
    transaction_weight: float = 0.5,
) -> ConfidenceResult:
    """
    Combine intent and transaction confidence into a single result.

    Args:
        use_case: One of the five mortgage use-case identifiers.
        extracted_fields: Key-value fields extracted by the AI system.
        intent_weight: Weight for intent confidence (default 0.5).
        transaction_weight: Weight for transaction confidence (default 0.5).

    Returns:
        ConfidenceResult with both levels and a combined score.
    """
    intent_score, intent_flags = score_intent_confidence(use_case, extracted_fields)
    txn_score, txn_flags = score_transaction_confidence(extracted_fields)
    combined = round(intent_weight * intent_score + transaction_weight * txn_score, 4)
    return ConfidenceResult(
        intent_confidence=intent_score,
        transaction_confidence=txn_score,
        combined=combined,
        intent_flags=intent_flags,
        transaction_flags=txn_flags,
    )
