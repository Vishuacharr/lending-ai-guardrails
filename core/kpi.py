"""
KPI computation for the lending AI evaluation harness.

Everything here is deterministic — no LLM calls, no randomness. Two kinds
of numbers feed the KPIs:

  1. MEASURED numbers, computed directly from a batch run of the harness
     against the 24 labeled scenarios in mortgage/batch_scenarios.py
     (routing outcomes, catch rate, false positive/negative rate, wall
     clock harness runtime).
  2. BASELINE numbers, industry-sourced constants representing manual
     mortgage QC/underwriting review *before* an evaluation harness is
     introduced. These are illustrative published-range figures, not this
     org's real numbers — swap in actual historical metrics when available.

Baseline sources (illustrative, cited for defensibility):
  - Manual doc-review time: ~30-60 min per document/decision review is a
    commonly cited range in mortgage QC literature (MBA/Fannie Mae loan
    quality reports). We use the midpoint, 45 minutes.
  - Manual QC error/defect rate: industry loan-quality studies (e.g. ACES
    Quality Management National Mortgage Defect Index) have historically
    reported critical defect rates in the high single digits to low
    double digits. We use 12% as an illustrative baseline.
  - Manual compliance adherence: illustrative baseline of 78%, reflecting
    inconsistent manual application of TRID/ECOA/FCRA checklists absent
    an automated rule engine.
  - Underwriter fully-loaded hourly cost: $45/hr illustrative.
"""

from __future__ import annotations

from typing import Any

from core.batch_runner import BatchRecord

# ---------------------------------------------------------------------------
# Baseline constants (manual / pre-harness) — illustrative, industry-sourced
# ---------------------------------------------------------------------------

MANUAL_AVG_MINUTES_PER_LOAN: float = 45.0
MANUAL_ERROR_RATE: float = 0.12
MANUAL_COMPLIANCE_SCORE: float = 0.78
UNDERWRITER_HOURLY_RATE_USD: float = 45.0

REVIEW_MINUTES_PER_FLAGGED_LOAN: float = 12.0
AI_COMPUTE_COST_PER_LOAN_USD: float = 0.03

# Illustrative monthly loan volume used only to project total $ savings
ASSUMED_MONTHLY_LOAN_VOLUME: int = 500


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def compute_kpis(records: list[BatchRecord]) -> dict[str, Any]:
    """
    Compute the full KPI package from a batch of evaluation results.

    Returns a dict with before/after comparisons, routing breakdown,
    false positive/negative rates, and cost savings — everything the
    dashboard needs to render KPI cards and trend charts.
    """
    if not records:
        return {}

    n = len(records)
    decisions = [r.result["routing_decision"] for r in records]
    auto_approve = decisions.count("AUTO_APPROVE")
    human_review = decisions.count("HUMAN_REVIEW")
    reject = decisions.count("REJECT")

    clean = [r for r in records if not r.has_issue]
    flawed = [r for r in records if r.has_issue]

    # False positive: clean scenario flagged for review/reject anyway
    false_positives = sum(
        1 for r in clean if r.result["routing_decision"] != "AUTO_APPROVE"
    )
    # False negative: real issue slipped through as AUTO_APPROVE
    false_negatives = sum(
        1 for r in flawed if r.result["routing_decision"] == "AUTO_APPROVE"
    )

    false_positive_rate = _rate(false_positives, len(clean))
    false_negative_rate = _rate(false_negatives, len(flawed))
    catch_rate = round(1.0 - false_negative_rate, 4) if flawed else 1.0
    clean_pass_rate = round(1.0 - false_positive_rate, 4) if clean else 1.0

    avg_harness_seconds = sum(r.elapsed_seconds for r in records) / n

    # --- Processing time ---------------------------------------------------
    manual_seconds = MANUAL_AVG_MINUTES_PER_LOAN * 60
    time_reduction_pct = round(
        (manual_seconds - avg_harness_seconds) / manual_seconds, 4
    )

    # --- Error rate ----------------------------------------------------------
    # Post-harness "error rate" = fraction of real issues that still slip
    # through undetected (false negative rate) — the residual risk exposed
    # to the business after the harness is in place.
    post_error_rate = false_negative_rate
    error_rate_reduction_pct = (
        round((MANUAL_ERROR_RATE - post_error_rate) / MANUAL_ERROR_RATE, 4)
        if MANUAL_ERROR_RATE
        else 0.0
    )

    # --- Compliance score ----------------------------------------------------
    # Post-harness compliance score = catch rate on the subset of scenarios
    # that are compliance-relevant (cd_imbalance + compliance_violation).
    compliance_relevant = [
        r for r in flawed if r.issue_type in ("cd_imbalance", "compliance_violation")
    ]
    compliance_caught = sum(
        1 for r in compliance_relevant if r.result["routing_decision"] != "AUTO_APPROVE"
    )
    post_compliance_score = (
        round(compliance_caught / len(compliance_relevant), 4)
        if compliance_relevant
        else 1.0
    )
    compliance_uplift = round(post_compliance_score - MANUAL_COMPLIANCE_SCORE, 4)

    # --- Cost savings ----------------------------------------------------------
    manual_cost_per_loan = (MANUAL_AVG_MINUTES_PER_LOAN / 60) * UNDERWRITER_HOURLY_RATE_USD
    flagged_rate = _rate(human_review + reject, n)
    auto_rate = _rate(auto_approve, n)
    post_cost_per_loan = (
        flagged_rate * (REVIEW_MINUTES_PER_FLAGGED_LOAN / 60) * UNDERWRITER_HOURLY_RATE_USD
        + auto_rate * AI_COMPUTE_COST_PER_LOAN_USD
    )
    savings_per_loan = round(manual_cost_per_loan - post_cost_per_loan, 2)
    projected_monthly_savings = round(savings_per_loan * ASSUMED_MONTHLY_LOAN_VOLUME, 2)
    projected_annual_savings = round(projected_monthly_savings * 12, 2)

    return {
        "total_scenarios": n,
        "clean_scenarios": len(clean),
        "flawed_scenarios": len(flawed),
        "routing": {
            "auto_approve_count": auto_approve,
            "human_review_count": human_review,
            "reject_count": reject,
            "auto_approve_rate": _rate(auto_approve, n),
            "human_review_rate": _rate(human_review, n),
            "reject_rate": _rate(reject, n),
        },
        "processing_time": {
            "manual_seconds_per_loan": manual_seconds,
            "harness_seconds_per_loan": round(avg_harness_seconds, 6),
            "reduction_pct": time_reduction_pct,
        },
        "error_rate": {
            "manual_baseline": MANUAL_ERROR_RATE,
            "post_harness": post_error_rate,
            "reduction_pct": error_rate_reduction_pct,
        },
        "compliance_score": {
            "manual_baseline": MANUAL_COMPLIANCE_SCORE,
            "post_harness": post_compliance_score,
            "uplift": compliance_uplift,
        },
        "cost_savings": {
            "manual_cost_per_loan_usd": round(manual_cost_per_loan, 2),
            "post_harness_cost_per_loan_usd": round(post_cost_per_loan, 2),
            "savings_per_loan_usd": savings_per_loan,
            "assumed_monthly_volume": ASSUMED_MONTHLY_LOAN_VOLUME,
            "projected_monthly_savings_usd": projected_monthly_savings,
            "projected_annual_savings_usd": projected_annual_savings,
        },
        "routing_accuracy": {
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "catch_rate": catch_rate,
            "clean_pass_rate": clean_pass_rate,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
    }
