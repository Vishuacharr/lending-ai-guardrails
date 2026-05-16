"""
Five end-to-end mortgage use cases that exercise the evaluation harness.

Each use_case_* function simulates:
  1. A mock AI output (what the AI system would produce)
  2. The source context it drew from
  3. The query it was answering
  4. Extracted fields + ground truth for accuracy scoring

All "AI outputs" are hard-coded strings representing realistic model outputs.
No LLM API calls are made.
"""

from __future__ import annotations

from core.evaluator import EvaluationHarness, EvaluationResult
from mortgage.rules import (
    check_closing_disclosure_balance,
    check_credit_decision_reason_codes,
    run_all_application_checks,
)
from mortgage.sample_data import (
    APPRAISAL_CONFLICTING,
    APPRAISAL_PRIMARY,
    APPLICATIONS,
    CD_BALANCED,
    CD_UNBALANCED,
    DECISION_APPROVED,
    DECISION_DENIED_INVALID_CODES,
    DECISION_DENIED_VALID,
    DOCUMENT_VERSIONS,
)
from mortgage.schemas import ComplianceCheck


def use_case_1_income_verification(harness: EvaluationHarness) -> EvaluationResult:
    """
    UC1: Income Verification.

    AI extracts gross income from a W-2. Harness checks extracted value
    against the borrower's stated income and validates field completeness.
    """
    app = APPLICATIONS["LA-001"]

    source_context = (
        "W-2 Wage and Tax Statement 2025 | Employer: Midwest Tech Solutions | "
        "Box 1 Wages, Tips: $145,000.00 | Box 2 Federal Tax Withheld: $28,950.00 | "
        "Box 3 Social Security Wages: $145,000.00 | Box 16 State Wages: $145,000.00 | "
        "Employee: Jennifer Martinez | SSN: ***-**-4821"
    )

    ai_output = (
        "Income verification complete for application LA-001. "
        "Gross annual income extracted from W-2 Box 1: $145,000.00. "
        "Monthly equivalent: $12,083.33. "
        "Income matches borrower-stated amount of $145,000 — no discrepancy detected. "
        "Document appears authentic and complete. Recommended: verified."
    )

    query = (
        "Extract the borrower's gross annual income from the provided W-2 document "
        "and compare against the stated income on the loan application LA-001."
    )

    extracted_fields = {
        "gross_income": 145_000.0,
        "monthly_income": 12_083.33,
        "income_period": "annual",
        "discrepancy_pct": 0.0,
    }

    ground_truth = {
        "gross_income": 145_000.0,
        "monthly_income": 12_083.33,
    }

    return harness.evaluate(
        use_case="income_verification",
        loan_application_id=app.application_id,
        ai_output=ai_output,
        source_context=source_context,
        query=query,
        extracted_fields=extracted_fields,
        ground_truth_fields=ground_truth,
        metadata={"document_type": "W2", "tax_year": 2025},
    )


def use_case_2_closing_disclosure(harness: EvaluationHarness) -> EvaluationResult:
    """
    UC2: Closing Disclosure Balancing.

    AI generates a Closing Disclosure. Harness validates that all line items
    sum correctly and flags TRID violations.
    """
    # Run compliance check before evaluation
    balance_check: ComplianceCheck = check_closing_disclosure_balance(CD_UNBALANCED)

    source_context = (
        "Loan Terms: $285,000 at 7.20% for 360 months. "
        "Closing Costs Section A: Origination $1,425. "
        "Section B: Appraisal $550. "
        "Section C: Title Insurance $950. "
        "Total closing costs stated on page 1: $9,000. "
        "Compliance flag: " + balance_check.finding
    )

    ai_output = (
        "Closing Disclosure generated for LA-002. "
        "Total closing costs: $9,000.00. "
        "Line items identified: Origination $1,425, Appraisal $550, Title $950. "
        "Total of itemized fees: $2,925. "
        "WARNING: CD does not balance. Discrepancy of $6,075 between stated total "
        "and sum of line items. TRID violation — requires correction before closing."
    )

    query = (
        "Validate the Closing Disclosure for application LA-002. "
        "Confirm all line items sum to the stated total closing costs. "
        "Flag any TRID compliance violations."
    )

    extracted_fields = {
        "loan_amount": 285_000.0,
        "closing_costs": 9_000.0,
        "line_item_total": 2_925.0,
        "discrepancy": 6_075.0,
        "balanced": False,
    }

    ground_truth = {
        "loan_amount": 285_000.0,
        "balanced": False,
        "discrepancy": 6_075.0,
    }

    return harness.evaluate(
        use_case="closing_disclosure",
        loan_application_id="LA-002",
        ai_output=ai_output,
        source_context=source_context,
        query=query,
        extracted_fields=extracted_fields,
        ground_truth_fields=ground_truth,
        metadata={
            "trid_check": balance_check.finding,
            "severity": balance_check.severity,
        },
    )


def use_case_3_appraisal_comparison(harness: EvaluationHarness) -> EvaluationResult:
    """
    UC3: Appraisal Comparison.

    AI compares two appraisal reports. Harness checks for value contradictions
    and flags whether the difference exceeds allowable variance.
    """
    val_a = APPRAISAL_PRIMARY.appraised_value
    val_b = APPRAISAL_CONFLICTING.appraised_value
    delta = abs(val_a - val_b)
    delta_pct = delta / val_a

    source_context = (
        f"Appraisal Report A ({APPRAISAL_PRIMARY.report_id}): "
        f"Appraised value ${val_a:,.0f}, Appraiser {APPRAISAL_PRIMARY.appraiser_name}, "
        f"Date {APPRAISAL_PRIMARY.report_date}, Condition {APPRAISAL_PRIMARY.condition}. "
        f"Comparable sales: 408 Maple Dr $365,000; 500 Birch Ln $380,000; 215 Elm St $372,000. "
        f"Appraisal Report B ({APPRAISAL_CONFLICTING.report_id}): "
        f"Appraised value ${val_b:,.0f}, Appraiser {APPRAISAL_CONFLICTING.appraiser_name}, "
        f"Date {APPRAISAL_CONFLICTING.report_date}, Condition {APPRAISAL_CONFLICTING.condition}. "
        f"Comparable sales: 101 Spruce Ave $335,000; 320 Cedar Rd $342,000; 88 Willow Way $345,000."
    )

    ai_output = (
        f"Appraisal comparison for LA-001: Two reports submitted with conflicting values. "
        f"Report A (APR-001-A) values property at ${val_a:,.0f}; "
        f"Report B (APR-001-B) values at ${val_b:,.0f}. "
        f"Variance: ${delta:,.0f} ({delta_pct:.1%}). "
        f"This exceeds the 10% maximum variance threshold. "
        f"Report A uses more recent comparables and C2 condition vs C3. "
        f"Recommendation: Order desk review or field review to reconcile. "
        f"Lower value (${val_b:,.0f}) should be used for LTV calculation per investor guideline."
    )

    query = (
        "Compare the two appraisal reports for application LA-001. "
        "Identify any contradictions in appraised value. "
        "Determine which value should be used for LTV calculation."
    )

    extracted_fields = {
        "appraised_value": val_b,  # Conservative value used
        "property_address": "412 Maple Drive, Springfield, IL",
        "value_variance_pct": round(delta_pct, 4),
        "conflict_detected": True,
        "recommended_value": val_b,
    }

    ground_truth = {
        "appraised_value": val_b,
        "conflict_detected": True,
    }

    return harness.evaluate(
        use_case="appraisal_comparison",
        loan_application_id="LA-001",
        ai_output=ai_output,
        source_context=source_context,
        query=query,
        extracted_fields=extracted_fields,
        ground_truth_fields=ground_truth,
        metadata={
            "report_a": APPRAISAL_PRIMARY.report_id,
            "report_b": APPRAISAL_CONFLICTING.report_id,
            "variance_usd": delta,
        },
    )


def use_case_4_credit_decision_explanation(harness: EvaluationHarness) -> EvaluationResult:
    """
    UC4: Credit Decision Explanation.

    AI generates reason codes for a denial. Harness validates codes against
    CFPB-approved list (ECOA/FCRA adverse action requirements).
    """
    compliance_check = check_credit_decision_reason_codes(DECISION_DENIED_INVALID_CODES)

    source_context = (
        "Credit Decision for LA-006: DENIED. "
        "Borrower credit score: 680. "
        "DTI ratio: 53.2% (exceeds 50% guideline). "
        "Stated reason codes: 999 (internal code), ABC (system error), 002 (insufficient income). "
        "CFPB Compliance Check: " + compliance_check.finding + ". "
        "Citation: " + compliance_check.cfpb_citation
    )

    ai_output = (
        "Credit decision explanation for LA-006 (DENIED). "
        "Primary denial reason: Excessive debt-to-income ratio of 53.2% (Code 010). "
        "Secondary reason: Insufficient income relative to obligations (Code 002). "
        "WARNING: Reason codes 999 and ABC are not recognized CFPB adverse action codes "
        "and must be replaced before issuing the adverse action notice. "
        "Compliant replacement codes suggested: 010 (excessive obligations) and 002 (insufficient income). "
        "Adverse action notice must be sent within 30 days per ECOA."
    )

    query = (
        "Review the credit decision for application LA-006. "
        "Validate that all reason codes comply with CFPB adverse action requirements. "
        "Flag any non-compliant codes and suggest corrections."
    )

    extracted_fields = {
        "decision": "DENIED",
        "reason_codes": ["999", "ABC", "002"],
        "compliant_codes": ["010", "002"],
        "invalid_codes_detected": ["999", "ABC"],
        "adverse_action_required": True,
    }

    ground_truth = {
        "decision": "DENIED",
        "adverse_action_required": True,
    }

    return harness.evaluate(
        use_case="credit_decision",
        loan_application_id="LA-006",
        ai_output=ai_output,
        source_context=source_context,
        query=query,
        extracted_fields=extracted_fields,
        ground_truth_fields=ground_truth,
        metadata={
            "compliance_check_passed": compliance_check.passed,
            "cfpb_citation": compliance_check.cfpb_citation,
        },
    )


def use_case_5_document_version_control(harness: EvaluationHarness) -> EvaluationResult:
    """
    UC5: Document Version Control.

    AI identifies which version of a W-2 document is authoritative.
    Harness validates the lineage chain and authoritative flag.
    """
    versions = DOCUMENT_VERSIONS
    authoritative = next(v for v in versions if v.is_authoritative)
    latest_version = max(v.version for v in versions)

    source_context = (
        f"Document lineage for application LA-005, W-2 documents: "
        f"Version 1 (DOC-001-v1): uploaded {versions[0].upload_date} by borrower_portal, "
        f"not authoritative. "
        f"Version 2 (DOC-001-v2): uploaded {versions[1].upload_date} by processor_jsmith, "
        f"supersedes v1, not authoritative. "
        f"Version 3 (DOC-001-v3): uploaded {versions[2].upload_date} by underwriter_kdavis, "
        f"supersedes v2, IS AUTHORITATIVE — marked by underwriter as final certified copy."
    )

    ai_output = (
        f"Document version analysis for application LA-005 W-2 documents. "
        f"Three versions found in the loan file. "
        f"Authoritative document identified: {authoritative.document_id} (Version {authoritative.version}). "
        f"Uploaded on {authoritative.upload_date} by {authoritative.uploaded_by}. "
        f"Lineage chain: DOC-001-v1 → DOC-001-v2 → DOC-001-v3 (current authoritative). "
        f"Versions 1 and 2 are superseded and should not be used for underwriting decisions. "
        f"Recommend: Lock document lineage and prevent further uploads without underwriter approval."
    )

    query = (
        "Identify the authoritative version of the W-2 document for application LA-005. "
        "Trace the full document lineage chain and confirm which version should be used "
        "for underwriting and income calculation."
    )

    extracted_fields = {
        "authoritative_version": authoritative.document_id,
        "document_id": "DOC-001-v3",
        "version_count": len(versions),
        "latest_version": latest_version,
        "lineage_intact": True,
    }

    ground_truth = {
        "authoritative_version": authoritative.document_id,
        "document_id": "DOC-001-v3",
    }

    return harness.evaluate(
        use_case="document_lineage",
        loan_application_id="LA-005",
        ai_output=ai_output,
        source_context=source_context,
        query=query,
        extracted_fields=extracted_fields,
        ground_truth_fields=ground_truth,
        metadata={
            "total_versions": len(versions),
            "authoritative_uploaded_by": authoritative.uploaded_by,
        },
    )


def run_all_use_cases(harness: EvaluationHarness) -> list[EvaluationResult]:
    """Run all five use cases and return results in order."""
    return [
        use_case_1_income_verification(harness),
        use_case_2_closing_disclosure(harness),
        use_case_3_appraisal_comparison(harness),
        use_case_4_credit_decision_explanation(harness),
        use_case_5_document_version_control(harness),
    ]
