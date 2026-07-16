"""
Batch validation scenarios for the evaluation harness.

24 anonymized sample loan records (synthetic names/SSNs/addresses) covering:
  - Clean loans that should sail through (AUTO_APPROVE)
  - Income mismatches between stated and document-extracted income
  - Document version conflicts (wrong/stale version surfaced)
  - Closing Disclosure imbalances (TRID violations)
  - Compliance violations (invalid reason codes, LTV/DTI breaches)

Each scenario carries a `has_issue` ground-truth label used to measure the
harness's false positive / false negative rate on real KPI validation runs.
No LLM calls — all "AI outputs" are hard-coded strings simulating what a
document-extraction or decisioning model would have produced.
"""

from __future__ import annotations

from typing import Any, TypedDict


class Scenario(TypedDict):
    id: str
    use_case: str
    loan_id: str
    ai_output: str
    source_context: str
    query: str
    extracted_fields: dict[str, Any]
    ground_truth_fields: dict[str, Any]
    has_issue: bool
    issue_type: str
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# 1-8: Clean loans — no injected issue, harness should AUTO_APPROVE
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    {
        "id": "S-001",
        "use_case": "income_verification",
        "loan_id": "LA-101",
        "ai_output": (
            "Income verification complete for application LA-101. Gross annual "
            "income extracted from W-2 Box 1: $92,400.00. Monthly equivalent: "
            "$7,700.00. Income matches borrower-stated amount — no discrepancy."
        ),
        "source_context": (
            "W-2 Wage and Tax Statement 2025 | Employer: Lakeside Manufacturing | "
            "Box 1 Wages: $92,400.00 | Employee: Maria Gonzalez | SSN: ***-**-3312"
        ),
        "query": "Extract gross annual income from the W-2 and compare to stated income.",
        "extracted_fields": {"gross_income": 92_400.0, "monthly_income": 7_700.0, "income_period": "annual"},
        "ground_truth_fields": {"gross_income": 92_400.0, "monthly_income": 7_700.0},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"document_type": "W2"},
    },
    {
        "id": "S-002",
        "use_case": "income_verification",
        "loan_id": "LA-102",
        "ai_output": (
            "Income verification complete for application LA-102. Gross annual "
            "income extracted from paystub YTD: $118,300.00. Monthly equivalent: "
            "$9,858.33. Matches stated income of $118,300 — verified."
        ),
        "source_context": (
            "Paystub | Pay Period Ending 2026-06-15 | YTD Gross: $59,150.00 (mid-year) "
            "annualized $118,300.00 | Employee: Thomas Baker"
        ),
        "query": "Extract gross annual income from the paystub and verify against stated income.",
        "extracted_fields": {"gross_income": 118_300.0, "monthly_income": 9_858.33, "income_period": "annual"},
        "ground_truth_fields": {"gross_income": 118_300.0, "monthly_income": 9_858.33},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"document_type": "PAYSTUB"},
    },
    {
        "id": "S-003",
        "use_case": "appraisal_comparison",
        "loan_id": "LA-103",
        "ai_output": (
            "Appraisal review for LA-103: single report submitted, no conflicting "
            "values. Appraised value $310,000 supported by three comparable sales "
            "within 5% variance. No conflict detected — value accepted as-is."
        ),
        "source_context": (
            "Appraisal Report APR-103-A: Appraised value $310,000, Condition C3. "
            "Comparables: 12 River Rd $305,000; 40 Cliff Ave $315,000; 8 Bay St $308,000."
        ),
        "query": "Review appraisal for LA-103 and confirm value is supportable.",
        "extracted_fields": {
            "appraised_value": 310_000.0,
            "property_address": "77 Sunset Ct, Denver, CO",
            "value_variance_pct": 0.02,
            "conflict_detected": False,
        },
        "ground_truth_fields": {"appraised_value": 310_000.0, "conflict_detected": False},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"document_type": "APPRAISAL"},
    },
    {
        "id": "S-004",
        "use_case": "credit_decision",
        "loan_id": "LA-104",
        "ai_output": (
            "Credit decision explanation for LA-104 (APPROVED). Borrower meets all "
            "underwriting guidelines. No adverse action required — reason code "
            "check not applicable for approvals."
        ),
        "source_context": (
            "Credit Decision for LA-104: APPROVED. Credit score 745. DTI 32%. "
            "No adverse action notice required."
        ),
        "query": "Review the credit decision for LA-104 and confirm compliance requirements.",
        "extracted_fields": {"decision": "APPROVED", "reason_codes": [], "adverse_action_required": False},
        "ground_truth_fields": {"decision": "APPROVED", "adverse_action_required": False},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"compliance_check_passed": True},
    },
    {
        "id": "S-005",
        "use_case": "document_lineage",
        "loan_id": "LA-105",
        "ai_output": (
            "Document version analysis for LA-105 bank statements. Two versions "
            "found. Authoritative document identified: DOC-105-v2 (Version 2), "
            "uploaded by underwriter_mlee. Lineage chain intact: v1 -> v2."
        ),
        "source_context": (
            "Document lineage for LA-105: Version 1 (DOC-105-v1) uploaded by borrower_portal, "
            "not authoritative. Version 2 (DOC-105-v2) uploaded by underwriter_mlee, "
            "supersedes v1, IS AUTHORITATIVE."
        ),
        "query": "Identify the authoritative bank statement version for LA-105.",
        "extracted_fields": {
            "authoritative_version": "DOC-105-v2",
            "document_id": "DOC-105-v2",
            "version_count": 2,
            "lineage_intact": True,
        },
        "ground_truth_fields": {"authoritative_version": "DOC-105-v2", "document_id": "DOC-105-v2"},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"document_type": "BANK_STATEMENT"},
    },
    {
        "id": "S-006",
        "use_case": "closing_disclosure",
        "loan_id": "LA-106",
        "ai_output": (
            "Closing Disclosure generated for LA-106. Total closing costs: "
            "$8,400.00. Line items sum to $8,400.00 — CD balances. No TRID "
            "violation detected."
        ),
        "source_context": (
            "Loan Terms: $265,000 at 6.90% for 360 months. Section A: Origination "
            "$1,325. Section B: Appraisal $525. Section C: Title $1,050. Section E: "
            "Recording $110. Section F: HOI $1,590. Section G: Taxes $1,900. "
            "Section H: Prepaid Interest $1,900. Total stated: $8,400."
        ),
        "query": "Validate the Closing Disclosure for LA-106 sums correctly.",
        "extracted_fields": {
            "loan_amount": 265_000.0,
            "closing_costs": 8_400.0,
            "line_item_total": 8_400.0,
            "discrepancy": 0.0,
            "balanced": True,
        },
        "ground_truth_fields": {"loan_amount": 265_000.0, "balanced": True, "discrepancy": 0.0},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"trid_check": "Closing Disclosure balances"},
    },
    {
        "id": "S-007",
        "use_case": "income_verification",
        "loan_id": "LA-107",
        "ai_output": (
            "Income verification complete for LA-107. Gross annual income "
            "extracted from tax return line 1: $76,500.00. Monthly equivalent: "
            "$6,375.00. Matches stated income — verified, no discrepancy."
        ),
        "source_context": (
            "Form 1040 2025 | Line 1a Wages: $76,500.00 | Filer: Angela Brooks | "
            "SSN: ***-**-9021"
        ),
        "query": "Extract gross annual income from the tax return and verify against stated income.",
        "extracted_fields": {"gross_income": 76_500.0, "monthly_income": 6_375.0, "income_period": "annual"},
        "ground_truth_fields": {"gross_income": 76_500.0, "monthly_income": 6_375.0},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"document_type": "TAX_RETURN"},
    },
    {
        "id": "S-008",
        "use_case": "credit_decision",
        "loan_id": "LA-108",
        "ai_output": (
            "Credit decision explanation for LA-108 (DENIED). Reason codes 007 "
            "(insufficient collateral value) and 010 (excessive obligations) both "
            "verified against CFPB-approved list. Adverse action notice compliant."
        ),
        "source_context": (
            "Credit Decision for LA-108: DENIED. Reason codes: 007, 010. Both codes "
            "present on CFPB standard list. Adverse action notice required within 30 days."
        ),
        "query": "Review the credit decision for LA-108 for CFPB reason code compliance.",
        "extracted_fields": {
            "decision": "DENIED",
            "reason_codes": ["007", "010"],
            "invalid_codes_detected": [],
            "adverse_action_required": True,
        },
        "ground_truth_fields": {"decision": "DENIED", "adverse_action_required": True},
        "has_issue": False,
        "issue_type": "clean",
        "metadata": {"compliance_check_passed": True},
    },

    # -----------------------------------------------------------------------
    # 9-12: Income mismatches
    # -----------------------------------------------------------------------
    {
        "id": "S-009",
        "use_case": "income_verification",
        "loan_id": "LA-109",
        "ai_output": (
            "Income verification for LA-109. Gross annual income extracted from "
            "W-2 Box 1: $61,200.00. Monthly equivalent: $5,100.00. Borrower "
            "stated income on application was $84,000 — DISCREPANCY of 27.1% "
            "detected. Requires manual review before proceeding."
        ),
        "source_context": (
            "W-2 Wage and Tax Statement 2025 | Employer: Coastal Retail Group | "
            "Box 1 Wages: $61,200.00 | Employee: Daniel Osei | SSN: ***-**-6674"
        ),
        "query": "Extract gross annual income from the W-2 and compare against stated income.",
        "extracted_fields": {
            "gross_income": 61_200.0,
            "monthly_income": 5_100.0,
            "income_period": "annual",
            "discrepancy_pct": 0.271,
        },
        "ground_truth_fields": {"gross_income": 61_200.0, "monthly_income": 5_100.0},
        "has_issue": True,
        "issue_type": "income_mismatch",
        "metadata": {"document_type": "W2", "stated_income": 84_000.0},
    },
    {
        "id": "S-010",
        "use_case": "income_verification",
        "loan_id": "LA-110",
        "ai_output": (
            "Income verification for LA-110. Gross annual income extracted from "
            "paystub YTD annualized: $138,000.00. Borrower stated income was "
            "$102,000 — extracted value is 35.3% HIGHER than stated. Flag for "
            "underwriter: possible bonus/commission inflation or extraction error."
        ),
        "source_context": (
            "Paystub | Pay Period Ending 2026-05-30 | YTD Gross (annualized): "
            "$138,000.00 includes non-recurring bonus | Employee: Priya Nair"
        ),
        "query": "Extract gross annual income from paystub and verify against stated income.",
        "extracted_fields": {
            "gross_income": 138_000.0,
            "monthly_income": 11_500.0,
            "income_period": "annual",
            "discrepancy_pct": 0.353,
        },
        "ground_truth_fields": {"gross_income": 138_000.0, "monthly_income": 11_500.0},
        "has_issue": True,
        "issue_type": "income_mismatch",
        "metadata": {"document_type": "PAYSTUB", "stated_income": 102_000.0},
    },
    {
        "id": "S-011",
        "use_case": "income_verification",
        "loan_id": "LA-111",
        "ai_output": (
            "Income verification for LA-111. Gross annual income extracted from "
            "tax return: $54,900.00. Borrower stated income on the loan "
            "application was $71,500 — DISCREPANCY of 23.2%. Self-employed "
            "borrower; review Schedule C deductions before finalizing."
        ),
        "source_context": (
            "Form 1040 2025 Schedule C | Net Profit Line 31: $54,900.00 | "
            "Filer: Marcus Webb | SSN: ***-**-4487"
        ),
        "query": "Extract net income from the tax return and verify against stated income.",
        "extracted_fields": {
            "gross_income": 54_900.0,
            "monthly_income": 4_575.0,
            "income_period": "annual",
            "discrepancy_pct": 0.232,
        },
        "ground_truth_fields": {"gross_income": 54_900.0, "monthly_income": 4_575.0},
        "has_issue": True,
        "issue_type": "income_mismatch",
        "metadata": {"document_type": "TAX_RETURN", "stated_income": 71_500.0},
    },
    {
        "id": "S-012",
        "use_case": "income_verification",
        "loan_id": "LA-112",
        "ai_output": (
            "Income verification for LA-112. Gross annual income extracted from "
            "W-2 Box 1: $99,750.00. Borrower stated income was $115,000 — "
            "DISCREPANCY of 13.3%. Below the 15% auto-flag threshold but still "
            "outside plausible tolerance for auto-approval."
        ),
        "source_context": (
            "W-2 Wage and Tax Statement 2025 | Employer: Horizon Logistics | "
            "Box 1 Wages: $99,750.00 | Employee: Rachel Kim | SSN: ***-**-2298"
        ),
        "query": "Extract gross annual income from the W-2 and compare against stated income.",
        "extracted_fields": {
            "gross_income": 99_750.0,
            "monthly_income": 8_312.50,
            "income_period": "annual",
            "discrepancy_pct": 0.133,
        },
        "ground_truth_fields": {"gross_income": 99_750.0, "monthly_income": 8_312.50},
        "has_issue": True,
        "issue_type": "income_mismatch",
        "metadata": {"document_type": "W2", "stated_income": 115_000.0},
    },

    # -----------------------------------------------------------------------
    # 13-16: Document version conflicts
    # -----------------------------------------------------------------------
    {
        "id": "S-013",
        "use_case": "document_lineage",
        "loan_id": "LA-113",
        "ai_output": (
            "Document version analysis for LA-113 W-2 documents. AI identified "
            "DOC-113-v1 as authoritative, but v2 was uploaded later by the "
            "underwriter and supersedes v1. INCORRECT VERSION SELECTED — stale "
            "document would have been used for income calculation."
        ),
        "source_context": (
            "Document lineage for LA-113: Version 1 (DOC-113-v1) uploaded by "
            "borrower_portal, NOT authoritative. Version 2 (DOC-113-v2) uploaded "
            "by underwriter_rjones, supersedes v1, IS AUTHORITATIVE — final certified copy."
        ),
        "query": "Identify the authoritative W-2 version for LA-113.",
        "extracted_fields": {
            "authoritative_version": "DOC-113-v1",
            "document_id": "DOC-113-v1",
            "version_count": 2,
            "lineage_intact": False,
        },
        "ground_truth_fields": {"authoritative_version": "DOC-113-v2", "document_id": "DOC-113-v2"},
        "has_issue": True,
        "issue_type": "document_version_conflict",
        "metadata": {"document_type": "W2"},
    },
    {
        "id": "S-014",
        "use_case": "document_lineage",
        "loan_id": "LA-114",
        "ai_output": (
            "Document version analysis for LA-114 bank statements. Three "
            "versions found in the loan file, but lineage chain is BROKEN — "
            "v2 does not reference v1 as its predecessor. Cannot confirm v3 is "
            "genuinely the latest authoritative copy without manual trace."
        ),
        "source_context": (
            "Document lineage for LA-114: Version 1 (DOC-114-v1) uploaded by "
            "borrower_portal. Version 2 (DOC-114-v2) uploaded by processor — "
            "supersedes field is MISSING/NULL. Version 3 (DOC-114-v3) marked "
            "authoritative by underwriter_tlee, supersedes v2."
        ),
        "query": "Identify the authoritative bank statement version for LA-114 and confirm lineage.",
        "extracted_fields": {
            "authoritative_version": "DOC-114-v3",
            "document_id": "DOC-114-v2",
            "version_count": 3,
            "lineage_intact": False,
        },
        "ground_truth_fields": {"authoritative_version": "DOC-114-v3", "document_id": "DOC-114-v3"},
        "has_issue": True,
        "issue_type": "document_version_conflict",
        "metadata": {"document_type": "BANK_STATEMENT"},
    },
    {
        "id": "S-015",
        "use_case": "document_lineage",
        "loan_id": "LA-115",
        "ai_output": (
            "Document version analysis for LA-115 appraisal documents. AI "
            "returned DOC-115-v1 as authoritative. Cross-check shows two "
            "competing 'authoritative' flags set (v1 and v2) — data integrity "
            "conflict in source system, cannot resolve automatically."
        ),
        "source_context": (
            "Document lineage for LA-115: Version 1 (DOC-115-v1) flagged "
            "authoritative=true by processor error. Version 2 (DOC-115-v2) "
            "flagged authoritative=true by underwriter_kpatel, supersedes v1. "
            "CONFLICTING AUTHORITATIVE FLAGS."
        ),
        "query": "Identify the single authoritative appraisal version for LA-115.",
        "extracted_fields": {
            "authoritative_version": "DOC-115-v1",
            "document_id": "DOC-115-v1",
            "version_count": 2,
            "lineage_intact": False,
        },
        "ground_truth_fields": {"authoritative_version": "DOC-115-v2", "document_id": "DOC-115-v2"},
        "has_issue": True,
        "issue_type": "document_version_conflict",
        "metadata": {"document_type": "APPRAISAL"},
    },
    {
        "id": "S-016",
        "use_case": "document_lineage",
        "loan_id": "LA-116",
        "ai_output": (
            "Document version analysis for LA-116 closing disclosure drafts. "
            "Four revisions found — AI selected v3 as authoritative, but v4 "
            "was uploaded 2 hours later by the closer with the final rate lock "
            "adjustment. Superseded draft would have been used."
        ),
        "source_context": (
            "Document lineage for LA-116: v1-v3 are drafts uploaded by processor. "
            "Version 4 (DOC-116-v4) uploaded by closer_dmartin, supersedes v3, "
            "IS AUTHORITATIVE — reflects final rate lock."
        ),
        "query": "Identify the authoritative closing disclosure draft for LA-116.",
        "extracted_fields": {
            "authoritative_version": "DOC-116-v3",
            "document_id": "DOC-116-v3",
            "version_count": 4,
            "lineage_intact": False,
        },
        "ground_truth_fields": {"authoritative_version": "DOC-116-v4", "document_id": "DOC-116-v4"},
        "has_issue": True,
        "issue_type": "document_version_conflict",
        "metadata": {"document_type": "CLOSING_DISCLOSURE"},
    },

    # -----------------------------------------------------------------------
    # 17-20: Closing Disclosure imbalances
    # -----------------------------------------------------------------------
    {
        "id": "S-017",
        "use_case": "closing_disclosure",
        "loan_id": "LA-117",
        "ai_output": (
            "Closing Disclosure generated for LA-117. Total closing costs: "
            "$7,800.00. Line items identified sum to $5,400.00. WARNING: CD "
            "does not balance. Discrepancy of $2,400. TRID violation — "
            "requires correction before closing."
        ),
        "source_context": (
            "Loan Terms: $310,000 at 7.05% for 360 months. Section A: Origination "
            "$1,550. Section B: Appraisal $600. Section C: Title $1,100. Section E: "
            "Recording $130. Total stated closing costs on page 1: $7,800. "
            "Sum of listed line items only reaches $5,400 — two sections missing."
        ),
        "query": "Validate the Closing Disclosure for LA-117 sums to the stated total.",
        "extracted_fields": {
            "loan_amount": 310_000.0,
            "closing_costs": 7_800.0,
            "line_item_total": 5_400.0,
            "discrepancy": 2_400.0,
            "balanced": False,
        },
        "ground_truth_fields": {"loan_amount": 310_000.0, "balanced": False, "discrepancy": 2_400.0},
        "has_issue": True,
        "issue_type": "cd_imbalance",
        "metadata": {"severity": "VIOLATION"},
    },
    {
        "id": "S-018",
        "use_case": "closing_disclosure",
        "loan_id": "LA-118",
        "ai_output": (
            "Closing Disclosure generated for LA-118. Total closing costs: "
            "$11,200.00. Line item sum: $12,850.00. WARNING: CD OVER-balances "
            "by $1,650 — itemized fees exceed stated total. TRID violation, "
            "requires correction."
        ),
        "source_context": (
            "Loan Terms: $340,000 at 7.10% for 360 months. Itemized sections A-H "
            "sum to $12,850. Total stated closing costs on page 1: $11,200. "
            "Overage of $1,650 not explained by any credit line."
        ),
        "query": "Validate the Closing Disclosure for LA-118 sums to the stated total.",
        "extracted_fields": {
            "loan_amount": 340_000.0,
            "closing_costs": 11_200.0,
            "line_item_total": 12_850.0,
            "discrepancy": 1_650.0,
            "balanced": False,
        },
        "ground_truth_fields": {"loan_amount": 340_000.0, "balanced": False, "discrepancy": 1_650.0},
        "has_issue": True,
        "issue_type": "cd_imbalance",
        "metadata": {"severity": "VIOLATION"},
    },
    {
        "id": "S-019",
        "use_case": "closing_disclosure",
        "loan_id": "LA-119",
        "ai_output": (
            "Closing Disclosure generated for LA-119. Total closing costs: "
            "$6,050.00. Line item sum: $6,050.00 — appears balanced at the "
            "summary level, but prepaid items total ($4,200) was double-counted "
            "inside both Section G and Section H. Net effect: cash-to-close "
            "understated by $2,100."
        ),
        "source_context": (
            "Loan Terms: $198,000 at 6.80% for 360 months. Section G Prepaid "
            "Taxes $2,100 and Section H Prepaid Interest lists the same $2,100 "
            "figure erroneously duplicated from Section G. Cash to close should "
            "be $2,100 higher than disclosed."
        ),
        "query": "Validate the Closing Disclosure for LA-119 for duplicate or miscategorized line items.",
        "extracted_fields": {
            "loan_amount": 198_000.0,
            "closing_costs": 6_050.0,
            "line_item_total": 6_050.0,
            "discrepancy": 2_100.0,
            "balanced": False,
        },
        "ground_truth_fields": {"loan_amount": 198_000.0, "balanced": False, "discrepancy": 2_100.0},
        "has_issue": True,
        "issue_type": "cd_imbalance",
        "metadata": {"severity": "VIOLATION"},
    },
    {
        "id": "S-020",
        "use_case": "closing_disclosure",
        "loan_id": "LA-120",
        "ai_output": (
            "Closing Disclosure generated for LA-120. Total closing costs: "
            "$9,900.00. Line item sum: $7,300.00. CD does not balance — "
            "discrepancy of $2,600. Missing Section F (insurance) entirely "
            "from the itemization. TRID violation, correction required."
        ),
        "source_context": (
            "Loan Terms: $275,000 at 6.95% for 360 months. Sections A, B, C, E, "
            "G, H itemized, summing to $7,300. Section F (Prepaids/Insurance) "
            "omitted from the generated disclosure. Stated total: $9,900."
        ),
        "query": "Validate the Closing Disclosure for LA-120 sums to the stated total.",
        "extracted_fields": {
            "loan_amount": 275_000.0,
            "closing_costs": 9_900.0,
            "line_item_total": 7_300.0,
            "discrepancy": 2_600.0,
            "balanced": False,
        },
        "ground_truth_fields": {"loan_amount": 275_000.0, "balanced": False, "discrepancy": 2_600.0},
        "has_issue": True,
        "issue_type": "cd_imbalance",
        "metadata": {"severity": "VIOLATION"},
    },

    # -----------------------------------------------------------------------
    # 21-24: Compliance violations (invalid reason codes / LTV / DTI)
    # -----------------------------------------------------------------------
    {
        "id": "S-021",
        "use_case": "credit_decision",
        "loan_id": "LA-121",
        "ai_output": (
            "Credit decision explanation for LA-121 (DENIED). Primary reason: "
            "internal risk code 'RSK-9' used instead of a CFPB-approved reason "
            "code. Code is NOT on the ECOA/FCRA approved list — adverse action "
            "notice cannot be issued as-is."
        ),
        "source_context": (
            "Credit Decision for LA-121: DENIED. Reason codes: RSK-9, 014. "
            "RSK-9 is an internal proprietary code not mapped to any CFPB "
            "standard adverse action reason."
        ),
        "query": "Review the credit decision for LA-121 for CFPB reason code compliance.",
        "extracted_fields": {
            "decision": "DENIED",
            "reason_codes": ["RSK-9", "014"],
            "invalid_codes_detected": ["RSK-9"],
            "adverse_action_required": True,
        },
        "ground_truth_fields": {"decision": "DENIED", "adverse_action_required": True},
        "has_issue": True,
        "issue_type": "compliance_violation",
        "metadata": {"compliance_check_passed": False, "cfpb_citation": "15 U.S.C. 1691(d)"},
    },
    {
        "id": "S-022",
        "use_case": "credit_decision",
        "loan_id": "LA-122",
        "ai_output": (
            "Credit decision explanation for LA-122 (DENIED). Reason codes "
            "'X01' and '077' returned by the decision engine. Neither code "
            "appears in the CFPB-approved list — both flagged as non-compliant, "
            "adverse action notice blocked pending correction."
        ),
        "source_context": (
            "Credit Decision for LA-122: DENIED. Reason codes: X01, 077. "
            "Neither code exists in the standard CFPB adverse action code table."
        ),
        "query": "Review the credit decision for LA-122 for CFPB reason code compliance.",
        "extracted_fields": {
            "decision": "DENIED",
            "reason_codes": ["X01", "077"],
            "invalid_codes_detected": ["X01", "077"],
            "adverse_action_required": True,
        },
        "ground_truth_fields": {"decision": "DENIED", "adverse_action_required": True},
        "has_issue": True,
        "issue_type": "compliance_violation",
        "metadata": {"compliance_check_passed": False},
    },
    {
        "id": "S-023",
        "use_case": "appraisal_comparison",
        "loan_id": "LA-123",
        "ai_output": (
            "Appraisal comparison for LA-123: two reports submitted with "
            "conflicting values. Report A values property at $420,000; Report "
            "B at $355,000. Variance: $65,000 (15.5%) — exceeds the 10% "
            "maximum variance threshold. AI selected the HIGHER value for LTV "
            "calculation, which violates conservative-value investor guideline."
        ),
        "source_context": (
            "Appraisal Report A (APR-123-A): $420,000, Condition C2. "
            "Appraisal Report B (APR-123-B): $355,000, Condition C4. "
            "Investor guideline requires the LOWER of two conflicting values "
            "be used for LTV when variance exceeds 10%."
        ),
        "query": "Compare appraisal reports for LA-123 and determine correct LTV value per guideline.",
        "extracted_fields": {
            "appraised_value": 420_000.0,
            "property_address": "300 Highland Ave, Portland, OR",
            "value_variance_pct": 0.155,
            "conflict_detected": True,
            "recommended_value": 420_000.0,
        },
        "ground_truth_fields": {"appraised_value": 355_000.0, "conflict_detected": True},
        "has_issue": True,
        "issue_type": "compliance_violation",
        "metadata": {"variance_usd": 65_000.0},
    },
    {
        "id": "S-024",
        "use_case": "closing_disclosure",
        "loan_id": "LA-124",
        "ai_output": (
            "Closing Disclosure generated for LA-124. Total closing costs: "
            "$15,400.00 against a $180,000 loan — closing costs represent "
            "8.6% of loan amount, well above the typical 3-5% norm. Combined "
            "with a 98.5% LTV, this exceeds conventional investor LTV limits. "
            "Compliance flag raised on LTV overlay."
        ),
        "source_context": (
            "Loan Terms: $180,000 conventional loan against $182,700 appraised "
            "value (98.5% LTV — exceeds 97% conventional maximum). Closing "
            "costs itemized at $15,400, matching the stated total (balances "
            "correctly), but LTV overlay violation present."
        ),
        "query": "Validate the Closing Disclosure and LTV compliance for LA-124.",
        "extracted_fields": {
            "loan_amount": 180_000.0,
            "closing_costs": 15_400.0,
            "line_item_total": 15_400.0,
            "discrepancy": 0.0,
            "balanced": True,
            "ltv_ratio": 0.985,
        },
        "ground_truth_fields": {"loan_amount": 180_000.0, "balanced": True, "discrepancy": 0.0},
        "has_issue": True,
        "issue_type": "compliance_violation",
        "metadata": {"ltv_ratio": 0.985, "ltv_limit": 0.97, "severity": "VIOLATION"},
    },
]


def get_scenarios_by_issue_type(issue_type: str) -> list[Scenario]:
    """Return all scenarios matching a given issue_type."""
    return [s for s in SCENARIOS if s["issue_type"] == issue_type]


def scenario_counts() -> dict[str, int]:
    """Return a count of scenarios per issue_type, for reporting."""
    counts: dict[str, int] = {}
    for s in SCENARIOS:
        counts[s["issue_type"]] = counts.get(s["issue_type"], 0) + 1
    return counts
