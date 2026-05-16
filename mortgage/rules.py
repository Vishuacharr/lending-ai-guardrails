"""
Simplified TRID/RESPA compliance rules for the evaluation harness.

These rules are illustrative — they capture the structure of real
compliance checks without attempting to be a complete regulatory engine.

References:
  - TRID: 12 CFR Part 1026 (CFPB TILA-RESPA Integrated Disclosure)
  - RESPA: 12 CFR Part 1024
  - ECOA/FCRA adverse action reason codes
"""

from __future__ import annotations

from mortgage.schemas import (
    ComplianceCheck,
    ClosingDisclosure,
    CreditDecision,
    LoanApplication,
)

CFPB_VALID_REASON_CODES = {
    "001": "Derogatory credit history",
    "002": "Insufficient income",
    "003": "Unable to verify income",
    "004": "Temporary or irregular employment",
    "005": "Unable to verify employment",
    "006": "Length of employment",
    "007": "Insufficient collateral value",
    "008": "Unacceptable property",
    "009": "Unable to verify credit references",
    "010": "Excessive obligations in relation to income",
    "011": "Inadequate down payment for property type",
    "012": "Mortgage delinquency or foreclosure",
    "013": "Too few credit references",
    "014": "Garnishment or attachment",
    "015": "Insufficient liquid assets to close",
    "016": "Bankruptcy",
    "017": "Number of recent inquiries on credit bureau report",
    "018": "Length of residence",
    "019": "Credit application incomplete",
    "020": "Unable to verify assets",
}

MAX_CONVENTIONAL_LTV = 0.97
MAX_FHA_LTV = 0.9650
MAX_DTI = 0.50
MAX_CLOSING_COST_TOLERANCE = 0.001  # $0.01 tolerance on CD balance


def check_ltv(application: LoanApplication) -> ComplianceCheck:
    """TRID Rule: LTV must not exceed product-specific limits."""
    ltv = application.ltv_ratio
    if application.loan_type.value == "FHA":
        limit = MAX_FHA_LTV
    elif application.loan_type.value in ("VA", "USDA"):
        limit = 1.0  # VA/USDA allow 100% financing
    else:
        limit = MAX_CONVENTIONAL_LTV

    passed = ltv <= limit
    return ComplianceCheck(
        application_id=application.application_id,
        check_name="LTV Limit",
        passed=passed,
        finding=(
            f"LTV {ltv:.2%} {'within' if passed else 'exceeds'} "
            f"{application.loan_type.value} limit of {limit:.2%}"
        ),
        severity="VIOLATION" if not passed else "INFO",
        cfpb_citation="12 CFR 1026.43(e)",
    )


def check_dti(application: LoanApplication) -> ComplianceCheck:
    """ATR Rule: DTI above 50% requires compensating factors."""
    dti = application.dti_ratio
    passed = dti <= MAX_DTI
    return ComplianceCheck(
        application_id=application.application_id,
        check_name="DTI Limit",
        passed=passed,
        finding=(
            f"DTI {dti:.2%} {'within' if passed else 'exceeds'} "
            f"maximum allowed {MAX_DTI:.0%}"
        ),
        severity="VIOLATION" if not passed else "INFO",
        cfpb_citation="12 CFR 1026.43(c)(2)(vi)",
    )


def check_closing_disclosure_balance(cd: ClosingDisclosure) -> ComplianceCheck:
    """TRID Rule: Closing Disclosure must balance within $0.01."""
    delta = cd.compute_balance()
    passed = delta <= MAX_CLOSING_COST_TOLERANCE
    return ComplianceCheck(
        application_id=cd.application_id,
        check_name="CD Balance",
        passed=passed,
        finding=(
            f"Closing Disclosure {'balances' if passed else f'out of balance by ${delta:.2f}'}"
        ),
        severity="VIOLATION" if not passed else "INFO",
        cfpb_citation="12 CFR 1026.38",
    )


def check_credit_decision_reason_codes(decision: CreditDecision) -> ComplianceCheck:
    """ECOA/FCRA: Adverse action notices must use CFPB-approved reason codes."""
    if decision.decision.value in ("APPROVED", "WITHDRAWN"):
        return ComplianceCheck(
            application_id=decision.application_id,
            check_name="Adverse Action Reason Codes",
            passed=True,
            finding="No adverse action — reason code check not required.",
            severity="INFO",
        )
    invalid = [
        code
        for code in decision.reason_codes
        if code not in CFPB_VALID_REASON_CODES
    ]
    passed = len(invalid) == 0
    return ComplianceCheck(
        application_id=decision.application_id,
        check_name="Adverse Action Reason Codes",
        passed=passed,
        finding=(
            "All reason codes valid."
            if passed
            else f"Invalid reason codes: {invalid}"
        ),
        severity="VIOLATION" if not passed else "INFO",
        cfpb_citation="15 U.S.C. 1691(d); 15 U.S.C. 1681m",
    )


def check_minimum_credit_score(application: LoanApplication) -> ComplianceCheck:
    """Product overlays: minimum FICO by loan type."""
    minimums = {"FHA": 580, "CONVENTIONAL": 620, "VA": 580, "USDA": 640, "JUMBO": 700}
    minimum = minimums.get(application.loan_type.value, 620)
    score = application.primary_borrower.credit_score
    passed = score >= minimum
    return ComplianceCheck(
        application_id=application.application_id,
        check_name="Minimum Credit Score",
        passed=passed,
        finding=(
            f"Credit score {score} {'meets' if passed else 'below'} "
            f"{application.loan_type.value} minimum of {minimum}"
        ),
        severity="VIOLATION" if not passed else "INFO",
        cfpb_citation="Product Overlay / Investor Guideline",
    )


def run_all_application_checks(application: LoanApplication) -> list[ComplianceCheck]:
    """Run all application-level compliance checks and return results."""
    return [
        check_ltv(application),
        check_dti(application),
        check_minimum_credit_score(application),
    ]
