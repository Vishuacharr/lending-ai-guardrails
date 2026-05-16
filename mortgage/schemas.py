"""
Pydantic models for the mortgage evaluation domain.

All models are strict about types to surface data quality issues early.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class LoanType(str, Enum):
    CONVENTIONAL = "CONVENTIONAL"
    FHA = "FHA"
    VA = "VA"
    USDA = "USDA"
    JUMBO = "JUMBO"


class LoanPurpose(str, Enum):
    PURCHASE = "PURCHASE"
    REFINANCE = "REFINANCE"
    CASH_OUT_REFINANCE = "CASH_OUT_REFINANCE"
    HOME_EQUITY = "HOME_EQUITY"


class CreditDecisionOutcome(str, Enum):
    APPROVED = "APPROVED"
    APPROVED_WITH_CONDITIONS = "APPROVED_WITH_CONDITIONS"
    SUSPENDED = "SUSPENDED"
    DENIED = "DENIED"
    WITHDRAWN = "WITHDRAWN"


class DocumentType(str, Enum):
    W2 = "W2"
    PAYSTUB = "PAYSTUB"
    BANK_STATEMENT = "BANK_STATEMENT"
    TAX_RETURN = "TAX_RETURN"
    APPRAISAL = "APPRAISAL"
    CLOSING_DISCLOSURE = "CLOSING_DISCLOSURE"
    LOAN_ESTIMATE = "LOAN_ESTIMATE"
    PURCHASE_AGREEMENT = "PURCHASE_AGREEMENT"


class Borrower(BaseModel):
    """Individual borrower profile."""

    borrower_id: str
    name: str
    ssn_last4: str = Field(..., min_length=4, max_length=4)
    credit_score: int = Field(..., ge=300, le=850)
    annual_income_stated: float = Field(..., ge=0)
    employment_type: str  # W2, Self-Employed, Retired, etc.
    years_employed: float = Field(..., ge=0)

    @field_validator("ssn_last4")
    @classmethod
    def validate_ssn_last4(cls, v: str) -> str:
        if not v.isdigit():
            raise ValueError("ssn_last4 must be 4 digits")
        return v


class Property(BaseModel):
    """Subject property details."""

    address: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str
    appraised_value: float = Field(..., ge=0)
    property_type: str  # SFR, Condo, Multi-Family, etc.
    occupancy_type: str  # Primary, Second Home, Investment


class LoanApplication(BaseModel):
    """Full loan application — primary input to the evaluation harness."""

    application_id: str
    submission_date: date
    loan_type: LoanType
    loan_purpose: LoanPurpose
    loan_amount: float = Field(..., ge=10_000, le=5_000_000)
    interest_rate: float = Field(..., ge=0.001, le=0.30)
    term_months: int = Field(..., ge=60, le=480)
    primary_borrower: Borrower
    co_borrower: Optional[Borrower] = None
    property: Property

    @property
    def ltv_ratio(self) -> float:
        """Loan-to-value ratio."""
        if self.property.appraised_value == 0:
            return 0.0
        return round(self.loan_amount / self.property.appraised_value, 4)

    @property
    def dti_ratio(self) -> float:
        """Simplified debt-to-income (monthly payment / monthly income)."""
        monthly_payment = (
            self.loan_amount
            * (self.interest_rate / 12)
            / (1 - (1 + self.interest_rate / 12) ** (-self.term_months))
        )
        monthly_income = self.primary_borrower.annual_income_stated / 12
        if monthly_income == 0:
            return 1.0
        return round(monthly_payment / monthly_income, 4)


class IncomeVerificationResult(BaseModel):
    """Output from the income verification use case."""

    application_id: str
    document_type: DocumentType
    gross_income_extracted: float
    monthly_income_extracted: float
    income_period: str  # "annual", "monthly"
    discrepancy_pct: float  # % difference from stated income
    verified: bool
    notes: str = ""


class ClosingDisclosureLineItem(BaseModel):
    """One line on a Closing Disclosure."""

    description: str
    amount: float
    section: str  # A, B, C, E, F, G, H, J, K, L, M, N


class ClosingDisclosure(BaseModel):
    """Simplified Closing Disclosure (TRID form)."""

    application_id: str
    loan_amount: float
    interest_rate: float
    closing_costs: float
    cash_to_close: float
    prepaid_items: float
    line_items: list[ClosingDisclosureLineItem] = Field(default_factory=list)
    balanced: bool = False

    def compute_balance(self) -> float:
        """Return the difference: closing_costs - sum(line_items)."""
        total_items = sum(item.amount for item in self.line_items)
        return round(abs(self.closing_costs - total_items), 2)


class AppraisalReport(BaseModel):
    """Simplified appraisal report."""

    report_id: str
    application_id: str
    appraiser_name: str
    report_date: date
    appraised_value: float
    comparable_sales: list[dict[str, Any]] = Field(default_factory=list)
    condition: str  # C1-C6
    approach_used: str  # Sales Comparison, Income, Cost


class CreditDecision(BaseModel):
    """AI-generated credit decision with CFPB reason codes."""

    application_id: str
    decision: CreditDecisionOutcome
    reason_codes: list[str]  # CFPB standard reason codes
    score_used: int
    decision_date: date
    underwriter_notes: str = ""
    adverse_action_required: bool = False


class ComplianceCheck(BaseModel):
    """Result of a TRID/RESPA compliance check."""

    application_id: str
    check_name: str
    passed: bool
    finding: str
    severity: str  # INFO, WARNING, VIOLATION
    cfpb_citation: str = ""


class DocumentVersion(BaseModel):
    """Represents a versioned document in the loan file."""

    document_id: str
    application_id: str
    document_type: DocumentType
    version: int
    upload_date: date
    uploaded_by: str
    is_authoritative: bool = False
    supersedes: Optional[str] = None  # document_id of the version this replaces
