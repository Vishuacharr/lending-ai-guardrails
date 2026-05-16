"""
Realistic sample loan applications and AI outputs for testing.

All data is synthetic — names, SSNs, and addresses are fictional.
"""

from __future__ import annotations

from datetime import date

from mortgage.schemas import (
    AppraisalReport,
    Borrower,
    ClosingDisclosure,
    ClosingDisclosureLineItem,
    CreditDecision,
    CreditDecisionOutcome,
    DocumentType,
    DocumentVersion,
    LoanApplication,
    LoanPurpose,
    LoanType,
    Property,
)

# ---------------------------------------------------------------------------
# Borrowers
# ---------------------------------------------------------------------------

BORROWER_HIGH_INCOME = Borrower(
    borrower_id="B-001",
    name="Jennifer Martinez",
    ssn_last4="4821",
    credit_score=760,
    annual_income_stated=145_000.0,
    employment_type="W2",
    years_employed=6.5,
)

BORROWER_SELF_EMPLOYED = Borrower(
    borrower_id="B-002",
    name="David Chen",
    ssn_last4="3309",
    credit_score=720,
    annual_income_stated=210_000.0,
    employment_type="Self-Employed",
    years_employed=8.0,
)

BORROWER_MARGINAL_CREDIT = Borrower(
    borrower_id="B-003",
    name="Aisha Thompson",
    ssn_last4="7756",
    credit_score=615,
    annual_income_stated=68_000.0,
    employment_type="W2",
    years_employed=2.0,
)

BORROWER_HIGH_DTI = Borrower(
    borrower_id="B-004",
    name="Robert Patel",
    ssn_last4="1122",
    credit_score=680,
    annual_income_stated=55_000.0,
    employment_type="W2",
    years_employed=4.0,
)

BORROWER_VA_ELIGIBLE = Borrower(
    borrower_id="B-005",
    name="Sarah Kowalski",
    ssn_last4="9988",
    credit_score=700,
    annual_income_stated=95_000.0,
    employment_type="W2",
    years_employed=3.5,
)

BORROWER_REFI = Borrower(
    borrower_id="B-006",
    name="Michael Johnson",
    ssn_last4="6643",
    credit_score=745,
    annual_income_stated=130_000.0,
    employment_type="W2",
    years_employed=12.0,
)

BORROWER_JUMBO = Borrower(
    borrower_id="B-007",
    name="Elizabeth Wong",
    ssn_last4="2277",
    credit_score=790,
    annual_income_stated=450_000.0,
    employment_type="W2",
    years_employed=15.0,
)

# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

PROP_SUBURBAN = Property(
    address="412 Maple Drive",
    city="Springfield",
    state="IL",
    zip_code="62701",
    appraised_value=375_000.0,
    property_type="SFR",
    occupancy_type="Primary",
)

PROP_CONDO = Property(
    address="88 Harbor View, Unit 4C",
    city="Boston",
    state="MA",
    zip_code="02110",
    appraised_value=620_000.0,
    property_type="Condo",
    occupancy_type="Primary",
)

PROP_INVESTMENT = Property(
    address="1901 Oak Street",
    city="Austin",
    state="TX",
    zip_code="78701",
    appraised_value=290_000.0,
    property_type="SFR",
    occupancy_type="Investment",
)

PROP_LUXURY = Property(
    address="5500 Lakefront Blvd",
    city="Chicago",
    state="IL",
    zip_code="60611",
    appraised_value=1_850_000.0,
    property_type="SFR",
    occupancy_type="Primary",
)

# ---------------------------------------------------------------------------
# Loan Applications (10+ samples)
# ---------------------------------------------------------------------------

APPLICATIONS: dict[str, LoanApplication] = {
    "LA-001": LoanApplication(
        application_id="LA-001",
        submission_date=date(2026, 1, 10),
        loan_type=LoanType.CONVENTIONAL,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=300_000.0,
        interest_rate=0.0685,
        term_months=360,
        primary_borrower=BORROWER_HIGH_INCOME,
        property=PROP_SUBURBAN,
    ),
    "LA-002": LoanApplication(
        application_id="LA-002",
        submission_date=date(2026, 1, 15),
        loan_type=LoanType.FHA,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=285_000.0,
        interest_rate=0.0720,
        term_months=360,
        primary_borrower=BORROWER_MARGINAL_CREDIT,
        property=PROP_SUBURBAN,
    ),
    "LA-003": LoanApplication(
        application_id="LA-003",
        submission_date=date(2026, 2, 1),
        loan_type=LoanType.VA,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=420_000.0,
        interest_rate=0.0650,
        term_months=360,
        primary_borrower=BORROWER_VA_ELIGIBLE,
        property=PROP_CONDO,
    ),
    "LA-004": LoanApplication(
        application_id="LA-004",
        submission_date=date(2026, 2, 14),
        loan_type=LoanType.CONVENTIONAL,
        loan_purpose=LoanPurpose.REFINANCE,
        loan_amount=240_000.0,
        interest_rate=0.0670,
        term_months=360,
        primary_borrower=BORROWER_REFI,
        property=PROP_SUBURBAN,
    ),
    "LA-005": LoanApplication(
        application_id="LA-005",
        submission_date=date(2026, 2, 20),
        loan_type=LoanType.JUMBO,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=1_480_000.0,
        interest_rate=0.0725,
        term_months=360,
        primary_borrower=BORROWER_JUMBO,
        property=PROP_LUXURY,
    ),
    "LA-006": LoanApplication(
        application_id="LA-006",
        submission_date=date(2026, 3, 5),
        loan_type=LoanType.CONVENTIONAL,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=260_000.0,
        interest_rate=0.0710,
        term_months=360,
        primary_borrower=BORROWER_HIGH_DTI,
        property=PROP_INVESTMENT,
    ),
    "LA-007": LoanApplication(
        application_id="LA-007",
        submission_date=date(2026, 3, 12),
        loan_type=LoanType.CONVENTIONAL,
        loan_purpose=LoanPurpose.CASH_OUT_REFINANCE,
        loan_amount=400_000.0,
        interest_rate=0.0740,
        term_months=360,
        primary_borrower=BORROWER_SELF_EMPLOYED,
        property=PROP_LUXURY,
    ),
    "LA-008": LoanApplication(
        application_id="LA-008",
        submission_date=date(2026, 3, 22),
        loan_type=LoanType.FHA,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=195_000.0,
        interest_rate=0.0730,
        term_months=360,
        primary_borrower=BORROWER_MARGINAL_CREDIT,
        property=Property(
            address="55 Pine Avenue",
            city="Memphis",
            state="TN",
            zip_code="38103",
            appraised_value=210_000.0,
            property_type="SFR",
            occupancy_type="Primary",
        ),
    ),
    "LA-009": LoanApplication(
        application_id="LA-009",
        submission_date=date(2026, 4, 2),
        loan_type=LoanType.USDA,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=180_000.0,
        interest_rate=0.0660,
        term_months=360,
        primary_borrower=Borrower(
            borrower_id="B-009",
            name="Carlos Rivera",
            ssn_last4="5544",
            credit_score=650,
            annual_income_stated=58_000.0,
            employment_type="W2",
            years_employed=3.0,
        ),
        property=Property(
            address="22 Rural Route 7",
            city="Peoria",
            state="IL",
            zip_code="61602",
            appraised_value=185_000.0,
            property_type="SFR",
            occupancy_type="Primary",
        ),
    ),
    "LA-010": LoanApplication(
        application_id="LA-010",
        submission_date=date(2026, 4, 15),
        loan_type=LoanType.CONVENTIONAL,
        loan_purpose=LoanPurpose.PURCHASE,
        loan_amount=550_000.0,
        interest_rate=0.0695,
        term_months=360,
        primary_borrower=BORROWER_SELF_EMPLOYED,
        co_borrower=BORROWER_HIGH_INCOME,
        property=PROP_CONDO,
    ),
}

# ---------------------------------------------------------------------------
# Closing Disclosures
# ---------------------------------------------------------------------------

CD_BALANCED = ClosingDisclosure(
    application_id="LA-001",
    loan_amount=300_000.0,
    interest_rate=0.0685,
    closing_costs=9_250.0,
    cash_to_close=84_250.0,
    prepaid_items=3_200.0,
    line_items=[
        ClosingDisclosureLineItem(description="Origination Fee", amount=1_500.0, section="A"),
        ClosingDisclosureLineItem(description="Appraisal Fee", amount=550.0, section="B"),
        ClosingDisclosureLineItem(description="Credit Report", amount=50.0, section="B"),
        ClosingDisclosureLineItem(description="Title Insurance", amount=1_200.0, section="C"),
        ClosingDisclosureLineItem(description="Recording Fee", amount=125.0, section="E"),
        ClosingDisclosureLineItem(description="HOI Premium", amount=1_800.0, section="F"),
        ClosingDisclosureLineItem(description="Property Tax (6 mo)", amount=2_025.0, section="G"),
        ClosingDisclosureLineItem(description="Prepaid Interest", amount=2_000.0, section="H"),
    ],
    balanced=True,
)

CD_UNBALANCED = ClosingDisclosure(
    application_id="LA-002",
    loan_amount=285_000.0,
    interest_rate=0.0720,
    closing_costs=9_000.0,
    cash_to_close=22_000.0,
    prepaid_items=3_000.0,
    line_items=[
        ClosingDisclosureLineItem(description="Origination Fee", amount=1_425.0, section="A"),
        ClosingDisclosureLineItem(description="Appraisal Fee", amount=550.0, section="B"),
        ClosingDisclosureLineItem(description="Title Insurance", amount=950.0, section="C"),
        # Missing line items — intentional error for testing
    ],
)

# ---------------------------------------------------------------------------
# Appraisals
# ---------------------------------------------------------------------------

APPRAISAL_PRIMARY = AppraisalReport(
    report_id="APR-001-A",
    application_id="LA-001",
    appraiser_name="Linda Forsythe, MAI",
    report_date=date(2026, 1, 5),
    appraised_value=375_000.0,
    comparable_sales=[
        {"address": "408 Maple Dr", "sale_price": 365_000, "date": "2025-11-15"},
        {"address": "500 Birch Ln", "sale_price": 380_000, "date": "2025-10-20"},
        {"address": "215 Elm St", "sale_price": 372_000, "date": "2025-12-01"},
    ],
    condition="C2",
    approach_used="Sales Comparison",
)

APPRAISAL_CONFLICTING = AppraisalReport(
    report_id="APR-001-B",
    application_id="LA-001",
    appraiser_name="James Rutherford, SRA",
    report_date=date(2026, 1, 8),
    appraised_value=340_000.0,  # $35k lower — triggers conflict flag
    comparable_sales=[
        {"address": "101 Spruce Ave", "sale_price": 335_000, "date": "2025-09-10"},
        {"address": "320 Cedar Rd", "sale_price": 342_000, "date": "2025-11-05"},
        {"address": "88 Willow Way", "sale_price": 345_000, "date": "2025-12-18"},
    ],
    condition="C3",
    approach_used="Sales Comparison",
)

# ---------------------------------------------------------------------------
# Credit Decisions
# ---------------------------------------------------------------------------

DECISION_APPROVED = CreditDecision(
    application_id="LA-001",
    decision=CreditDecisionOutcome.APPROVED,
    reason_codes=[],
    score_used=760,
    decision_date=date(2026, 1, 20),
    underwriter_notes="Strong file. Approving per guidelines.",
    adverse_action_required=False,
)

DECISION_DENIED_VALID = CreditDecision(
    application_id="LA-002",
    decision=CreditDecisionOutcome.DENIED,
    reason_codes=["001", "010", "006"],
    score_used=615,
    decision_date=date(2026, 2, 1),
    underwriter_notes="Derogatory history and high DTI.",
    adverse_action_required=True,
)

DECISION_DENIED_INVALID_CODES = CreditDecision(
    application_id="LA-006",
    decision=CreditDecisionOutcome.DENIED,
    reason_codes=["999", "ABC", "002"],  # 999 and ABC are not valid CFPB codes
    score_used=680,
    decision_date=date(2026, 3, 10),
    underwriter_notes="DTI exceeds guidelines.",
    adverse_action_required=True,
)

# ---------------------------------------------------------------------------
# Document Versions
# ---------------------------------------------------------------------------

DOCUMENT_VERSIONS: list[DocumentVersion] = [
    DocumentVersion(
        document_id="DOC-001-v1",
        application_id="LA-005",
        document_type=DocumentType.W2,
        version=1,
        upload_date=date(2026, 2, 15),
        uploaded_by="borrower_portal",
        is_authoritative=False,
    ),
    DocumentVersion(
        document_id="DOC-001-v2",
        application_id="LA-005",
        document_type=DocumentType.W2,
        version=2,
        upload_date=date(2026, 2, 18),
        uploaded_by="processor_jsmith",
        is_authoritative=False,
        supersedes="DOC-001-v1",
    ),
    DocumentVersion(
        document_id="DOC-001-v3",
        application_id="LA-005",
        document_type=DocumentType.W2,
        version=3,
        upload_date=date(2026, 2, 22),
        uploaded_by="underwriter_kdavis",
        is_authoritative=True,  # This is the authoritative version
        supersedes="DOC-001-v2",
    ),
]
