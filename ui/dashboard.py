"""
Streamlit dashboard for the mortgage AI evaluation harness.

Displays real-time evaluation metrics, confidence distribution,
routing breakdown, and a full audit log table.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when run via `streamlit run ui/dashboard.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from core.evaluator import EvaluationHarness
from mortgage.use_cases import run_all_use_cases
from core.router import RoutingDecision

st.set_page_config(
    page_title="Mortgage AI Eval Harness",
    page_icon="🏠",
    layout="wide",
)

DECISION_COLORS = {
    RoutingDecision.AUTO_APPROVE.value: "#2ECC71",
    RoutingDecision.HUMAN_REVIEW.value: "#F39C12",
    RoutingDecision.REJECT.value: "#E74C3C",
}

USE_CASE_LABELS = {
    "income_verification": "UC1: Income Verification",
    "closing_disclosure": "UC2: Closing Disclosure",
    "appraisal_comparison": "UC3: Appraisal Comparison",
    "credit_decision": "UC4: Credit Decision",
    "document_lineage": "UC5: Document Lineage",
}


@st.cache_data(show_spinner=False)
def _run_harness() -> tuple[list[dict], dict]:
    """Run all use cases once and cache results."""
    harness = EvaluationHarness(
        audit_log_path=str(ROOT / "sample_output" / "eval_report.jsonl"),
    )
    results = run_all_use_cases(harness)
    records = [r.to_dict() for r in results]
    for i, r in enumerate(results):
        records[i]["use_case_label"] = USE_CASE_LABELS.get(
            r.audit_record.use_case, r.audit_record.use_case
        )
        records[i]["loan_id"] = r.audit_record.loan_application_id
        records[i]["timestamp"] = r.audit_record.timestamp
        records[i]["record_id"] = r.audit_record.record_id
    summary = harness.summary()
    harness.export_report(ROOT / "sample_output" / "eval_report.json")
    return records, summary


def _decision_badge(decision: str) -> str:
    color = DECISION_COLORS.get(decision, "#888888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{decision}</span>'


def main() -> None:
    st.title("Mortgage AI Evaluation Harness")
    st.caption(
        "Production-grade framework for scoring AI outputs in mortgage pipelines. "
        "CFPB-compliant audit trail — zero LLM calls."
    )

    with st.spinner("Running evaluation pipeline..."):
        records, summary = _run_harness()

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------------
    # Row 1: KPI cards
    # -----------------------------------------------------------------------
    st.subheader("Pipeline Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric("Total Evaluations", summary.get("total_evaluations", 0))
    with c2:
        st.metric(
            "Auto-Approve Rate",
            f"{summary.get('auto_approve_rate', 0):.0%}",
            delta=f"{summary.get('auto_approve_count', 0)} decisions",
        )
    with c3:
        st.metric(
            "Human Review Rate",
            f"{summary.get('human_review_rate', 0):.0%}",
            delta=f"{summary.get('human_review_count', 0)} decisions",
        )
    with c4:
        st.metric(
            "Reject Rate",
            f"{summary.get('reject_rate', 0):.0%}",
            delta=f"{summary.get('reject_count', 0)} decisions",
        )
    with c5:
        st.metric(
            "Avg Confidence",
            f"{summary.get('avg_confidence', 0):.3f}",
        )
    with c6:
        st.metric(
            "Avg Faithfulness",
            f"{summary.get('avg_faithfulness', 0):.3f}",
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Row 2: Metrics bar chart + Confidence distribution
    # -----------------------------------------------------------------------
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Scores by Use Case")
        chart_df = df[
            ["use_case_label", "faithfulness", "relevance", "accuracy", "composite_score"]
        ].set_index("use_case_label")
        st.bar_chart(chart_df, height=280)

    with col_right:
        st.subheader("Confidence Distribution")
        conf_df = pd.DataFrame({
            "Confidence": df["combined_confidence"],
            "Use Case": df["use_case_label"],
        })
        # Render as a simple bar chart indexed by use case
        pivot = conf_df.set_index("Use Case")
        st.bar_chart(pivot, height=280)

    st.divider()

    # -----------------------------------------------------------------------
    # Row 3: Routing breakdown + Threshold diagram
    # -----------------------------------------------------------------------
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Routing Decisions")
        routing_counts = df["routing_decision"].value_counts().reset_index()
        routing_counts.columns = ["Decision", "Count"]
        st.dataframe(routing_counts, use_container_width=True, hide_index=True)

        # Visual routing indicator per use case
        st.markdown("**Decision per use case:**")
        for _, row in df.iterrows():
            badge = _decision_badge(row["routing_decision"])
            st.markdown(
                f"{row['use_case_label']}: {badge}",
                unsafe_allow_html=True,
            )

    with col_b:
        st.subheader("Confidence Threshold Reference")
        st.markdown(
            """
| Range | Routing | Action |
|-------|---------|--------|
| > 0.85 | AUTO_APPROVE | AI decision accepted |
| 0.60 – 0.85 | HUMAN_REVIEW | Sent to underwriter queue |
| < 0.60 | REJECT | Returned with explanation |
"""
        )
        st.info(
            "Thresholds are configurable at harness initialization. "
            "Mortgage default: high=0.85, low=0.60."
        )

        # Show intent vs transaction confidence
        st.subheader("Intent vs Transaction Confidence")
        conf_breakdown = df[
            ["use_case_label", "intent_confidence", "transaction_confidence"]
        ].set_index("use_case_label")
        st.bar_chart(conf_breakdown, height=200)

    st.divider()

    # -----------------------------------------------------------------------
    # Row 4: Full audit log
    # -----------------------------------------------------------------------
    st.subheader("Audit Log (CFPB-Defensible Trail)")
    st.caption(
        "Every record is immutable, timestamped, and written to "
        "`sample_output/eval_report.jsonl`. "
        "Each row captures: input summary, all scores, routing decision, and flags."
    )

    audit_cols = [
        "timestamp",
        "loan_id",
        "use_case_label",
        "faithfulness",
        "relevance",
        "accuracy",
        "composite_score",
        "combined_confidence",
        "routing_decision",
        "routing_reason",
        "flags",
    ]
    audit_df = df[audit_cols].copy()
    audit_df["flags"] = audit_df["flags"].apply(
        lambda f: "; ".join(f) if isinstance(f, list) else str(f)
    )
    audit_df = audit_df.rename(columns={
        "use_case_label": "Use Case",
        "loan_id": "Loan ID",
        "faithfulness": "Faith.",
        "relevance": "Relev.",
        "accuracy": "Accur.",
        "composite_score": "Composite",
        "combined_confidence": "Confidence",
        "routing_decision": "Decision",
        "routing_reason": "Reason",
    })

    def color_decision(val: str) -> str:
        color = DECISION_COLORS.get(val, "#888888")
        return f"background-color: {color}20; color: {color}; font-weight: bold"

    styled = audit_df.style.applymap(color_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=350)

    # Download button
    csv = audit_df.to_csv(index=False)
    st.download_button(
        label="Download Audit Log (CSV)",
        data=csv,
        file_name="mortgage_eval_audit.csv",
        mime="text/csv",
    )

    st.divider()
    st.caption(
        "mortgage-eval-harness | Built for Supreme Lending interview | "
        "Zero API calls | Rule-based evaluation engine"
    )


if __name__ == "__main__":
    main()
