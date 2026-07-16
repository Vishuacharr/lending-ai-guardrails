"""
Streamlit dashboard for the mortgage AI evaluation harness.

Tabs:
  1. KPI Dashboard   — before/after impact of introducing the harness
  2. Validation Batch — 24 anonymized loan scenarios run through the pipeline
  3. Five Use Cases   — the original narrative walkthroughs
  4. Audit Log        — full CFPB-defensible audit trail
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core.evaluator import EvaluationHarness
from core.batch_runner import run_batch
from core.kpi import compute_kpis
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

ISSUE_TYPE_LABELS = {
    "clean": "Clean (no issue)",
    "income_mismatch": "Income Mismatch",
    "document_version_conflict": "Document Version Conflict",
    "cd_imbalance": "CD Imbalance",
    "compliance_violation": "Compliance Violation",
}

CARD_CSS = """
<style>
.kpi-card {
    background: linear-gradient(135deg, rgba(46,204,113,0.08), rgba(46,204,113,0.02));
    border: 1px solid rgba(46,204,113,0.25);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 8px;
}
.kpi-card.warn {
    background: linear-gradient(135deg, rgba(243,156,18,0.08), rgba(243,156,18,0.02));
    border: 1px solid rgba(243,156,18,0.25);
}
.kpi-label {
    font-size: 13px;
    opacity: 0.75;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 34px;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 13px;
    opacity: 0.65;
    margin-top: 4px;
}
</style>
"""


@st.cache_data(show_spinner=False)
def _run_use_cases() -> tuple[list[dict], dict]:
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


@st.cache_data(show_spinner=False)
def _run_validation_batch() -> tuple[list[dict], dict]:
    harness = EvaluationHarness(
        audit_log_path=str(ROOT / "sample_output" / "batch_audit.jsonl"),
    )
    records = run_batch(harness)
    kpis = compute_kpis(records)
    batch_rows = [r.to_dict() for r in records]
    for row in batch_rows:
        row["issue_label"] = ISSUE_TYPE_LABELS.get(row["issue_type"], row["issue_type"])
        row["use_case_label"] = USE_CASE_LABELS.get(row["use_case"], row["use_case"])
    return batch_rows, kpis


def _decision_badge(decision: str) -> str:
    color = DECISION_COLORS.get(decision, "#888888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{decision}</span>'


def _kpi_card(label: str, value: str, sub: str = "", warn: bool = False) -> str:
    cls = "kpi-card warn" if warn else "kpi-card"
    return (
        f'<div class="{cls}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>'
    )


def _render_kpi_dashboard(kpis: dict) -> None:
    st.subheader("Impact: Before vs. After the Harness")
    st.caption(
        "Measured against a 24-scenario validation batch (8 clean, 16 with an injected "
        "real-world defect). 'Before' figures are illustrative manual-QC industry "
        "baselines; 'After' figures are computed directly from this run."
    )

    st.markdown(CARD_CSS, unsafe_allow_html=True)

    pt = kpis["processing_time"]
    er = kpis["error_rate"]
    cs = kpis["compliance_score"]
    sav = kpis["cost_savings"]
    routing = kpis["routing"]
    acc = kpis["routing_accuracy"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            _kpi_card(
                "Processing Time Reduction",
                f"{pt['reduction_pct']:.0%}",
                f"{pt['manual_seconds_per_loan']/60:.0f} min (manual) → "
                f"{pt['harness_seconds_per_loan']*1000:.2f} ms (harness)",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _kpi_card(
                "Error Rate Reduction",
                f"{er['reduction_pct']:.0%}",
                f"{er['manual_baseline']:.0%} baseline → {er['post_harness']:.0%} post-harness",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _kpi_card(
                "Compliance Score Uplift",
                f"+{cs['uplift']:.0%}",
                f"{cs['manual_baseline']:.0%} baseline → {cs['post_harness']:.0%} post-harness",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _kpi_card(
                "Est. Savings / Loan",
                f"${sav['savings_per_loan_usd']:.2f}",
                f"${sav['manual_cost_per_loan_usd']:.2f} → ${sav['post_harness_cost_per_loan_usd']:.2f}",
            ),
            unsafe_allow_html=True,
        )

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(
            _kpi_card(
                "Auto-Approve Rate",
                f"{routing['auto_approve_rate']:.0%}",
                f"{routing['auto_approve_count']} of {kpis['total_scenarios']} scenarios",
            ),
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            _kpi_card(
                "Human Review Rate",
                f"{routing['human_review_rate']:.0%}",
                f"{routing['human_review_count']} scenarios sent to underwriter queue",
            ),
            unsafe_allow_html=True,
        )
    with c7:
        st.markdown(
            _kpi_card(
                "Defect Catch Rate",
                f"{acc['catch_rate']:.0%}",
                f"{kpis['flawed_scenarios'] - acc['false_negatives']} of {kpis['flawed_scenarios']} injected defects caught",
            ),
            unsafe_allow_html=True,
        )
    with c8:
        st.markdown(
            _kpi_card(
                "False Positive Rate",
                f"{acc['false_positive_rate']:.0%}",
                f"{acc['false_positives']} of {kpis['clean_scenarios']} clean loans over-flagged",
                warn=acc["false_positive_rate"] > 0.15,
            ),
            unsafe_allow_html=True,
        )

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Processing Time: Manual vs. Harness (log scale)**")
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Manual Review", "AI + Harness"],
                y=[pt["manual_seconds_per_loan"], max(pt["harness_seconds_per_loan"], 0.001)],
                marker_color=["#E74C3C", "#2ECC71"],
                text=[
                    f"{pt['manual_seconds_per_loan']/60:.0f} min",
                    f"{pt['harness_seconds_per_loan']*1000:.2f} ms",
                ],
                textposition="outside",
            )
        )
        fig.update_yaxes(type="log", title="Seconds (log scale)")
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("**Error Rate & Compliance Score: Before vs. After**")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Before (manual)",
            x=["Error Rate", "Compliance Score"],
            y=[er["manual_baseline"], cs["manual_baseline"]],
            marker_color="#E74C3C",
        ))
        fig2.add_trace(go.Bar(
            name="After (harness)",
            x=["Error Rate", "Compliance Score"],
            y=[er["post_harness"], cs["post_harness"]],
            marker_color="#2ECC71",
        ))
        fig2.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Routing Breakdown**")
        pie = px.pie(
            names=["Auto-Approve", "Human Review", "Reject"],
            values=[routing["auto_approve_count"], routing["human_review_count"], routing["reject_count"]],
            color=["Auto-Approve", "Human Review", "Reject"],
            color_discrete_map={
                "Auto-Approve": "#2ECC71",
                "Human Review": "#F39C12",
                "Reject": "#E74C3C",
            },
        )
        pie.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(pie, use_container_width=True)

    with col_b:
        st.markdown("**Routing Accuracy on Labeled Validation Set**")
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Bar(
            x=["Catch Rate<br>(defects flagged)", "Clean Pass Rate<br>(clean auto-approved)"],
            y=[acc["catch_rate"], acc["clean_pass_rate"]],
            marker_color=["#2ECC71", "#3498DB"],
            text=[f"{acc['catch_rate']:.0%}", f"{acc['clean_pass_rate']:.0%}"],
            textposition="outside",
        ))
        acc_fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
        acc_fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(acc_fig, use_container_width=True)

    st.divider()
    st.markdown("**Projected Cost Savings**")
    p1, p2, p3 = st.columns(3)
    p1.metric("Savings per Loan", f"${sav['savings_per_loan_usd']:.2f}")
    p2.metric(
        f"Projected Monthly Savings (@{sav['assumed_monthly_volume']} loans/mo)",
        f"${sav['projected_monthly_savings_usd']:,.0f}",
    )
    p3.metric("Projected Annual Savings", f"${sav['projected_annual_savings_usd']:,.0f}")
    st.caption(
        "Monthly loan volume is an illustrative assumption for projection purposes only — "
        "substitute your actual origination volume for a production estimate."
    )


def _render_validation_batch(batch_rows: list[dict]) -> None:
    st.subheader("Validation Batch — 24 Anonymized Sample Loan Records")
    st.caption(
        "Synthetic names, SSNs, and addresses. Covers clean loans plus four categories "
        "of real-world defects: income mismatches, document version conflicts, Closing "
        "Disclosure imbalances, and compliance violations."
    )

    df = pd.DataFrame(batch_rows)

    issue_counts = df["issue_label"].value_counts().reset_index()
    issue_counts.columns = ["Issue Type", "Count"]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(issue_counts, use_container_width=True, hide_index=True)
    with col2:
        fig = px.bar(
            df.groupby(["issue_label", "routing_decision"]).size().reset_index(name="count"),
            x="issue_label",
            y="count",
            color="routing_decision",
            color_discrete_map=DECISION_COLORS,
            labels={"issue_label": "Issue Type", "count": "Scenarios", "routing_decision": "Decision"},
        )
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Full Batch Results**")
    display_cols = [
        "scenario_id", "loan_id", "use_case_label", "issue_label",
        "composite_score", "combined_confidence", "routing_decision", "elapsed_seconds",
    ]
    batch_df = df[display_cols].rename(columns={
        "scenario_id": "Scenario",
        "loan_id": "Loan ID",
        "use_case_label": "Use Case",
        "issue_label": "Issue Type",
        "composite_score": "Composite",
        "combined_confidence": "Confidence",
        "routing_decision": "Decision",
        "elapsed_seconds": "Time (s)",
    })

    def color_decision(val: str) -> str:
        color = DECISION_COLORS.get(val, "#888888")
        return f"background-color: {color}20; color: {color}; font-weight: bold"

    styled = batch_df.style.map(color_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=400)


def _render_use_cases(records: list[dict], summary: dict) -> None:
    st.subheader("Five End-to-End Use Cases")
    df = pd.DataFrame(records)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Evaluations", summary.get("total_evaluations", 0))
    with c2:
        st.metric("Auto-Approve Rate", f"{summary.get('auto_approve_rate', 0):.0%}")
    with c3:
        st.metric("Human Review Rate", f"{summary.get('human_review_rate', 0):.0%}")
    with c4:
        st.metric("Reject Rate", f"{summary.get('reject_rate', 0):.0%}")
    with c5:
        st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.3f}")
    with c6:
        st.metric("Avg Faithfulness", f"{summary.get('avg_faithfulness', 0):.3f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Scores by Use Case**")
        chart_df = df[
            ["use_case_label", "faithfulness", "relevance", "accuracy", "composite_score"]
        ].set_index("use_case_label")
        st.bar_chart(chart_df, height=280)

    with col_right:
        st.markdown("**Confidence Distribution**")
        conf_df = pd.DataFrame({
            "Confidence": df["combined_confidence"],
            "Use Case": df["use_case_label"],
        }).set_index("Use Case")
        st.bar_chart(conf_df, height=280)

    st.markdown("**Decision per use case:**")
    for _, row in df.iterrows():
        badge = _decision_badge(row["routing_decision"])
        st.markdown(f"{row['use_case_label']}: {badge}", unsafe_allow_html=True)


def _render_audit_log(use_case_records: list[dict], batch_rows: list[dict]) -> None:
    st.subheader("Audit Log (CFPB-Defensible Trail)")
    st.caption(
        "Every record is immutable, timestamped, and written to JSONL. "
        "Each row captures: input summary, all scores, routing decision, and flags."
    )

    all_rows = use_case_records + batch_rows
    df = pd.DataFrame(all_rows)

    audit_cols = [
        "loan_id", "use_case_label", "faithfulness", "relevance", "accuracy",
        "composite_score", "combined_confidence", "routing_decision", "routing_reason", "flags",
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

    styled = audit_df.style.map(color_decision, subset=["Decision"])
    st.dataframe(styled, use_container_width=True, height=450)

    csv = audit_df.to_csv(index=False)
    st.download_button(
        label="Download Full Audit Log (CSV)",
        data=csv,
        file_name="mortgage_eval_audit.csv",
        mime="text/csv",
    )


def main() -> None:
    st.title("Mortgage AI Evaluation Harness")
    st.caption(
        "Production-grade framework for scoring AI outputs in mortgage pipelines. "
        "CFPB-compliant audit trail — zero LLM calls, fully deterministic."
    )

    with st.spinner("Running evaluation pipeline..."):
        use_case_records, use_case_summary = _run_use_cases()
        batch_rows, kpis = _run_validation_batch()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 KPI Dashboard", "🧪 Validation Batch (24)", "📋 Five Use Cases", "📜 Audit Log"]
    )

    with tab1:
        _render_kpi_dashboard(kpis)
    with tab2:
        _render_validation_batch(batch_rows)
    with tab3:
        _render_use_cases(use_case_records, use_case_summary)
    with tab4:
        _render_audit_log(use_case_records, batch_rows)

    st.divider()
    st.caption(
        "lending-ai-guardrails | CFPB-compliant audit trail | "
        "Zero API calls | Rule-based evaluation engine"
    )


if __name__ == "__main__":
    main()
