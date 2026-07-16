"""
Entry point: streamlit run app.py

Launches the mortgage AI evaluation dashboard.
Also supports a CLI mode for headless reporting:

    python app.py --report
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_headless_report() -> None:
    """Run all use cases and print a summary report to stdout."""
    from core.evaluator import EvaluationHarness
    from mortgage.use_cases import run_all_use_cases

    print("=" * 60)
    print("Mortgage AI Evaluation Harness — Headless Report")
    print("=" * 60)

    harness = EvaluationHarness(
        audit_log_path=ROOT / "sample_output" / "eval_report.jsonl"
    )
    results = run_all_use_cases(harness)

    USE_CASE_LABELS = [
        "UC1: Income Verification",
        "UC2: Closing Disclosure",
        "UC3: Appraisal Comparison",
        "UC4: Credit Decision",
        "UC5: Document Lineage",
    ]

    print()
    for label, result in zip(USE_CASE_LABELS, results):
        print(f"  {label}")
        print(f"    Faithfulness : {result.faithfulness:.4f}")
        print(f"    Relevance    : {result.relevance:.4f}")
        print(f"    Accuracy     : {result.accuracy:.4f}")
        print(f"    Composite    : {result.composite:.4f}")
        print(f"    Confidence   : {result.confidence.combined:.4f}")
        print(f"    Decision     : {result.route_result.decision.value}")
        print()

    summary = harness.summary()
    print("-" * 60)
    print("Summary:")
    print(json.dumps(summary, indent=2))
    print("-" * 60)

    out_path = harness.export_report(ROOT / "sample_output" / "eval_report.json")
    print(f"\nFull audit report written to: {out_path}")

    # -------------------------------------------------------------------
    # Validation batch: 24 anonymized loan scenarios + KPI computation
    # -------------------------------------------------------------------
    from core.batch_runner import run_batch
    from core.kpi import compute_kpis

    print()
    print("=" * 60)
    print("Validation Batch — 24 Anonymized Sample Loan Records")
    print("=" * 60)

    batch_harness = EvaluationHarness(
        audit_log_path=ROOT / "sample_output" / "batch_audit.jsonl"
    )
    batch_records = run_batch(batch_harness)
    for r in batch_records:
        print(
            f"  {r.scenario_id} [{r.issue_type:<26}] {r.loan_id:<8} "
            f"-> {r.result['routing_decision']:<13} "
            f"confidence={r.result['combined_confidence']:.3f}"
        )

    kpis = compute_kpis(batch_records)
    print()
    print("-" * 60)
    print("KPI Summary (before manual baseline -> after harness):")
    print(json.dumps(kpis, indent=2))
    print("-" * 60)

    batch_harness.export_report(ROOT / "sample_output" / "batch_report.json")
    print(f"\nBatch audit report written to: {ROOT / 'sample_output' / 'batch_report.json'}")


def _running_under_streamlit() -> bool:
    """Detect `streamlit run app.py` vs. plain `python app.py`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except ImportError:
        return False


if __name__ == "__main__" and _running_under_streamlit():
    from ui.dashboard import main as run_dashboard

    run_dashboard()
elif __name__ == "__main__" and "--report" in sys.argv:
    run_headless_report()
elif __name__ == "__main__":
    print("Usage:")
    print("  streamlit run app.py          # Launch dashboard")
    print("  python app.py --report        # Headless CLI report")
