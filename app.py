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


if __name__ == "__main__" and "--report" in sys.argv:
    run_headless_report()
elif __name__ == "__main__":
    # When invoked as `streamlit run app.py`, streamlit handles __main__
    # This branch handles direct `python app.py` — show usage hint
    print("Usage:")
    print("  streamlit run app.py          # Launch dashboard")
    print("  python app.py --report        # Headless CLI report")
