"""
Standalone runner for Experiment 6: Finite-Sample Bootstrap Validation.

Usage:
    python -m exp.run_exp6
    # or from the project root:
    python exp/run_exp6.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import build_panel, build_increments
from src.exp6_bootstrap import run_exp6


def main() -> None:
    print("=" * 60)
    print("Experiment 6: Finite-Sample Bootstrap Validation")
    print("=" * 60)

    print("Building panel and increments…")
    panel = build_panel(force_synthetic=True)
    incs = build_increments(panel)

    print("Running Exp 6 (this may take several minutes)…")
    results = run_exp6(panel=panel, incs=incs)

    print("\n--- Parameter-recovery summary ---")
    for tenor, tbl in results["recovery_tables"].items():
        print(f"\nTenor: {tenor}")
        print(tbl.to_string(index=False))

    print("\n--- Power study (1M) ---")
    print(results["power_table"].to_string(index=False))

    print(f"\nSaved: {results['recovery_table_path']}")
    print(f"Saved: {results['power_table_path']}")
    print(f"Saved: {results['power_curve_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
