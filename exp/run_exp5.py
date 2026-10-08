"""
Standalone runner for Experiment 5: Transport-versus-Source Decomposition.

Usage:
    python -m exp.run_exp5
    # or from the project root:
    python exp/run_exp5.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_pipeline import build_panel, build_increments
from src.exp5_source import run_exp5


def main() -> None:
    print("=" * 60)
    print("Experiment 5: Transport-versus-Source Decomposition")
    print("=" * 60)

    print("Building panel and increments…")
    panel = build_panel(force_synthetic=True)
    incs = build_increments(panel)

    print("Running Exp 5…")
    results = run_exp5(panel=panel, incs=incs)

    print("\n--- Fixed-node results ---")
    print(results["fixednode_results"].to_string(index=False))

    print("\n--- Tracked-minimum results ---")
    print(results["tracked_min_results"].to_string(index=False))

    print(f"\nSaved: {results['fixednode_table_path']}")
    print(f"Saved: {results['tracked_min_table_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
