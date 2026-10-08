"""
Standalone runner for Experiment 7: Numerical PDE Transport Illustration.

Usage:
    python -m exp.run_exp7
    # or from the project root:
    python exp/run_exp7.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exp7_pde import run_exp7


def main() -> None:
    print("=" * 60)
    print("Experiment 7: Numerical PDE Transport Illustration")
    print("=" * 60)

    print("Running convergence study…")
    results = run_exp7()

    print("\n--- Convergence table ---")
    print(results["convergence_table"].to_string(index=False))

    print(f"\nEstimated convergence slope (log-log): {results['convergence_slope']:.3f}")
    print(f"Numerical solution positive: {results['positivity_flag']}")
    print(f"Gatheral g non-negative: {results['positivity_g_flag']}")

    print(f"\nSaved: {results['convergence_table_path']}")
    print(f"Saved: {results['smile_transport_path']}")
    print(f"Saved: {results['convergence_plot_path']}")
    print(f"Saved: {results['gatheral_g_path']}")
    print("Done.")


if __name__ == "__main__":
    main()
