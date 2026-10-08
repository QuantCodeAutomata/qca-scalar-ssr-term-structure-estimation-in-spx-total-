"""Standalone runner for Experiment 3: Strike-Dependent Local Velocity Profile."""

import sys
from pathlib import Path

# Make the project root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import build_panel, build_increments
from src.exp3_local_vel import run_exp3


def main() -> None:
    import numpy as np

    print("Building panel (synthetic)…")
    panel = build_panel(force_synthetic=True)

    print("Running Experiment 3…")
    results = run_exp3(panel=panel)

    print("\n=== Experiment 3 Summary ===")
    for tenor, res in results.items():
        wald = res["wald"]
        v = res["v_hat"]
        finite = v[np.isfinite(v)]
        v_range = f"[{finite.min():.3f}, {finite.max():.3f}]" if len(finite) else "n/a"
        print(
            f"  {tenor:>4s}  v̂ range {v_range}  "
            f"Wald={wald['wald_stat']:.2f}  p={wald['p_value']:.4f}"
        )

    print("\nResults saved to /workspace/project/results/")


if __name__ == "__main__":
    main()
