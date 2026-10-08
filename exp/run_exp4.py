"""Standalone runner for Experiment 4: Out-of-Sample Forecast Comparison."""

import sys
from pathlib import Path

# Make the project root importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_pipeline import build_panel, build_increments
from src.exp4_forecast import run_exp4, MODEL_NAMES


def main() -> None:
    print("Building panel (synthetic)…")
    panel = build_panel(force_synthetic=True)
    incs = build_increments(panel)

    print("Running Experiment 4…")
    results = run_exp4(panel=panel, incs=incs)

    print("\n=== Experiment 4 Summary (average CV RMSE) ===")
    header = f"{'Tenor':>6s}  {'Model':>10s}  {'rmse_a0':>9s}  {'rmse_a1':>9s}  {'rmse_a2':>9s}  {'rmse_smile':>11s}"
    print(header)
    print("-" * len(header))
    for tenor, tenor_res in results.items():
        for model, scores in tenor_res.items():
            print(
                f"  {tenor:>4s}  {model:>10s}  "
                f"{scores['rmse_a0']:9.6f}  "
                f"{scores['rmse_a1']:9.6f}  "
                f"{scores['rmse_a2']:9.6f}  "
                f"{scores['rmse_smile']:11.6f}"
            )

    print("\nResults saved to /workspace/project/results/")


if __name__ == "__main__":
    main()
