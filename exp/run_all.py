"""
Master runner: executes all 7 experiments in sequence.

Usage::

    python exp/run_all.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

from src.data_pipeline import build_panel, build_increments

os.makedirs("results", exist_ok=True)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main() -> None:
    t0 = time.time()

    _section("Building shared SPX panel and increment dataset")
    panel = build_panel(force_synthetic=True)
    incs = build_increments(panel)
    print(f"Panel shape:      {panel.shape}")
    print(f"Increments shape: {incs.shape}")

    # ------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------
    _section("Experiment 1 – Scalar SSR Term Structure")
    from src.exp1_ssr import run_exp1
    r1 = run_exp1(panel=panel, incs=incs)
    for tenor, v in r1["full_sample"].items():
        print(f"  {tenor}: beta={v['beta']:.4f}  R2={v['r_squared']:.4f}  "
              f"t={v['t_stat']:.2f}  N={v['n_obs']}")

    # ------------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------------
    _section("Experiment 2 – ATM Jet Hierarchy")
    from src.exp2_jets import run_exp2
    r2 = run_exp2(panel=panel, incs=incs)
    for tenor, v in r2["estimates"].items():
        print(f"  {tenor}: beta={v['beta']:.4f}  eta={v['eta']:.4f}  "
              f"zeta={v['zeta']:.4f}  "
              f"Wald={v.get('wald_stat', float('nan')):.1f}  "
              f"p={v.get('wald_pvalue', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # Experiment 3
    # ------------------------------------------------------------------
    _section("Experiment 3 – Strike-Dependent Local Velocity")
    from src.exp3_local_vel import run_exp3
    r3 = run_exp3(panel=panel, incs=incs)
    for tenor, v in r3["summaries"].items():
        lo, hi = v["v_range"]
        print(f"  {tenor}: v range [{lo:.3f}, {hi:.3f}]  "
              f"Wald={v['wald_stat']:.2f}  p={v['wald_pval']:.4f}")

    # ------------------------------------------------------------------
    # Experiment 4
    # ------------------------------------------------------------------
    _section("Experiment 4 – Out-of-Sample Forecast Comparison")
    from src.exp4_forecast import run_exp4
    r4 = run_exp4(panel=panel, incs=incs)
    print(r4["table"].to_string(index=False))

    # ------------------------------------------------------------------
    # Experiment 5
    # ------------------------------------------------------------------
    _section("Experiment 5 – Transport-vs-Source Decomposition")
    from src.exp5_source import run_exp5
    r5 = run_exp5(panel=panel, incs=incs)
    print(r5["fixednode_table"][["tenor", "node", "s", "t_s", "p_s"]].to_string(index=False))

    # ------------------------------------------------------------------
    # Experiment 6
    # ------------------------------------------------------------------
    _section("Experiment 6 – Finite-Sample Bootstrap Validation")
    from src.exp6_bootstrap import run_exp6
    r6 = run_exp6(panel=panel, incs=incs)
    for tenor, tbl in r6["recovery_tables"].items():
        print(f"\n  {tenor}:")
        print(tbl.to_string(index=False))
    print("\n  Power study (1M):")
    print(r6["power_table"].to_string(index=False))

    # ------------------------------------------------------------------
    # Experiment 7 (pure numerical – no market data)
    # ------------------------------------------------------------------
    _section("Experiment 7 – Numerical PDE Transport")
    from src.exp7_pde import run_exp7
    r7 = run_exp7()
    print(f"  Convergence slope: {r7['convergence_slope']:.3f}")
    print(f"  Gatheral g ≥ 0:    {r7['positivity_flag']}")

    elapsed = time.time() - t0
    _section(f"All experiments complete  [{elapsed:.1f}s]")
    print("Results saved to results/")


if __name__ == "__main__":
    main()
