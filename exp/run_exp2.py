#!/usr/bin/env python3
"""Runner script for Experiment 2: ATM Jet Hierarchy."""

import os
import sys

# Ensure project root is on the path when script is invoked directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)

from src.data_pipeline import build_panel, build_increments
from src.exp2_jets import run_exp2

panel = build_panel()
incs = build_increments(panel)
results = run_exp2(panel=panel, incs=incs)

print("\nExperiment 2 – ATM Jet Hierarchy Results")
print(
    f"{'Tenor':>6}  {'beta':>7}  {'SE':>6}  {'eta':>7}  {'SE':>6}  "
    f"{'zeta':>7}  {'SE':>6}  {'Wald':>8}  {'p-val':>7}  {'N':>6}"
)
print("-" * 80)
for tenor in results["estimates"]:
    e = results["estimates"][tenor]
    w = results["wald"][tenor]
    print(
        f"{tenor:>6}  {e['beta']:7.4f}  {e['beta_se']:6.4f}  "
        f"{e['eta']:7.4f}  {e['eta_se']:6.4f}  "
        f"{e['zeta']:7.4f}  {e['zeta_se']:6.4f}  "
        f"{w['wald_stat']:8.2f}  {w['wald_pvalue']:7.4f}  {e['n_obs']:6d}"
    )

print("\nResults saved to results/exp2_jets_table.csv")
print("Plot saved to results/exp2_jets_plot.png")
