#!/usr/bin/env python3
"""Runner script for Experiment 1: Scalar SSR Term Structure."""

import os
import sys

# Ensure project root is on the path when script is invoked directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)

from src.data_pipeline import build_panel, build_increments
from src.exp1_ssr import run_exp1

panel = build_panel()
incs = build_increments(panel)
results = run_exp1(panel=panel, incs=incs)

print("\nExperiment 1 – Full-Sample Scalar SSR Results")
print(f"{'Tenor':>6}  {'beta':>8}  {'SE(HC3)':>9}  {'t-stat':>8}  {'p-value':>9}  {'R2':>7}  {'N':>6}")
print("-" * 64)
for tenor, v in results["full_sample"].items():
    print(
        f"{tenor:>6}  {v['beta']:8.4f}  {v['se_hc3']:9.4f}  "
        f"{v['t_stat']:8.2f}  {v['p_value']:9.4f}  {v['r_squared']:7.4f}  {v['n_obs']:6d}"
    )

print("\nResults saved to results/exp1_ssr_table.csv")
print("Plots saved to results/exp1_beta_term_structure.png and results/exp1_rolling_beta.png")
