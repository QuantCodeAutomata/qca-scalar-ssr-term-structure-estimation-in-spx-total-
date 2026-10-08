"""
Experiment 1: Scalar SSR Term Structure.

Estimates the scalar speed-of-reversion (SSR) beta for each tenor by
regressing daily ATM variance increments on the skew-times-underlying
increment term, without intercept, using HC3 robust standard errors.

Regression per tenor:
    Delta_a0 = beta * (a1 * Delta_u) + epsilon

A 60-day rolling version tracks how beta evolves over time.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.data_pipeline import (
    TENOR_LABELS,
    build_increments,
    build_panel,
    get_tenor_increments,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def estimate_scalar_ssr(df: pd.DataFrame, tenor: str) -> dict:
    """Estimate scalar SSR beta without intercept using HC3 robust SEs.

    Regression: Delta_a0 = beta * (a1 * Delta_u) + epsilon

    Uses statsmodels OLS with HC3 robust covariance.

    Parameters
    ----------
    df    : single-tenor increment DataFrame (output of get_tenor_increments)
    tenor : tenor label, e.g. '1M'

    Returns
    -------
    dict with keys: tenor, beta, se_hc3, t_stat, p_value, r_squared, n_obs
    """
    df = df.dropna(subset=["Delta_a0", "a1", "Delta_u"])
    y = df["Delta_a0"].to_numpy(dtype=float)
    X = (df["a1"] * df["Delta_u"]).to_numpy(dtype=float).reshape(-1, 1)

    model = sm.OLS(y, X)
    res = model.fit(cov_type="HC3")

    return {
        "tenor": tenor,
        "beta": float(res.params[0]),
        "se_hc3": float(res.bse[0]),
        "t_stat": float(res.tvalues[0]),
        "p_value": float(res.pvalues[0]),
        "r_squared": float(res.rsquared),
        "n_obs": int(res.nobs),
    }


def estimate_rolling_ssr(
    df: pd.DataFrame,
    tenor: str,
    window: int = 60,
) -> pd.DataFrame:
    """60-day rolling beta estimates.

    Skips windows with fewer than 20 usable increments.

    Parameters
    ----------
    df     : single-tenor increment DataFrame (output of get_tenor_increments)
    tenor  : tenor label
    window : rolling window size in observations

    Returns
    -------
    DataFrame indexed by date with columns [beta, n_obs]
    """
    df = df.dropna(subset=["Delta_a0", "a1", "Delta_u"]).copy()
    df = df.sort_index()

    n = len(df)
    results = []

    for end in range(window, n + 1):
        block = df.iloc[end - window : end]
        y = block["Delta_a0"].to_numpy(dtype=float)
        X = (block["a1"] * block["Delta_u"]).to_numpy(dtype=float).reshape(-1, 1)

        valid = np.isfinite(y) & np.isfinite(X[:, 0])
        n_valid = int(valid.sum())
        if n_valid < 20:
            continue

        y_v, X_v = y[valid], X[valid]
        try:
            res = sm.OLS(y_v, X_v).fit(cov_type="HC3")
            beta = float(res.params[0])
        except Exception:
            continue

        results.append({
            "date": block.index[-1],
            "beta": beta,
            "n_obs": n_valid,
        })

    if not results:
        return pd.DataFrame(columns=["beta", "n_obs"])

    out = pd.DataFrame(results).set_index("date")
    out.index = pd.to_datetime(out.index)
    return out


def run_exp1(
    panel: Optional[pd.DataFrame] = None,
    incs: Optional[pd.DataFrame] = None,
) -> dict:
    """Run full Experiment 1 for all tenors.

    Estimates full-sample and 60-day rolling scalar SSR beta for every tenor.
    Saves a summary CSV and two figures to /workspace/project/results/.

    Parameters
    ----------
    panel : pre-built panel (built if None)
    incs  : pre-built increments (built from panel if None)

    Returns
    -------
    dict with keys:
        'full_sample' : {tenor: estimate_dict, ...}
        'rolling'     : {tenor: rolling_DataFrame, ...}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if panel is None:
        panel = build_panel()
    if incs is None:
        incs = build_increments(panel)

    full_sample: dict = {}
    rolling: dict = {}

    for tenor in TENOR_LABELS:
        df = get_tenor_increments(incs, tenor)
        if df.empty:
            continue
        full_sample[tenor] = estimate_scalar_ssr(df, tenor)
        rolling[tenor] = estimate_rolling_ssr(df, tenor)

    # ------------------------------------------------------------------ #
    # Save full-sample table                                               #
    # ------------------------------------------------------------------ #
    rows = list(full_sample.values())
    table = pd.DataFrame(rows).set_index("tenor")
    table = table.reindex(TENOR_LABELS)
    table.to_csv(os.path.join(RESULTS_DIR, "exp1_ssr_table.csv"))

    # ------------------------------------------------------------------ #
    # Plot 1: beta term structure                                          #
    # ------------------------------------------------------------------ #
    tenors_present = [t for t in TENOR_LABELS if t in full_sample]
    betas = [full_sample[t]["beta"] for t in tenors_present]
    ses = [full_sample[t]["se_hc3"] for t in tenors_present]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tenors_present))
    ax.errorbar(
        x, betas, yerr=[1.96 * s for s in ses],
        fmt="o-", color="steelblue", ecolor="lightblue",
        elinewidth=2, capsize=4, linewidth=2, markersize=7,
        label="beta ± 1.96 SE (HC3)",
    )
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tenors_present)
    ax.set_xlabel("Tenor")
    ax.set_ylabel(r"$\hat\beta$")
    ax.set_title("Experiment 1 – Scalar SSR Beta Term Structure")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_beta_term_structure.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # Plot 2: rolling beta for all tenors                                  #
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TENOR_LABELS)))
    for i, tenor in enumerate(TENOR_LABELS):
        if tenor not in rolling or rolling[tenor].empty:
            continue
        roll = rolling[tenor]
        ax.plot(roll.index, roll["beta"], label=tenor, color=colors[i], linewidth=1.2)

    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Rolling $\hat\beta$ (60-day window)")
    ax.set_title("Experiment 1 – Rolling Scalar SSR Beta")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_rolling_beta.png"), dpi=150)
    plt.close(fig)

    return {"full_sample": full_sample, "rolling": rolling}
