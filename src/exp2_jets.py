"""
Experiment 2: ATM Jet Hierarchy.

Estimates the full jet-transport coefficient triple (beta, eta, zeta) for
each tenor using a sequential forward-substitution procedure:

    Stage 1:  Delta_a0 = beta  * a1 * Delta_u + eps0
    Stage 2:  Y1       = eta   * a1 * Delta_u + eps1
              Y1 = Delta_a1 - beta_hat * a2 * Delta_u
    Stage 3:  Y2       = zeta  * a1 * Delta_u + eps2
              Y2 = Delta_a2 - beta_hat * a3 * Delta_u - 2*eta_hat * a2 * Delta_u

Standard errors are obtained via pairs (row-resampling) bootstrap with B=500
draws.  A 2-df Wald test of H0: eta=0 AND zeta=0 is also reported.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm

from src.data_pipeline import (
    TENOR_LABELS,
    build_increments,
    build_panel,
    get_tenor_increments,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def _ols_no_intercept(y: np.ndarray, x: np.ndarray) -> float:
    """OLS slope without intercept: beta = (x'x)^{-1} x'y."""
    denom = float(x @ x)
    if abs(denom) < 1e-14:
        return np.nan
    return float(x @ y) / denom


def estimate_jets(df: pd.DataFrame) -> dict:
    """Sequential forward-substitution estimator for beta, eta, zeta.

    Stage 1:  Delta_a0 = beta * a1 * Delta_u + eps0
    Stage 2:  Y1 = Delta_a1 - beta_hat*a2*Delta_u;  Y1 = eta*a1*Delta_u + eps1
    Stage 3:  Y2 = Delta_a2 - beta_hat*a3*Delta_u - 2*eta_hat*a2*Delta_u;
              Y2 = zeta*a1*Delta_u + eps2

    Parameters
    ----------
    df : single-tenor increment DataFrame (output of get_tenor_increments)

    Returns
    -------
    dict with keys: beta, eta, zeta, beta_se, eta_se, zeta_se, n_obs
        (SE entries are NaN until bootstrap is called separately)
    """
    df = df.dropna(subset=["Delta_a0", "Delta_a1", "Delta_a2", "a1", "a2", "a3", "Delta_u"])
    n = len(df)

    regressor = (df["a1"] * df["Delta_u"]).to_numpy(dtype=float)

    # Stage 1
    y0 = df["Delta_a0"].to_numpy(dtype=float)
    X0 = regressor.reshape(-1, 1)
    beta_hat = float(sm.OLS(y0, X0).fit().params[0])

    # Stage 2
    y1 = (df["Delta_a1"] - beta_hat * df["a2"] * df["Delta_u"]).to_numpy(dtype=float)
    X1 = regressor.reshape(-1, 1)
    eta_hat = float(sm.OLS(y1, X1).fit().params[0])

    # Stage 3
    y2 = (
        df["Delta_a2"]
        - beta_hat * df["a3"] * df["Delta_u"]
        - 2.0 * eta_hat * df["a2"] * df["Delta_u"]
    ).to_numpy(dtype=float)
    X2 = regressor.reshape(-1, 1)
    zeta_hat = float(sm.OLS(y2, X2).fit().params[0])

    return {
        "beta": beta_hat,
        "eta": eta_hat,
        "zeta": zeta_hat,
        "beta_se": np.nan,
        "eta_se": np.nan,
        "zeta_se": np.nan,
        "n_obs": n,
    }


# ---------------------------------------------------------------------------
# Pairs bootstrap
# ---------------------------------------------------------------------------

def pairs_bootstrap_jets(
    df: pd.DataFrame,
    B: int = 500,
    seed: int = 42,
) -> dict:
    """Pairs (row-resampling) bootstrap for jet coefficients.

    Resamples rows of df with replacement B times, re-runs the sequential
    estimator each time, and reports bootstrap standard errors and the
    bootstrap covariance matrix of [eta, zeta].

    Also computes a 2-df Wald test: H0: eta = 0 AND zeta = 0.
        W = theta' Sigma^{-1} theta,   theta = [eta_hat, zeta_hat]
    under H0: W ~ chi^2(2).

    Parameters
    ----------
    df   : single-tenor increment DataFrame
    B    : number of bootstrap replications
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        se_beta, se_eta, se_zeta  : bootstrap standard errors
        cov_eta_zeta              : 2x2 np.ndarray – bootstrap cov of [eta, zeta]
        wald_stat, wald_pvalue    : Wald test statistic and chi-sq p-value
    """
    df = df.dropna(subset=["Delta_a0", "Delta_a1", "Delta_a2", "a1", "a2", "a3", "Delta_u"])
    n = len(df)
    rng = np.random.default_rng(seed)

    # Point estimates on full data (plug-in)
    est = estimate_jets(df)
    beta_hat = est["beta"]
    eta_hat = est["eta"]
    zeta_hat = est["zeta"]

    boot_betas: list[float] = []
    boot_etas: list[float] = []
    boot_zetas: list[float] = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        b_df = df.iloc[idx]

        reg = (b_df["a1"] * b_df["Delta_u"]).to_numpy(dtype=float)

        # Stage 1
        y0 = b_df["Delta_a0"].to_numpy(dtype=float)
        b_beta = _ols_no_intercept(y0, reg)
        if not np.isfinite(b_beta):
            continue

        # Stage 2
        y1 = (b_df["Delta_a1"] - b_beta * b_df["a2"] * b_df["Delta_u"]).to_numpy(dtype=float)
        b_eta = _ols_no_intercept(y1, reg)
        if not np.isfinite(b_eta):
            continue

        # Stage 3
        y2 = (
            b_df["Delta_a2"]
            - b_beta * b_df["a3"] * b_df["Delta_u"]
            - 2.0 * b_eta * b_df["a2"] * b_df["Delta_u"]
        ).to_numpy(dtype=float)
        b_zeta = _ols_no_intercept(y2, reg)
        if not np.isfinite(b_zeta):
            continue

        boot_betas.append(b_beta)
        boot_etas.append(b_eta)
        boot_zetas.append(b_zeta)

    boot_betas_arr = np.array(boot_betas)
    boot_etas_arr = np.array(boot_etas)
    boot_zetas_arr = np.array(boot_zetas)

    se_beta = float(np.std(boot_betas_arr, ddof=1)) if len(boot_betas_arr) > 1 else np.nan
    se_eta = float(np.std(boot_etas_arr, ddof=1)) if len(boot_etas_arr) > 1 else np.nan
    se_zeta = float(np.std(boot_zetas_arr, ddof=1)) if len(boot_zetas_arr) > 1 else np.nan

    # Bootstrap covariance of [eta, zeta]
    if len(boot_etas_arr) > 1:
        ez_stack = np.column_stack([boot_etas_arr, boot_zetas_arr])
        cov_eta_zeta = np.cov(ez_stack, rowvar=False)
    else:
        cov_eta_zeta = np.full((2, 2), np.nan)

    # Wald test: W = theta' Sigma^{-1} theta, chi^2(2)
    theta = np.array([eta_hat, zeta_hat])
    try:
        Sigma_inv = np.linalg.pinv(cov_eta_zeta)
        wald_stat = float(theta @ Sigma_inv @ theta)
    except Exception:
        wald_stat = np.nan

    if np.isfinite(wald_stat):
        wald_pvalue = float(scipy.stats.chi2(2).sf(wald_stat))
    else:
        wald_pvalue = np.nan

    return {
        "se_beta": se_beta,
        "se_eta": se_eta,
        "se_zeta": se_zeta,
        "cov_eta_zeta": cov_eta_zeta,
        "wald_stat": wald_stat,
        "wald_pvalue": wald_pvalue,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_exp2(
    panel: Optional[pd.DataFrame] = None,
    incs: Optional[pd.DataFrame] = None,
) -> dict:
    """Run full Experiment 2 for all tenors.

    For each tenor: estimates (beta, eta, zeta) via sequential substitution,
    computes bootstrap SEs, and performs the joint Wald test on (eta, zeta).

    Saves:
        /workspace/project/results/exp2_jets_table.csv
        /workspace/project/results/exp2_jets_plot.png

    Parameters
    ----------
    panel : pre-built panel (built if None)
    incs  : pre-built increments (built from panel if None)

    Returns
    -------
    dict with keys:
        'estimates' : {tenor: {beta, eta, zeta, beta_se, eta_se, zeta_se, n_obs}}
        'wald'      : {tenor: {wald_stat, wald_pvalue}}
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if panel is None:
        panel = build_panel()
    if incs is None:
        incs = build_increments(panel)

    estimates: dict = {}
    wald_results: dict = {}

    for tenor in TENOR_LABELS:
        df = get_tenor_increments(incs, tenor)
        if df.empty:
            continue

        est = estimate_jets(df)
        boot = pairs_bootstrap_jets(df)

        est["beta_se"] = boot["se_beta"]
        est["eta_se"] = boot["se_eta"]
        est["zeta_se"] = boot["se_zeta"]

        estimates[tenor] = est
        wald_results[tenor] = {
            "wald_stat": boot["wald_stat"],
            "wald_pvalue": boot["wald_pvalue"],
        }

    # ------------------------------------------------------------------ #
    # Save results table                                                   #
    # ------------------------------------------------------------------ #
    rows = []
    for tenor in TENOR_LABELS:
        if tenor not in estimates:
            continue
        e = estimates[tenor]
        w = wald_results[tenor]
        rows.append({
            "tenor": tenor,
            "beta": e["beta"],
            "beta_se": e["beta_se"],
            "eta": e["eta"],
            "eta_se": e["eta_se"],
            "zeta": e["zeta"],
            "zeta_se": e["zeta_se"],
            "n_obs": e["n_obs"],
            "wald_stat": w["wald_stat"],
            "wald_pvalue": w["wald_pvalue"],
        })
    table = pd.DataFrame(rows).set_index("tenor")
    table.to_csv(os.path.join(RESULTS_DIR, "exp2_jets_table.csv"))

    # ------------------------------------------------------------------ #
    # Plot: jet coefficients across tenors                                 #
    # ------------------------------------------------------------------ #
    tenors_present = [t for t in TENOR_LABELS if t in estimates]
    x = np.arange(len(tenors_present))
    width = 0.28

    betas = [estimates[t]["beta"] for t in tenors_present]
    etas = [estimates[t]["eta"] for t in tenors_present]
    zetas = [estimates[t]["zeta"] for t in tenors_present]
    beta_ses = [estimates[t]["beta_se"] for t in tenors_present]
    eta_ses = [estimates[t]["eta_se"] for t in tenors_present]
    zeta_ses = [estimates[t]["zeta_se"] for t in tenors_present]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars_b = ax.bar(x - width, betas, width, label=r"$\hat\beta$",
                    color="steelblue", yerr=[1.96 * s for s in beta_ses],
                    capsize=3, error_kw={"elinewidth": 1.2})
    bars_e = ax.bar(x, etas, width, label=r"$\hat\eta$",
                    color="darkorange", yerr=[1.96 * s for s in eta_ses],
                    capsize=3, error_kw={"elinewidth": 1.2})
    bars_z = ax.bar(x + width, zetas, width, label=r"$\hat\zeta$",
                    color="seagreen", yerr=[1.96 * s for s in zeta_ses],
                    capsize=3, error_kw={"elinewidth": 1.2})

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tenors_present)
    ax.set_xlabel("Tenor")
    ax.set_ylabel("Coefficient estimate")
    ax.set_title("Experiment 2 – ATM Jet Hierarchy Coefficients (±1.96 bootstrap SE)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_jets_plot.png"), dpi=150)
    plt.close(fig)

    return {"estimates": estimates, "wald": wald_results}
