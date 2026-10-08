"""
Experiment 6: Finite-Sample Bootstrap Validation.

Two sub-experiments:
    1. Parameter-recovery simulation: for tenors 1M, 3M, 6M verify that
       the OLS estimator recovers the true DGP parameters (beta, eta, zeta)
       with correct coverage using an outer S=500 / inner B=150 bootstrap.
    2. Power study: rejection rate of H0: eta=0 at 5% level as a function
       of sample size N in {50, 100, 200, 500} for the 1M tenor.

The outer loop uses a moving-block bootstrap (block_length=5) to resample
residuals while keeping the observed covariate path fixed.  The inner loop
uses a standard pairs bootstrap for SE estimation.

Usage::

    from src.exp6_bootstrap import run_exp6
    results = run_exp6()
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed

from src.data_pipeline import (
    _TRUE_BETA,
    _TRUE_ETA,
    _TRUE_ZETA,
    build_panel,
    build_increments,
    get_tenor_increments,
)

# Tenors for parameter-recovery study
_RECOVERY_TENORS = ["1M", "3M", "6M"]
_POWER_TENOR = "1M"
_POWER_SIZES = [50, 100, 200, 500]


# ---------------------------------------------------------------------------
# Sequential OLS estimator for (beta, eta, zeta)
# ---------------------------------------------------------------------------

def _fit_params(df: pd.DataFrame) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Sequentially estimate beta, eta, zeta by OLS (no intercept).

    Hierarchy equations (no-intercept regressions):
        Delta_a0 = beta  * (a1 * Delta_u)
        Delta_a1 = beta  * (a2 * Delta_u) + eta  * (a1 * Delta_u)
        Delta_a2 = beta  * (a3 * Delta_u) + 2*eta*(a2*Delta_u) + zeta*(a1*Delta_u)

    Parameters
    ----------
    df : single-tenor increment DataFrame with columns
         [a1, a2, a3, Delta_a0, Delta_a1, Delta_a2, Delta_u]

    Returns
    -------
    beta, eta, zeta : float estimates
    ses             : (3,) HAC standard errors [se_beta, se_eta, se_zeta]
    tvals           : (3,) t-values
    """
    a1 = df["a1"].values
    a2 = df["a2"].values
    a3 = df["a3"].values
    du = df["Delta_u"].values

    da0 = df["Delta_a0"].values
    da1 = df["Delta_a1"].values
    da2 = df["Delta_a2"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # --- beta from Level-0 equation ---
        x_beta = (a1 * du).reshape(-1, 1)
        res0 = sm.OLS(da0, x_beta).fit()
        hac0 = res0.get_robustcov_results(
            cov_type="HAC", maxlags=5, use_t=True
        )
        beta = float(hac0.params[0])
        se_beta = float(np.sqrt(hac0.cov_params()[0, 0]))
        t_beta = float(hac0.tvalues[0])

        # --- eta from Level-1 equation (beta known) ---
        resid1 = da1 - beta * (a2 * du)
        x_eta = (a1 * du).reshape(-1, 1)
        res1 = sm.OLS(resid1, x_eta).fit()
        hac1 = res1.get_robustcov_results(
            cov_type="HAC", maxlags=5, use_t=True
        )
        eta = float(hac1.params[0])
        se_eta = float(np.sqrt(hac1.cov_params()[0, 0]))
        t_eta = float(hac1.tvalues[0])

        # --- zeta from Level-2 equation (beta, eta known) ---
        resid2 = da2 - beta * (a3 * du) - 2 * eta * (a2 * du)
        x_zeta = (a1 * du).reshape(-1, 1)
        res2 = sm.OLS(resid2, x_zeta).fit()
        hac2 = res2.get_robustcov_results(
            cov_type="HAC", maxlags=5, use_t=True
        )
        zeta = float(hac2.params[0])
        se_zeta = float(np.sqrt(hac2.cov_params()[0, 0]))
        t_zeta = float(hac2.tvalues[0])

    ses = np.array([se_beta, se_eta, se_zeta])
    tvals = np.array([t_beta, t_eta, t_zeta])
    return beta, eta, zeta, ses, tvals


def _fit_params_simple(df: pd.DataFrame) -> tuple[float, float, float]:
    """Fast no-SE version of _fit_params for bootstrap inner loop."""
    a1 = df["a1"].values
    a2 = df["a2"].values
    a3 = df["a3"].values
    du = df["Delta_u"].values
    da0 = df["Delta_a0"].values
    da1 = df["Delta_a1"].values
    da2 = df["Delta_a2"].values

    x = (a1 * du)
    beta = float(np.dot(x, da0) / np.dot(x, x)) if np.dot(x, x) > 1e-30 else np.nan
    resid1 = da1 - beta * (a2 * du)
    eta = float(np.dot(x, resid1) / np.dot(x, x)) if np.dot(x, x) > 1e-30 else np.nan
    resid2 = da2 - beta * (a3 * du) - 2 * eta * (a2 * du)
    zeta = float(np.dot(x, resid2) / np.dot(x, x)) if np.dot(x, x) > 1e-30 else np.nan
    return beta, eta, zeta


# ---------------------------------------------------------------------------
# Residual extraction and simulation
# ---------------------------------------------------------------------------

def extract_residuals(
    df: pd.DataFrame,
    beta: float,
    eta: float,
    zeta: float,
) -> np.ndarray:
    """Compute residual triples from the three hierarchy equations.

    eps0 = Delta_a0 - beta * a1 * Delta_u
    eps1 = Delta_a1 - beta * a2 * Delta_u - eta * a1 * Delta_u
    eps2 = Delta_a2 - beta * a3 * Delta_u - 2*eta*a2*Delta_u - zeta*a1*Delta_u

    Parameters
    ----------
    df   : single-tenor increment DataFrame
    beta, eta, zeta : DGP parameters

    Returns
    -------
    residuals : (N, 3) array of [eps0, eps1, eps2]
    """
    a1 = df["a1"].values
    a2 = df["a2"].values
    a3 = df["a3"].values
    du = df["Delta_u"].values

    eps0 = df["Delta_a0"].values - beta * a1 * du
    eps1 = df["Delta_a1"].values - beta * a2 * du - eta * a1 * du
    eps2 = (
        df["Delta_a2"].values
        - beta * a3 * du
        - 2 * eta * a2 * du
        - zeta * a1 * du
    )
    return np.column_stack([eps0, eps1, eps2])


def _moving_block_resample(
    residuals: np.ndarray,
    rng: np.random.Generator,
    block_length: int = 5,
) -> np.ndarray:
    """Moving-block bootstrap resampler for residual array.

    Steps:
      1. Create all possible blocks of length `block_length`.
      2. Draw ceil(N/block_length) blocks with replacement.
      3. Concatenate and trim to length N.

    Parameters
    ----------
    residuals    : (N, 3) residual array
    rng          : numpy random Generator
    block_length : block size (default 5)

    Returns
    -------
    resampled : (N, 3) resampled residuals
    """
    N = len(residuals)
    n_blocks_needed = math.ceil(N / block_length)
    n_possible = N - block_length + 1

    # Block start indices
    starts = rng.integers(0, n_possible, size=n_blocks_needed)
    pieces = [residuals[s: s + block_length] for s in starts]
    resampled = np.concatenate(pieces, axis=0)[:N]
    return resampled


def simulate_synthetic_sample(
    df: pd.DataFrame,
    residuals: np.ndarray,
    beta_true: float,
    eta_true: float,
    zeta_true: float,
    rng: np.random.Generator,
    block_length: int = 5,
) -> pd.DataFrame:
    """Generate one synthetic sample via moving-block residual resampling.

    Keeps the observed covariate path [a1, a2, a3, Delta_u] fixed and
    constructs new responses from the true DGP plus resampled residuals.

    Parameters
    ----------
    df           : original single-tenor increment DataFrame
    residuals    : (N, 3) residuals from extract_residuals()
    beta_true, eta_true, zeta_true : DGP parameters
    rng          : numpy random Generator
    block_length : moving-block length (default 5)

    Returns
    -------
    df_sim : same-shape DataFrame with synthetic [Delta_a0, Delta_a1, Delta_a2]
    """
    eps = _moving_block_resample(residuals, rng, block_length)
    a1 = df["a1"].values
    a2 = df["a2"].values
    a3 = df["a3"].values
    du = df["Delta_u"].values

    da0_sim = beta_true * a1 * du + eps[:, 0]
    da1_sim = beta_true * a2 * du + eta_true * a1 * du + eps[:, 1]
    da2_sim = (
        beta_true * a3 * du
        + 2 * eta_true * a2 * du
        + zeta_true * a1 * du
        + eps[:, 2]
    )

    df_sim = df.copy()
    df_sim["Delta_a0"] = da0_sim
    df_sim["Delta_a1"] = da1_sim
    df_sim["Delta_a2"] = da2_sim
    return df_sim


# ---------------------------------------------------------------------------
# Bootstrap SE (pairs bootstrap)
# ---------------------------------------------------------------------------

def _pairs_bootstrap_se(
    df_sim: pd.DataFrame,
    beta_hat: float,
    eta_hat: float,
    zeta_hat: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pairs bootstrap standard errors for (beta, eta, zeta).

    Resamples rows of df_sim with replacement B times.

    Returns
    -------
    ses : (3,) bootstrap standard errors [se_beta, se_eta, se_zeta]
    """
    N = len(df_sim)
    boot_params = np.full((B, 3), np.nan)

    for b in range(B):
        idx = rng.integers(0, N, size=N)
        df_boot = df_sim.iloc[idx].reset_index(drop=True)
        try:
            b_b, e_b, z_b = _fit_params_simple(df_boot)
            boot_params[b] = [b_b, e_b, z_b]
        except Exception:
            pass

    valid = boot_params[~np.any(np.isnan(boot_params), axis=1)]
    if len(valid) < 10:
        return np.full(3, np.nan)
    return np.std(valid, axis=0, ddof=1)


# ---------------------------------------------------------------------------
# One outer replication
# ---------------------------------------------------------------------------

def _one_recovery_replication(
    df: pd.DataFrame,
    residuals: np.ndarray,
    beta_true: float,
    eta_true: float,
    zeta_true: float,
    B: int,
    rng: np.random.Generator,
) -> dict:
    """One outer replication: simulate sample → fit → inner bootstrap SE."""
    df_sim = simulate_synthetic_sample(
        df, residuals, beta_true, eta_true, zeta_true, rng
    )

    try:
        beta_hat, eta_hat, zeta_hat, _, _ = _fit_params(df_sim)
    except Exception:
        return None  # type: ignore[return-value]

    ses = _pairs_bootstrap_se(df_sim, beta_hat, eta_hat, zeta_hat, B, rng)

    def _covered(true, est, se):
        if np.isnan(se) or np.isnan(est):
            return np.nan
        return float(abs(est - true) <= 1.96 * se)

    return {
        "beta_hat": beta_hat,
        "eta_hat": eta_hat,
        "zeta_hat": zeta_hat,
        "se_beta": ses[0],
        "se_eta": ses[1],
        "se_zeta": ses[2],
        "cover_beta": _covered(beta_true, beta_hat, ses[0]),
        "cover_eta": _covered(eta_true, eta_hat, ses[1]),
        "cover_zeta": _covered(zeta_true, zeta_hat, ses[2]),
    }


def _recovery_worker(seed: int, df: pd.DataFrame, residuals: np.ndarray,
                     beta_true: float, eta_true: float, zeta_true: float,
                     B: int) -> Optional[dict]:
    rng = np.random.default_rng(seed)
    return _one_recovery_replication(df, residuals, beta_true, eta_true, zeta_true, B, rng)


def run_parameter_recovery(
    df: pd.DataFrame,
    tenor: str,
    S: int = 500,
    B: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Outer S simulations with inner B bootstrap for parameter recovery.

    Parameters
    ----------
    df    : single-tenor increment DataFrame
    tenor : tenor label (used to look up true DGP params)
    S     : number of outer replications
    B     : number of inner bootstrap replications
    seed  : base seed for reproducibility

    Returns
    -------
    summary : DataFrame with one row per parameter (beta, eta, zeta)
              columns: [param, true_val, mean_est, bias, emp_std, mean_se, coverage]
    """
    beta_true = _TRUE_BETA[tenor]
    eta_true = _TRUE_ETA[tenor]
    zeta_true = _TRUE_ZETA[tenor]

    residuals = extract_residuals(df, beta_true, eta_true, zeta_true)

    seeds = np.random.default_rng(seed).integers(0, 2**31, size=S).tolist()

    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_recovery_worker)(int(s), df, residuals, beta_true, eta_true, zeta_true, B)
        for s in seeds
    )
    results = [r for r in results if r is not None]

    if not results:
        return pd.DataFrame()

    rdf = pd.DataFrame(results)

    rows = []
    for param, true_val in [("beta", beta_true), ("eta", eta_true), ("zeta", zeta_true)]:
        ests = rdf[f"{param}_hat"].dropna()
        ses = rdf[f"se_{param}"].dropna()
        covers = rdf[f"cover_{param}"].dropna()
        rows.append({
            "param": param,
            "true_val": true_val,
            "mean_est": float(ests.mean()),
            "bias": float(ests.mean() - true_val),
            "emp_std": float(ests.std(ddof=1)),
            "mean_se": float(ses.mean()),
            "coverage": float(covers.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Power study
# ---------------------------------------------------------------------------

def _power_one_replication(
    df_full: pd.DataFrame,
    residuals: np.ndarray,
    beta_true: float,
    eta_true_nonzero: float,
    zeta_true: float,
    N: int,
    seed: int,
) -> float:
    """One replication for the power study; returns 1.0 if H0: eta=0 rejected."""
    rng = np.random.default_rng(seed)

    # Subsample N rows from the original df
    idx = rng.choice(len(df_full), size=N, replace=False)
    df_sub = df_full.iloc[idx].reset_index(drop=True)
    res_sub = residuals[idx]

    # Simulate from H1 (eta = eta_true ≠ 0)
    df_sim = simulate_synthetic_sample(
        df_sub, res_sub, beta_true, eta_true_nonzero, zeta_true, rng
    )

    try:
        _, _, _, ses, tvals = _fit_params(df_sim)
    except Exception:
        return np.nan

    t_eta = tvals[1]
    return float(abs(t_eta) > 1.96)


def run_power_study(
    df_1m: pd.DataFrame,
    sample_sizes: Optional[list] = None,
    S: int = 200,
    B: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """Power of H0: eta=0 test at 5% level for varying sample sizes.

    Simulates under H1 (eta = _TRUE_ETA['1M'] ≈ 1.93) and reports the
    fraction of replications where the null is rejected.

    Parameters
    ----------
    df_1m        : 1M tenor increment DataFrame
    sample_sizes : list of N values (default [50, 100, 200, 500])
    S            : outer replications per N
    B            : inner bootstrap (unused here – HAC SEs used directly)
    seed         : base seed

    Returns
    -------
    DataFrame with columns [N, rejection_rate]
    """
    if sample_sizes is None:
        sample_sizes = _POWER_SIZES

    beta_true = _TRUE_BETA[_POWER_TENOR]
    eta_true = _TRUE_ETA[_POWER_TENOR]
    zeta_true = _TRUE_ZETA[_POWER_TENOR]

    residuals = extract_residuals(df_1m, beta_true, eta_true, zeta_true)
    rng0 = np.random.default_rng(seed)
    rows = []

    for N in sample_sizes:
        if N > len(df_1m):
            N = len(df_1m)

        seeds = rng0.integers(0, 2**31, size=S).tolist()
        rejects = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_power_one_replication)(
                df_1m, residuals, beta_true, eta_true, zeta_true, N, int(s)
            )
            for s in seeds
        )
        rejects = [r for r in rejects if not np.isnan(r)]
        rate = float(np.mean(rejects)) if rejects else np.nan
        rows.append({"N": N, "rejection_rate": rate})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_exp6(
    panel: Optional[pd.DataFrame] = None,
    incs: Optional[pd.DataFrame] = None,
) -> dict:
    """Run Experiment 6: Finite-Sample Bootstrap Validation.

    Parameters
    ----------
    panel : pre-built panel (built if None)
    incs  : pre-built increments (built from panel if None)

    Returns
    -------
    dict with 'recovery_tables', 'power_table', and file paths.
    """
    os.makedirs("results", exist_ok=True)

    if panel is None:
        panel = build_panel(force_synthetic=True)
    if incs is None:
        incs = build_increments(panel)

    # ------------------------------------------------------------------
    # 1. Parameter-recovery study
    # ------------------------------------------------------------------
    recovery_tables = {}
    all_recovery_rows = []

    for tenor in _RECOVERY_TENORS:
        df = get_tenor_increments(incs, tenor)
        if df.empty:
            continue
        print(f"  Exp 6 – recovery study for {tenor} (N={len(df)})…")
        summary = run_parameter_recovery(df, tenor, S=500, B=150)
        summary.insert(0, "tenor", tenor)
        recovery_tables[tenor] = summary
        all_recovery_rows.append(summary)

    if all_recovery_rows:
        recovery_table = pd.concat(all_recovery_rows, ignore_index=True)
    else:
        recovery_table = pd.DataFrame()

    recovery_table.to_csv("results/exp6_recovery_table.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Power study (1M tenor)
    # ------------------------------------------------------------------
    df_1m = get_tenor_increments(incs, _POWER_TENOR)
    print(f"  Exp 6 – power study for {_POWER_TENOR} (N_full={len(df_1m)})…")
    power_table = run_power_study(df_1m, sample_sizes=_POWER_SIZES, S=200, B=150)
    power_table.to_csv("results/exp6_power_table.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Power curve plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(power_table["N"], power_table["rejection_rate"],
            "o-", color="steelblue", linewidth=2, markersize=6)
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="5% level")
    ax.axhline(0.80, color="orange", linestyle=":", linewidth=0.8, label="80% power")
    ax.set_xlabel("Sample size N")
    ax.set_ylabel("Rejection rate (H₀: η=0)")
    ax.set_title(f"Exp 6 – Power curve (1M, η_true={_TRUE_ETA[_POWER_TENOR]:.2f})")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig("results/exp6_power_curve.png", dpi=150)
    plt.close(fig)

    return {
        "recovery_tables": recovery_tables,
        "power_table": power_table,
        "recovery_table_path": "results/exp6_recovery_table.csv",
        "power_table_path": "results/exp6_power_table.csv",
        "power_curve_path": "results/exp6_power_curve.png",
    }
