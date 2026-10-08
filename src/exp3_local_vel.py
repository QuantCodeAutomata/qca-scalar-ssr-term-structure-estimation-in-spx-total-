"""
Experiment 3: Strike-Dependent Local Velocity Profile.

Estimates the local transport velocity v0(k0) at fixed expansion centers
across the smile.  For each date-tenor pair, a local cubic polynomial is
fitted around each k0 to extract the level b0 and slope b1 at that point.
Day-to-day changes in b0 are then regressed on b1 * Delta_u (OLS, no
intercept) to recover v0(k0).  Moving-block bootstrap covariance and a
smooth-trend Wald test complete the analysis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_pipeline import (
    K_GRID,
    TENOR_LABELS,
    build_increments,
    build_panel,
    get_tenor_increments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K0_CENTERS: np.ndarray = np.log([0.905, 0.928, 0.951, 0.975, 1.000, 1.025, 1.051])
KF0_CENTERS: np.ndarray = np.array([0.905, 0.928, 0.951, 0.975, 1.000, 1.025, 1.051])
RESULTS_DIR = Path("/workspace/project/results")

_SKEW_THRESHOLD_DEFAULT = 0.001
_N_GRID = 11  # number of smile grid nodes


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def local_cubic_jets(
    k_grid: np.ndarray,
    w_grid: np.ndarray,
    k0: float,
) -> tuple:
    """
    Fit local cubic polynomial w(k) ≈ b0 + b1*(k-k0) + b2*(k-k0)² + b3*(k-k0)³
    using all nodes in k_grid.

    Parameters
    ----------
    k_grid : (N,) log-moneyness nodes
    w_grid : (N,) total-variance values at those nodes
    k0     : expansion center

    Returns
    -------
    (b0, b1, b2, r2)
        b0  – polynomial value at k0  (= w(k0))
        b1  – first derivative at k0  (= dw/dk at k0)
        b2  – coefficient on (k-k0)²
        r2  – cross-sectional R² of local fit
    """
    x = k_grid - k0
    X = np.column_stack([np.ones(len(x)), x, x ** 2, x ** 3])
    c, _, _, _ = np.linalg.lstsq(X, w_grid, rcond=None)

    w_hat = X @ c
    ss_res = float(np.sum((w_grid - w_hat) ** 2))
    ss_tot = float(np.sum((w_grid - np.mean(w_grid)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 1.0

    return float(c[0]), float(c[1]), float(c[2]), r2


def build_local_velocity_data(
    panel: pd.DataFrame,
    tenor: str,
    k0: float,
    skew_threshold: float = _SKEW_THRESHOLD_DEFAULT,
) -> pd.DataFrame:
    """
    Build the regression dataset for v0 at a single (tenor, k0) node.

    For every retained increment row the function fits local cubic jets at
    time t and t+1 and records:
        Delta_b0 = b0(t+1) - b0(t)   [smile change at k0]
        b1_du    = b1(t) * Delta_u    [regressor]

    Rows where |b1(t)| < skew_threshold are dropped.

    Returns
    -------
    DataFrame with columns [date, Delta_b0, b1_du]
    """
    incs = build_increments(panel)
    df = get_tenor_increments(incs, tenor)

    records = []
    for date, row in df.iterrows():
        w_t = np.array([row[f"w_t_{j}"] for j in range(_N_GRID)])
        w_t1 = np.array([row[f"w_t1_{j}"] for j in range(_N_GRID)])

        b0_t, b1_t, _, _ = local_cubic_jets(K_GRID, w_t, k0)
        b0_t1, _, _, _ = local_cubic_jets(K_GRID, w_t1, k0)

        if abs(b1_t) < skew_threshold:
            continue

        records.append(
            {
                "date": date,
                "Delta_b0": b0_t1 - b0_t,
                "b1_du": b1_t * float(row["Delta_u"]),
            }
        )

    return pd.DataFrame(records, columns=["date", "Delta_b0", "b1_du"])


def estimate_local_velocity(df_node: pd.DataFrame) -> dict:
    """
    OLS estimate of v0 at a single (tenor, k0) node.

    Regresses Delta_b0 on b1_du without intercept using HC3 standard errors.

    Returns
    -------
    dict with keys: v0, se_hc3, t_stat, p_val
    """
    y = df_node["Delta_b0"].values
    X = df_node["b1_du"].values
    result = sm.OLS(y, X).fit(cov_type="HC3", use_t=False)
    return {
        "v0": float(np.atleast_1d(result.params)[0]),
        "se_hc3": float(np.atleast_1d(result.bse)[0]),
        "t_stat": float(np.atleast_1d(result.tvalues)[0]),
        "p_val": float(np.atleast_1d(result.pvalues)[0]),
    }


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------


def moving_block_bootstrap_cov(
    panel: pd.DataFrame,
    tenor: str,
    k0_centers: np.ndarray,
    B: int = 500,
    block_length: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Moving-block bootstrap covariance matrix (7×7) of v̂ across k0 centers.

    For each bootstrap replication, floor(T/block_length) overlapping blocks
    of length `block_length` are drawn with replacement and concatenated.
    v0 is re-estimated at each center via OLS (no intercept) and the sample
    covariance of the B replication vectors is returned.

    Returns
    -------
    Sigma : (n_centers, n_centers) ndarray
    """
    incs = build_increments(panel)
    df = get_tenor_increments(incs, tenor)
    T = len(df)
    n_c = len(k0_centers)

    # Pre-compute Delta_b0 and b1*du for every (obs, k0_center) pair
    Delta_b0_all = np.zeros((T, n_c))
    b1_du_all = np.zeros((T, n_c))
    valid_all = np.ones((T, n_c), dtype=bool)

    df_reset = df.reset_index(drop=True)
    for i, row in df_reset.iterrows():
        w_t = np.array([row[f"w_t_{j}"] for j in range(_N_GRID)])
        w_t1 = np.array([row[f"w_t1_{j}"] for j in range(_N_GRID)])
        du = float(row["Delta_u"])
        for m, k0 in enumerate(k0_centers):
            b0_t, b1_t, _, _ = local_cubic_jets(K_GRID, w_t, k0)
            b0_t1, _, _, _ = local_cubic_jets(K_GRID, w_t1, k0)
            if abs(b1_t) < _SKEW_THRESHOLD_DEFAULT:
                valid_all[i, m] = False
            else:
                Delta_b0_all[i, m] = b0_t1 - b0_t
                b1_du_all[i, m] = b1_t * du

    rng = np.random.default_rng(seed)
    n_blocks = max(T // block_length, 1)
    max_start = max(T - block_length, 0)
    v_boots = np.zeros((B, n_c))

    for b in range(B):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate(
            [np.arange(s, min(s + block_length, T)) for s in starts]
        )[:T]

        for m in range(n_c):
            mask = valid_all[idx, m]
            db0 = Delta_b0_all[idx[mask], m]
            b1du = b1_du_all[idx[mask], m]
            denom = float(np.dot(b1du, b1du))
            v_boots[b, m] = float(np.dot(b1du, db0)) / denom if denom > 1e-14 else 0.0

    return np.cov(v_boots.T)


# ---------------------------------------------------------------------------
# Smooth-trend Wald test
# ---------------------------------------------------------------------------


def smooth_trend_wald_test(
    v_hat: np.ndarray,
    k0_centers: np.ndarray,
    Sigma: np.ndarray,
) -> dict:
    """
    Fit v0(k0) = c_const + c1*k0 + c2*k0² by OLS and test H0: c1 = c2 = 0.

    The covariance of c_hat is computed via the delta method using the
    bootstrap Sigma as the covariance of v_hat.  The Wald statistic is
    chi-squared with 2 degrees of freedom.

    Returns
    -------
    dict with keys: c1, c2, wald_stat, p_value
    """
    Z = np.column_stack([np.ones(len(k0_centers)), k0_centers, k0_centers ** 2])
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    c_hat = ZtZ_inv @ Z.T @ v_hat  # (3,)

    # Covariance via delta method
    V_c = ZtZ_inv @ Z.T @ Sigma @ Z @ ZtZ_inv  # (3, 3)

    # Restriction: H0: c1 = c2 = 0  →  R = [[0,1,0],[0,0,1]]
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Rc = R @ c_hat  # (2,)
    RVR = R @ V_c @ R.T  # (2, 2)

    try:
        wald_stat = float(Rc @ np.linalg.solve(RVR, Rc))
    except np.linalg.LinAlgError:
        wald_stat = float("nan")

    p_value = float(1.0 - stats.chi2.cdf(wald_stat, df=2)) if np.isfinite(wald_stat) else float("nan")

    return {
        "c1": float(c_hat[1]),
        "c2": float(c_hat[2]),
        "wald_stat": wald_stat,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_exp3(panel: pd.DataFrame | None = None, incs: pd.DataFrame | None = None) -> dict:
    """
    Run Experiment 3 for all seven tenors.

    For each tenor:
    1. Estimate v0(k0) at the seven k0 centers (OLS, HC3).
    2. Compute moving-block bootstrap covariance (7×7).
    3. Smooth-trend Wald test for shape of the velocity profile.

    Saves
    -----
    results/exp3_local_velocity_table.csv
    results/exp3_velocity_profiles.png
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if panel is None:
        panel = build_panel(force_synthetic=True)

    all_results: dict = {}
    table_rows: list[dict] = []

    for tenor in TENOR_LABELS:
        v_hat = np.full(len(K0_CENTERS), np.nan)
        se_arr = np.full(len(K0_CENTERS), np.nan)
        node_ests: list[dict] = []

        for m, k0 in enumerate(K0_CENTERS):
            df_node = build_local_velocity_data(panel, tenor, k0)
            if len(df_node) < 3:
                node_ests.append({})
                continue
            est = estimate_local_velocity(df_node)
            v_hat[m] = est["v0"]
            se_arr[m] = est["se_hc3"]
            node_ests.append(est)

        # Moving-block bootstrap covariance
        Sigma = moving_block_bootstrap_cov(panel, tenor, K0_CENTERS)

        # Smooth-trend Wald test on valid centers
        valid = ~np.isnan(v_hat)
        if valid.sum() >= 3:
            wald = smooth_trend_wald_test(
                v_hat[valid],
                K0_CENTERS[valid],
                Sigma[np.ix_(valid, valid)],
            )
        else:
            wald = {"c1": np.nan, "c2": np.nan, "wald_stat": np.nan, "p_value": np.nan}

        all_results[tenor] = {
            "v_hat": v_hat,
            "se": se_arr,
            "Sigma": Sigma,
            "wald": wald,
        }

        # Collect table rows
        for m, k0 in enumerate(K0_CENTERS):
            est = node_ests[m]
            table_rows.append(
                {
                    "tenor": tenor,
                    "k0": round(k0, 4),
                    "kf": round(float(np.exp(k0)), 4),
                    "v0": est.get("v0", np.nan),
                    "se_hc3": est.get("se_hc3", np.nan),
                    "t_stat": est.get("t_stat", np.nan),
                    "p_val": est.get("p_val", np.nan),
                    "wald_stat": wald["wald_stat"],
                    "wald_pval": wald["p_value"],
                }
            )

    # Save table
    pd.DataFrame(table_rows).to_csv(
        RESULTS_DIR / "exp3_local_velocity_table.csv", index=False
    )

    # Save plot
    _plot_velocity_profiles(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_velocity_profiles(all_results: dict) -> None:
    kf_centers = KF0_CENTERS
    n_tenors = len(TENOR_LABELS)
    ncols = 4
    nrows = (n_tenors + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()

    for i, tenor in enumerate(TENOR_LABELS):
        ax = axes_flat[i]
        res = all_results[tenor]
        v = res["v_hat"]
        se = res["se"]
        valid = ~np.isnan(v)

        if valid.any():
            ax.plot(kf_centers[valid], v[valid], "o-", color="steelblue", lw=1.5)
            ax.fill_between(
                kf_centers[valid],
                v[valid] - 1.96 * se[valid],
                v[valid] + 1.96 * se[valid],
                alpha=0.25,
                color="steelblue",
            )
        ax.axhline(0, color="gray", lw=0.8, ls="--")

        wald = res["wald"]
        title = f"{tenor}"
        if np.isfinite(wald.get("wald_stat", float("nan"))):
            title += f"\nWald={wald['wald_stat']:.2f} p={wald['p_value']:.3f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("K/F", fontsize=8)
        ax.set_ylabel("v₀(k₀)", fontsize=8)

    for j in range(n_tenors, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Experiment 3: Local Transport Velocity Profile v₀(k₀)", fontsize=11)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_velocity_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
