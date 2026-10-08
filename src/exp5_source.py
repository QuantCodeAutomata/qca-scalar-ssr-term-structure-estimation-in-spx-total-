"""
Experiment 5: Transport-versus-Source Decomposition.

Tests whether smile increments at a fixed moneyness node are driven by
transport (s * Delta_u) or a residual source term (v * b1 * Delta_u).
Also tests the tracked-minimum procedure where the transport coefficient
vanishes by construction (b1(k*) = 0).

Two sub-experiments:
    1. Fixed-node: regress Delta_w(k0,t) on Delta_u and b1(k0,t)*Delta_u
       at ATM (all 4 tenors) and near-smile-min (1M, 3M).
    2. Tracked-minimum: find k*(t) daily from zero-crossing of b1,
       regress Delta_w(k*,t) on Delta_u only.

Usage::

    from src.exp5_source import run_exp5
    results = run_exp5()
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from src.data_pipeline import (
    K_GRID,
    KF_GRID,
    R2_THRESHOLD,
    DELTA_U_MIN,
    build_panel,
    build_increments,
    get_tenor_increments,
)

# Tenors targeted by Exp 5
_FIXED_NODE_TENORS = ["1M", "3M", "6M", "12M"]
_TRACKED_TENORS = ["1M", "3M"]
_N_FINE = 41  # number of fine k-nodes


# ---------------------------------------------------------------------------
# Core smile utilities
# ---------------------------------------------------------------------------

def interpolate_smile_41nodes(
    k_quoted: np.ndarray,
    w_quoted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate w(k) to 41 evenly spaced nodes using shape-preserving PCHIP.

    Parameters
    ----------
    k_quoted : (11,) forward log-moneyness at the quoted K/F grid
    w_quoted : (11,) total implied variance at each quoted node

    Returns
    -------
    k_fine : (41,) evenly spaced k nodes over the quoted range
    w_fine : (41,) interpolated total variance
    """
    k_fine = np.linspace(k_quoted[0], k_quoted[-1], _N_FINE)
    interp = PchipInterpolator(k_quoted, w_quoted, extrapolate=False)
    w_fine = interp(k_fine)
    w_fine = np.maximum(w_fine, 1e-8)
    return k_fine, w_fine


def local_skew_at_node(
    k_fine: np.ndarray,
    w_fine: np.ndarray,
    center_idx: int,
    half_width: int = 5,
) -> float:
    """Local cubic derivative (skew) at a fine-grid node.

    Fits w(k) ≈ c0 + c1*(k-k0) + c2*(k-k0)^2 + c3*(k-k0)^3 over the
    window [center_idx-half_width, center_idx+half_width] and returns c1.

    Parameters
    ----------
    k_fine      : fine k-grid (41 nodes)
    w_fine      : smile values on fine grid
    center_idx  : index of the target node
    half_width  : half-window width in nodes (default 5 → 11-node window)

    Returns
    -------
    b1 : float, local skew dw/dk at k_fine[center_idx]
    """
    lo = max(0, center_idx - half_width)
    hi = min(len(k_fine), center_idx + half_width + 1)
    k_win = k_fine[lo:hi]
    w_win = w_fine[lo:hi]
    k0 = k_fine[center_idx]
    kc = k_win - k0
    X = np.column_stack([np.ones(len(kc)), kc, kc ** 2, kc ** 3])
    try:
        c, _, _, _ = np.linalg.lstsq(X, w_win, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(c[1])


def find_smile_minimum(
    k_fine: np.ndarray,
    b1_fine: np.ndarray,
) -> Optional[float]:
    """Find interior zero-crossing of b1(k) (smile minimum location).

    Searches for a sign change from negative to positive in b1_fine
    (left side of smile has negative skew, right side positive).
    Uses Brent's method on the piecewise-linear interpolation between nodes.

    Parameters
    ----------
    k_fine  : fine k-grid
    b1_fine : skew values at each fine node

    Returns
    -------
    k_star : float or None if no interior zero-crossing found
    """
    for i in range(len(b1_fine) - 1):
        if np.isnan(b1_fine[i]) or np.isnan(b1_fine[i + 1]):
            continue
        # Exact zero at left node
        if b1_fine[i] == 0.0 and b1_fine[i + 1] > 0:
            return float(k_fine[i])
        # Strict sign change from negative to positive
        if b1_fine[i] < 0 < b1_fine[i + 1]:
            def _lin(k, i=i):
                t = (k - k_fine[i]) / (k_fine[i + 1] - k_fine[i])
                return (1 - t) * b1_fine[i] + t * b1_fine[i + 1]

            try:
                k_star = brentq(_lin, k_fine[i], k_fine[i + 1], xtol=1e-10)
                return float(k_star)
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Fixed-node regression data builder
# ---------------------------------------------------------------------------

def build_fixednode_data(
    panel: pd.DataFrame,
    tenor: str,
    k0: float,
    n_nodes: int = 41,
) -> pd.DataFrame:
    """Build regression data for fixed-node transport-source test.

    For each consecutive-day pair in the tenor slice of the panel:
      - Interpolates smile to n_nodes fine nodes.
      - Evaluates w and b1 at the nearest fine node to k0.
      - Records (Delta_w_k0, Delta_u, b1_du, Delta_t).

    Parameters
    ----------
    panel   : MultiIndex (date, tenor) panel from build_panel()
    tenor   : target tenor string
    k0      : target log-moneyness node
    n_nodes : number of fine grid nodes (default 41)

    Returns
    -------
    DataFrame with columns [Delta_w_k0, Delta_u, b1_du, Delta_t]
    """
    try:
        df_t = panel.xs(tenor, level="tenor").sort_index()
    except KeyError:
        return pd.DataFrame(columns=["Delta_w_k0", "Delta_u", "b1_du", "Delta_t"])

    df_ok = df_t[df_t["R2"] >= R2_THRESHOLD].copy()
    dates = df_ok.index.tolist()

    # Build fine grid template (same k_fine for all days)
    k_fine_template = np.linspace(K_GRID[0], K_GRID[-1], n_nodes)
    k0_idx = int(np.argmin(np.abs(k_fine_template - k0)))

    w_cols_t = [f"w_{j}" for j in range(len(KF_GRID))]
    records = []

    for i in range(len(dates) - 1):
        d_t, d_t1 = dates[i], dates[i + 1]
        row_t = df_ok.loc[d_t]
        row_t1 = df_ok.loc[d_t1]

        delta_u = float(row_t1["u"] - row_t["u"])
        if abs(delta_u) < DELTA_U_MIN:
            continue

        w_t = np.array([float(row_t[c]) for c in w_cols_t])
        w_t1 = np.array([float(row_t1[c]) for c in w_cols_t])

        k_fine, w_fine_t = interpolate_smile_41nodes(K_GRID, w_t)
        _, w_fine_t1 = interpolate_smile_41nodes(K_GRID, w_t1)

        b1 = local_skew_at_node(k_fine, w_fine_t, k0_idx)
        if np.isnan(b1):
            continue

        delta_w = float(w_fine_t1[k0_idx] - w_fine_t[k0_idx])
        records.append({
            "Delta_w_k0": delta_w,
            "Delta_u": delta_u,
            "b1_du": b1 * delta_u,
            "Delta_t": 1.0 / 252.0,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# HAC regression
# ---------------------------------------------------------------------------

def estimate_transport_source(df_node: pd.DataFrame) -> dict:
    """HAC regression for transport-source decomposition.

    Fits: Delta_w = c + s*Delta_u + v*(b1*Delta_u) + theta*Delta_t + epsilon
    using Newey-West HAC standard errors (maxlags=5).

    Because Delta_t = 1/252 is constant, it is collinear with the intercept.
    The model is therefore estimated without a separate Delta_t column;
    the constant c absorbs both c and theta/252. theta is reported as NaN.

    Parameters
    ----------
    df_node : DataFrame with columns [Delta_w_k0, Delta_u, b1_du, Delta_t]

    Returns
    -------
    dict with keys: c, s, v, theta, t_c, t_s, t_v, t_theta, n_obs, p_s
    """
    df = df_node.dropna()
    if len(df) < 10:
        return {k: np.nan for k in ["c", "s", "v", "theta",
                                     "t_c", "t_s", "t_v", "t_theta",
                                     "n_obs", "p_s"]}

    y = df["Delta_w_k0"].values
    X = sm.add_constant(
        np.column_stack([df["Delta_u"].values, df["b1_du"].values]),
        has_constant="add",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.OLS(y, X).fit()
        hac = res.get_robustcov_results(
            cov_type="HAC",
            maxlags=5,
            use_t=True,
        )

    params = hac.params      # [c, s, v]
    tvals = hac.tvalues      # [t_c, t_s, t_v]
    pvals = hac.pvalues

    return {
        "c": float(params[0]),
        "s": float(params[1]),
        "v": float(params[2]),
        "theta": np.nan,          # absorbed into c (Delta_t constant)
        "t_c": float(tvals[0]),
        "t_s": float(tvals[1]),
        "t_v": float(tvals[2]),
        "t_theta": np.nan,
        "n_obs": int(hac.nobs),
        "p_s": float(pvals[1]),
    }


# ---------------------------------------------------------------------------
# Tracked-minimum procedure
# ---------------------------------------------------------------------------

def run_tracked_minimum_test(
    panel: pd.DataFrame,
    tenor: str,
) -> dict:
    """Tracked-minimum regression for the given tenor.

    For each day, finds k*(t) as the interior zero-crossing of b1(k,t)
    (smile minimum).  Regresses Delta_w(k*,t) on Delta_u.
    Since b1(k*) = 0 by construction, the transport correction term
    v*b1*Delta_u is absent.

    Parameters
    ----------
    panel : MultiIndex panel from build_panel()
    tenor : '1M' or '3M'

    Returns
    -------
    dict with keys: s_min, t_stat, p_value, n_obs
    """
    try:
        df_t = panel.xs(tenor, level="tenor").sort_index()
    except KeyError:
        return {"s_min": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": 0}

    df_ok = df_t[df_t["R2"] >= R2_THRESHOLD].copy()
    dates = df_ok.index.tolist()
    w_cols_t = [f"w_{j}" for j in range(len(KF_GRID))]

    rows = []
    for i in range(len(dates) - 1):
        d_t, d_t1 = dates[i], dates[i + 1]
        row_t = df_ok.loc[d_t]
        row_t1 = df_ok.loc[d_t1]

        delta_u = float(row_t1["u"] - row_t["u"])
        if abs(delta_u) < DELTA_U_MIN:
            continue

        w_t = np.array([float(row_t[c]) for c in w_cols_t])
        w_t1 = np.array([float(row_t1[c]) for c in w_cols_t])

        # Interpolate smiles to fine grid
        interp_t = PchipInterpolator(K_GRID, w_t, extrapolate=False)
        interp_t1 = PchipInterpolator(K_GRID, w_t1, extrapolate=False)

        k_fine = np.linspace(K_GRID[0], K_GRID[-1], _N_FINE)
        # b1 = first derivative of the Pchip interpolant
        b1_fine = interp_t.derivative()(k_fine)

        k_star = find_smile_minimum(k_fine, b1_fine)
        if k_star is None:
            continue

        # Evaluate Delta_w at k* by interpolation
        w_kstar_t = float(interp_t(k_star))
        w_kstar_t1 = float(interp_t1(k_star))
        if np.isnan(w_kstar_t) or np.isnan(w_kstar_t1):
            continue

        delta_w = w_kstar_t1 - w_kstar_t
        rows.append({"Delta_w_kstar": delta_w, "Delta_u": delta_u})

    if len(rows) < 10:
        return {"s_min": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_obs": len(rows)}

    df_reg = pd.DataFrame(rows).dropna()
    y = df_reg["Delta_w_kstar"].values
    X = sm.add_constant(df_reg["Delta_u"].values, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.OLS(y, X).fit()
        hac = res.get_robustcov_results(
            cov_type="HAC",
            maxlags=5,
            use_t=True,
        )

    return {
        "s_min": float(hac.params[1]),
        "t_stat": float(hac.tvalues[1]),
        "p_value": float(hac.pvalues[1]),
        "n_obs": int(hac.nobs),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _median_smile_min(panel: pd.DataFrame, tenor: str) -> Optional[float]:
    """Return the median smile-minimum k* across all usable days for a tenor."""
    try:
        df_t = panel.xs(tenor, level="tenor").sort_index()
    except KeyError:
        return None

    df_ok = df_t[df_t["R2"] >= R2_THRESHOLD]
    w_cols = [f"w_{j}" for j in range(len(KF_GRID))]
    k_fine = np.linspace(K_GRID[0], K_GRID[-1], _N_FINE)
    k_stars = []

    for d in df_ok.index:
        w = np.array([float(df_ok.loc[d, c]) for c in w_cols])
        b1_fine = PchipInterpolator(K_GRID, w).derivative()(k_fine)
        k_star = find_smile_minimum(k_fine, b1_fine)
        if k_star is not None:
            k_stars.append(k_star)

    return float(np.median(k_stars)) if k_stars else None


def run_exp5(
    panel: Optional[pd.DataFrame] = None,
    incs: Optional[pd.DataFrame] = None,
) -> dict:
    """Run Experiment 5: Transport-versus-Source Decomposition.

    Performs:
      - Fixed-node regression at ATM (k0=0) for tenors 1M, 3M, 6M, 12M.
      - Fixed-node regression near smile minimum for 1M and 3M.
      - Tracked-minimum regression for 1M and 3M.

    Parameters
    ----------
    panel : pre-built panel (built if None)
    incs  : pre-built increments (unused; kept for interface consistency)

    Returns
    -------
    dict with keys 'fixednode_results', 'tracked_min_results',
    and file paths of saved artefacts.
    """
    os.makedirs("results", exist_ok=True)

    if panel is None:
        panel = build_panel(force_synthetic=True)

    # ------------------------------------------------------------------
    # 1. Fixed-node regressions
    # ------------------------------------------------------------------
    fixednode_rows = []

    # ATM for all 4 tenors
    for tenor in _FIXED_NODE_TENORS:
        df_node = build_fixednode_data(panel, tenor, k0=0.0)
        res = estimate_transport_source(df_node)
        fixednode_rows.append({
            "tenor": tenor,
            "node": "ATM (k0=0)",
            **res,
        })

    # Near smile-min for 1M and 3M
    for tenor in ["1M", "3M"]:
        k_min = _median_smile_min(panel, tenor)
        if k_min is None:
            continue
        df_node = build_fixednode_data(panel, tenor, k0=k_min)
        res = estimate_transport_source(df_node)
        fixednode_rows.append({
            "tenor": tenor,
            "node": f"near min (k0={k_min:.4f})",
            **res,
        })

    fixednode_table = pd.DataFrame(fixednode_rows)
    fixednode_table.to_csv("results/exp5_fixednode_table.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Tracked-minimum regressions
    # ------------------------------------------------------------------
    tracked_rows = []
    for tenor in _TRACKED_TENORS:
        res = run_tracked_minimum_test(panel, tenor)
        tracked_rows.append({"tenor": tenor, **res})
    tracked_table = pd.DataFrame(tracked_rows)
    tracked_table.to_csv("results/exp5_tracked_min_table.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Plots
    # ------------------------------------------------------------------
    _plot_fixednode_tstat(fixednode_table)
    _plot_smile_and_min(panel)

    return {
        "fixednode_results": fixednode_table,
        "tracked_min_results": tracked_table,
        "fixednode_table_path": "results/exp5_fixednode_table.csv",
        "tracked_min_table_path": "results/exp5_tracked_min_table.csv",
    }


def _plot_fixednode_tstat(table: pd.DataFrame) -> None:
    """Bar chart of t-statistics for H0: s=0 in fixed-node regressions."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{r['tenor']} / {r['node']}" for _, r in table.iterrows()]
    tvals = table["t_s"].values
    colors = ["steelblue" if abs(t) > 1.96 else "salmon" for t in tvals]
    bars = ax.bar(range(len(labels)), tvals, color=colors)
    ax.axhline(1.96, color="k", linestyle="--", linewidth=0.8, label="|t|=1.96")
    ax.axhline(-1.96, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("t-statistic (H₀: s=0)")
    ax.set_title("Exp 5 – Fixed-node: t-stat for pure-transport null")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp5_fixednode_tstats.png", dpi=150)
    plt.close(fig)


def _plot_smile_and_min(panel: pd.DataFrame) -> None:
    """Plot a representative smile for 1M with its minimum marked."""
    try:
        df_1m = panel.xs("1M", level="tenor").sort_index()
    except KeyError:
        return

    df_ok = df_1m[df_1m["R2"] >= R2_THRESHOLD]
    if df_ok.empty:
        return

    # Use the median date
    mid = df_ok.index[len(df_ok) // 2]
    w_cols = [f"w_{j}" for j in range(len(KF_GRID))]
    w = np.array([float(df_ok.loc[mid, c]) for c in w_cols])

    k_fine, w_fine = interpolate_smile_41nodes(K_GRID, w)
    b1_fine = PchipInterpolator(K_GRID, w).derivative()(k_fine)
    k_star = find_smile_minimum(k_fine, b1_fine)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Smile
    ax = axes[0]
    ax.plot(k_fine, w_fine, "b-", lw=2, label="w(k) 1M")
    ax.scatter(K_GRID, w, color="k", zorder=5, s=30, label="quoted nodes")
    if k_star is not None:
        w_star = float(PchipInterpolator(K_GRID, w)(k_star))
        ax.axvline(k_star, color="r", linestyle="--", label=f"min k*={k_star:.3f}")
        ax.scatter([k_star], [w_star], color="r", zorder=6, s=60)
    ax.axvline(0, color="gray", linestyle=":", label="ATM")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("total variance w(k)")
    ax.set_title("1M smile (representative day)")
    ax.legend(fontsize=7)

    # Skew
    ax = axes[1]
    ax.plot(k_fine, b1_fine, "g-", lw=2)
    ax.axhline(0, color="k", linewidth=0.8)
    if k_star is not None:
        ax.axvline(k_star, color="r", linestyle="--", label=f"b1=0 at k*={k_star:.3f}")
        ax.legend(fontsize=7)
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("b₁(k) = ∂w/∂k")
    ax.set_title("Local skew (1M)")

    fig.tight_layout()
    fig.savefig("results/exp5_smile_and_min.png", dpi=150)
    plt.close(fig)
