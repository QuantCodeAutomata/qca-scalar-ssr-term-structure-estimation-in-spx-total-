"""
Experiment 7: Numerical PDE Transport Illustration.

Pure numerical experiment (no market data needed).

Solves the transport PDE:
    ∂_u w = beta * ∂_k w                              (linear advection)

for an SVI initial condition using the Beam-Warming second-order upwind
scheme with Heun (RK2) time integration.  The exact solution is

    w(k, u) = w0(k + beta*u)

which is used both for boundary conditions and as the reference for
the convergence study.

Convergence study: L2 and L∞ errors vs. exact at u = u_max over grid
sizes N_k ∈ {151, 301, 601, 1201}.  Expected slope in log-log: ≈ 2.

Gatheral butterfly density g[w](k) is computed from the numerical
solution to verify non-negativity (no-arbitrage).

Usage::

    from src.exp7_pde import run_exp7
    results = run_exp7()
"""

from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SVI smile and exact solution
# ---------------------------------------------------------------------------

_DEFAULT_SVI = {
    "a": 0.02,
    "b": 0.15,
    "rho": -0.4,
    "m": 0.0,
    "sigma": 0.2,
}
_DEFAULT_BETA = 1.3
_DEFAULT_K_MIN = -1.0
_DEFAULT_K_MAX = 1.0
_DEFAULT_U_MAX = 0.05
_DEFAULT_CFL = 0.4
_DEFAULT_GRID_SIZES = [151, 301, 601, 1201]


def svi_smile(
    k: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> np.ndarray:
    """Raw SVI total variance.

    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

    Parameters
    ----------
    k               : log-moneyness grid
    a, b, rho, m, sigma : SVI parameters

    Returns
    -------
    w : (len(k),) total variance values (positive by construction for valid params)
    """
    x = k - m
    return a + b * (rho * x + np.sqrt(x ** 2 + sigma ** 2))


def exact_solution(
    k: np.ndarray,
    u: float,
    beta: float,
    svi_params: Optional[dict] = None,
) -> np.ndarray:
    """Exact advected smile: w0(k + beta*u).

    The PDE ∂_u w = beta * ∂_k w has the travelling-wave solution
    w(k, u) = w0(k + beta*u), where w0 is the initial condition.

    Parameters
    ----------
    k          : k-grid at which to evaluate the exact solution
    u          : current 'time' (ATM variance level)
    beta       : transport coefficient
    svi_params : dict of SVI parameters (uses defaults if None)

    Returns
    -------
    w_exact : (len(k),) exact solution values
    """
    if svi_params is None:
        svi_params = _DEFAULT_SVI
    k_shifted = k + beta * u
    return svi_smile(k_shifted, **svi_params)


# ---------------------------------------------------------------------------
# Gatheral butterfly density
# ---------------------------------------------------------------------------

def compute_gatheral_g(
    k: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """Butterfly density functional g[w](k) (Gatheral 2004).

    g[w](k) = (1 - k*w_k/(2*w))^2 - (w_k^2/4)*(1/w + 1/4) + w_kk/2

    where w_k = ∂w/∂k and w_kk = ∂²w/∂k² are computed via central
    differences in the interior and second-order one-sided differences
    at the boundaries.

    Parameters
    ----------
    k : evenly spaced k-grid (N,)
    w : total variance at each node (N,)

    Returns
    -------
    g : (N,) butterfly density (should be ≥ 0 for no static arbitrage)
    """
    N = len(k)
    dk = k[1] - k[0]
    w_k = np.empty(N)
    w_kk = np.empty(N)

    # Interior: central differences
    w_k[1:-1] = (w[2:] - w[:-2]) / (2 * dk)
    w_kk[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / dk ** 2

    # Left boundary: forward 2nd-order
    w_k[0] = (-3 * w[0] + 4 * w[1] - w[2]) / (2 * dk)
    w_kk[0] = (w[2] - 2 * w[1] + w[0]) / dk ** 2

    # Right boundary: backward 2nd-order
    w_k[-1] = (3 * w[-1] - 4 * w[-2] + w[-3]) / (2 * dk)
    w_kk[-1] = (w[-3] - 2 * w[-2] + w[-1]) / dk ** 2

    w_safe = np.maximum(w, 1e-12)
    term1 = (1.0 - k * w_k / (2.0 * w_safe)) ** 2
    term2 = (w_k ** 2 / 4.0) * (1.0 / w_safe + 0.25)
    g = term1 - term2 + 0.5 * w_kk
    return g


# ---------------------------------------------------------------------------
# Beam-Warming upwind operator
# ---------------------------------------------------------------------------

def beam_warming_rhs(
    w: np.ndarray,
    dk: float,
    beta: float,
) -> np.ndarray:
    """Second-order Beam-Warming upwind spatial operator L(w).

    Implements the RHS of the semi-discrete system ∂_u w = L(w)
    where L(w)_i approximates beta * ∂_k w_i.

    For beta > 0 (wave moves left, upwind side is the right boundary):
        L(w)_i = beta * (-3w_i + 4w_{i+1} - w_{i+2}) / (2*dk)
        Applied for i = 0, …, N-3.  Boundary nodes i = N-2, N-1 are
        left as 0 and filled from the exact solution in rk2_step.

    For beta < 0 (wave moves right, upwind side is the left boundary):
        L(w)_i = beta * ( 3w_i - 4w_{i-1} + w_{i-2}) / (2*dk)
        Applied for i = 2, …, N-1.  Boundary nodes i = 0, 1 are
        left as 0 and filled from the exact solution in rk2_step.

    Parameters
    ----------
    w    : current solution vector (N,)
    dk   : uniform spatial step
    beta : transport coefficient

    Returns
    -------
    rhs : (N,) spatial operator output (boundary nodes zeroed)
    """
    N = len(w)
    rhs = np.zeros(N)

    if beta > 0:
        # Forward-biased stencil; last two nodes need exact BCs
        i = np.arange(N - 2)
        rhs[i] = beta * (-3.0 * w[i] + 4.0 * w[i + 1] - w[i + 2]) / (2.0 * dk)
    else:
        # Backward-biased stencil; first two nodes need exact BCs
        i = np.arange(2, N)
        rhs[i] = beta * (3.0 * w[i] - 4.0 * w[i - 1] + w[i - 2]) / (2.0 * dk)

    return rhs


def rk2_step(
    w: np.ndarray,
    dt: float,
    dk: float,
    beta: float,
    w_boundary_func,
) -> np.ndarray:
    """One RK2 (Heun method) time step with exact boundary enforcement.

    Algorithm:
        k1      = dt * L(w^n)
        w*      = w^n + k1              (predicted; boundaries filled from exact)
        k2      = dt * L(w*)
        w^{n+1} = w^n + 0.5*(k1+k2)   (corrected; boundaries filled from exact)

    Boundary nodes are overwritten from the exact solution after each stage.
    The exact values are provided by calling ``w_boundary_func()`` which must
    return a full-domain array at u^n + dt.

    Parameters
    ----------
    w                : current solution (N,) at u^n
    dt               : time step
    dk               : spatial step
    beta             : transport coefficient
    w_boundary_func  : zero-argument callable → ndarray(N,) with exact solution
                       at u^n + dt for the full spatial domain.  The caller
                       creates a closure capturing the k-grid and u_next.

    Returns
    -------
    w_new : updated solution (N,) at u^n + dt
    """
    k1 = beam_warming_rhs(w, dk, beta)
    w_star = w + dt * k1

    # Enforce inflow boundary on the intermediate state
    w_bc = w_boundary_func()
    if beta > 0:
        w_star[-2:] = w_bc[-2:]
    else:
        w_star[:2] = w_bc[:2]

    k2 = beam_warming_rhs(w_star, dk, beta)
    w_new = w + 0.5 * dt * (k1 + k2)

    # Enforce inflow boundary on the corrected state
    if beta > 0:
        w_new[-2:] = w_bc[-2:]
    else:
        w_new[:2] = w_bc[:2]

    return w_new


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def run_convergence_study(
    beta: float = _DEFAULT_BETA,
    svi_params: Optional[dict] = None,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    u_max: float = _DEFAULT_U_MAX,
    cfl: float = _DEFAULT_CFL,
    grid_sizes: Optional[list] = None,
) -> pd.DataFrame:
    """Convergence study for the Beam-Warming + RK2 scheme.

    For each grid size N_k:
      1. Build uniform k-grid over [k_min, k_max].
      2. Set initial condition from svi_smile.
      3. Time-march with dt = cfl * dk / |beta| until u = u_max.
         Boundary nodes filled from exact solution at each step.
      4. Compute L2 and L∞ errors vs. exact solution.

    Parameters
    ----------
    beta        : transport coefficient
    svi_params  : SVI parameters (uses defaults if None)
    k_min, k_max: spatial domain
    u_max       : final 'time'
    cfl         : CFL number (dt = cfl * dk / |beta|)
    grid_sizes  : list of N_k values to test

    Returns
    -------
    DataFrame with columns [N_k, dk, L2_error, Linf_error]
    """
    if svi_params is None:
        svi_params = _DEFAULT_SVI
    if grid_sizes is None:
        grid_sizes = _DEFAULT_GRID_SIZES

    rows = []
    for N_k in grid_sizes:
        k = np.linspace(k_min, k_max, N_k)
        dk = k[1] - k[0]
        dt = cfl * dk / abs(beta)
        n_steps = math.ceil(u_max / dt)
        dt_actual = u_max / n_steps  # adjust to land exactly at u_max

        w = svi_smile(k, **svi_params)
        u_curr = 0.0

        for _ in range(n_steps):
            u_next = u_curr + dt_actual
            # Closure captures k, u_next, beta, svi_params for this step
            _u = u_next
            w_bc_func = lambda _k=k, _u=_u: exact_solution(_k, _u, beta, svi_params)
            w = rk2_step(w, dt_actual, dk, beta, w_bc_func)
            u_curr = u_next

        w_ref = exact_solution(k, u_max, beta, svi_params)
        l2_err = float(np.sqrt(np.mean((w - w_ref) ** 2)))
        linf_err = float(np.max(np.abs(w - w_ref)))

        rows.append({
            "N_k": N_k,
            "dk": float(dk),
            "L2_error": l2_err,
            "Linf_error": linf_err,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_exp7(
    beta: float = _DEFAULT_BETA,
    svi_params: Optional[dict] = None,
    k_min: float = _DEFAULT_K_MIN,
    k_max: float = _DEFAULT_K_MAX,
    u_max: float = _DEFAULT_U_MAX,
    cfl: float = _DEFAULT_CFL,
    grid_sizes: Optional[list] = None,
) -> dict:
    """Run Experiment 7: Numerical PDE Transport Illustration.

    Produces:
      - Convergence table (L2 and L∞ vs dk).
      - Log-log convergence plot with fitted slope.
      - Smile transport plot (initial, numerical, exact at u_max).
      - Gatheral g plot verifying non-negativity.

    Returns
    -------
    dict with keys: convergence_slope, positivity_flag, and file paths.
    """
    os.makedirs("results", exist_ok=True)

    if svi_params is None:
        svi_params = _DEFAULT_SVI
    if grid_sizes is None:
        grid_sizes = _DEFAULT_GRID_SIZES

    # ------------------------------------------------------------------
    # 1. Convergence study
    # ------------------------------------------------------------------
    conv_table = run_convergence_study(
        beta=beta,
        svi_params=svi_params,
        k_min=k_min,
        k_max=k_max,
        u_max=u_max,
        cfl=cfl,
        grid_sizes=grid_sizes,
    )
    conv_table.to_csv("results/exp7_convergence_table.csv", index=False)

    # Estimate convergence slope in log-log space
    log_dk = np.log(conv_table["dk"].values)
    log_l2 = np.log(conv_table["L2_error"].values)
    slope = float(np.polyfit(log_dk, log_l2, 1)[0])

    # ------------------------------------------------------------------
    # 2. Detailed simulation on finest grid for plots
    # ------------------------------------------------------------------
    N_plot = grid_sizes[-1]
    k = np.linspace(k_min, k_max, N_plot)
    dk = k[1] - k[0]
    dt = cfl * dk / abs(beta)
    n_steps = math.ceil(u_max / dt)
    dt_actual = u_max / n_steps

    w = svi_smile(k, **svi_params)
    w_init = w.copy()
    u_curr = 0.0

    for _ in range(n_steps):
        u_next = u_curr + dt_actual
        _u = u_next
        w_bc_func = lambda _k=k, _u=_u: exact_solution(_k, _u, beta, svi_params)
        w = rk2_step(w, dt_actual, dk, beta, w_bc_func)
        u_curr = u_next

    w_exact_final = exact_solution(k, u_max, beta, svi_params)

    # Positivity check on numerical solution
    positivity_flag = bool(np.all(w > 0))

    # Gatheral g on initial and final smiles
    g_init = compute_gatheral_g(k, w_init)
    g_final = compute_gatheral_g(k, w)
    positivity_g_flag = bool(np.all(g_final >= -1e-6))

    # ------------------------------------------------------------------
    # 3. Smile transport plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k, w_init, "b--", lw=1.5, label=f"Initial w(k, u=0)")
    ax.plot(k, w_exact_final, "g-", lw=2, label=f"Exact w(k, u={u_max})")
    ax.plot(k, w, "r:", lw=2, label=f"Numerical (N_k={N_plot})")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("total variance w(k)")
    ax.set_title(f"Exp 7 – Smile transport (β={beta}, u_max={u_max})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp7_smile_transport.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Convergence plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(conv_table["dk"], conv_table["L2_error"],
              "o-", color="steelblue", label="L2 error")
    ax.loglog(conv_table["dk"], conv_table["Linf_error"],
              "s--", color="tomato", label="L∞ error")

    # Reference line for 2nd order
    dk_ref = conv_table["dk"].values
    ref_scale = conv_table["L2_error"].values[0] / dk_ref[0] ** 2
    ax.loglog(dk_ref, ref_scale * dk_ref ** 2, "k:", lw=1,
              label=f"O(dk²) reference")
    ax.set_xlabel("dk (grid spacing)")
    ax.set_ylabel("Error")
    ax.set_title(f"Exp 7 – Convergence (slope={slope:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/exp7_convergence.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Gatheral g plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(k, g_init, "b-", lw=1.5, label="g[w](k) initial")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("g[w](k)")
    ax.set_title("Gatheral g – initial SVI smile")
    ax.legend()

    ax = axes[1]
    ax.plot(k, g_final, "r-", lw=1.5, label="g[w](k) final (numerical)")
    ax.plot(k, compute_gatheral_g(k, w_exact_final), "g--", lw=1.5,
            label="g[w](k) final (exact)")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("k")
    ax.set_ylabel("g[w](k)")
    ax.set_title(f"Gatheral g – after transport (u={u_max})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("results/exp7_gatheral_g.png", dpi=150)
    plt.close(fig)

    return {
        "convergence_table": conv_table,
        "convergence_slope": slope,
        "positivity_flag": positivity_flag,
        "positivity_g_flag": positivity_g_flag,
        "convergence_table_path": "results/exp7_convergence_table.csv",
        "smile_transport_path": "results/exp7_smile_transport.png",
        "convergence_plot_path": "results/exp7_convergence.png",
        "gatheral_g_path": "results/exp7_gatheral_g.png",
    }
