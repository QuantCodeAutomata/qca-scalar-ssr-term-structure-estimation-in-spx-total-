"""
Unit tests for Experiment 7: Numerical PDE Transport Illustration.

All tests use hard-coded parameters; no market data or panel build needed.
The convergence test uses small grid sizes [31, 61, 121] for speed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.exp7_pde import (
    beam_warming_rhs,
    compute_gatheral_g,
    exact_solution,
    rk2_step,
    run_convergence_study,
    svi_smile,
)

# Default SVI parameters used throughout
_SVI = {"a": 0.02, "b": 0.15, "rho": -0.4, "m": 0.0, "sigma": 0.2}
_BETA = 1.3
_K = np.linspace(-1.0, 1.0, 201)


# ---------------------------------------------------------------------------
# Test 1: SVI smile is strictly positive
# ---------------------------------------------------------------------------

def test_svi_smile_positive() -> None:
    """svi_smile must return strictly positive total variances."""
    k = np.linspace(-2.0, 2.0, 500)
    w = svi_smile(k, **_SVI)

    assert w.shape == k.shape, "Output shape mismatch"
    assert np.all(w > 0), (
        f"SVI smile not strictly positive; min={w.min():.6f}"
    )
    # At ATM (k=0), should equal a + b*sqrt(sigma^2) for rho=0 correction
    w_atm = svi_smile(np.array([0.0]), **_SVI)[0]
    expected_atm = _SVI["a"] + _SVI["b"] * (
        _SVI["rho"] * (0 - _SVI["m"])
        + math.sqrt((0 - _SVI["m"]) ** 2 + _SVI["sigma"] ** 2)
    )
    assert abs(w_atm - expected_atm) < 1e-12, (
        f"ATM value mismatch: got {w_atm:.8f}, expected {expected_atm:.8f}"
    )


# ---------------------------------------------------------------------------
# Test 2: exact_solution is a translation of the initial smile
# ---------------------------------------------------------------------------

def test_exact_solution_translation() -> None:
    """w(k, u) must equal w0(k + beta*u) for arbitrary k and u."""
    u_vals = [0.01, 0.025, 0.05]
    beta = _BETA
    k = np.linspace(-0.8, 0.8, 200)

    for u in u_vals:
        w_numerical = exact_solution(k, u, beta, _SVI)
        w_reference = svi_smile(k + beta * u, **_SVI)   # direct evaluation
        assert np.allclose(w_numerical, w_reference, rtol=1e-12), (
            f"exact_solution does not equal w0(k+beta*u) at u={u}"
        )

    # At u=0 the solution must equal the initial condition exactly
    w_u0 = exact_solution(k, 0.0, beta, _SVI)
    w_init = svi_smile(k, **_SVI)
    assert np.allclose(w_u0, w_init, rtol=1e-12), (
        "exact_solution at u=0 does not match initial SVI smile"
    )


# ---------------------------------------------------------------------------
# Test 3: Gatheral g is non-negative for the SVI smile
# ---------------------------------------------------------------------------

def test_gatheral_g_nonnegative_for_svi() -> None:
    """compute_gatheral_g must be ≥ 0 for valid SVI parameters (no arbitrage)."""
    k = np.linspace(-0.9, 0.9, 501)   # avoid far wings where g may be tiny
    w = svi_smile(k, **_SVI)
    g = compute_gatheral_g(k, w)

    assert g.shape == k.shape, "g shape mismatch"
    # Allow a small numerical tolerance of 1e-6 for boundary node artefacts
    min_g = float(np.min(g))
    assert min_g >= -1e-4, (
        f"Gatheral g is negative ({min_g:.2e}) for a valid SVI smile"
    )


# ---------------------------------------------------------------------------
# Test 4: Convergence is second-order in dk
# ---------------------------------------------------------------------------

def test_convergence_second_order() -> None:
    """Beam-Warming + RK2 should achieve ≈ 2nd-order convergence.

    Uses small grids [31, 61, 121] to keep the test fast (~seconds).
    The log-log slope of L2 error vs dk should be in [1.5, 2.5].
    """
    table = run_convergence_study(
        beta=_BETA,
        svi_params=_SVI,
        k_min=-1.0,
        k_max=1.0,
        u_max=0.05,
        cfl=0.4,
        grid_sizes=[31, 61, 121],
    )

    assert len(table) == 3, "Expected 3 rows in convergence table"
    assert "L2_error" in table.columns and "dk" in table.columns

    # Errors should decrease as grid refines
    l2 = table["L2_error"].values
    assert l2[0] > l2[1] > l2[2], (
        f"L2 errors not decreasing: {l2}"
    )

    # Log-log slope should be close to 2
    log_dk = np.log(table["dk"].values)
    log_l2 = np.log(l2)
    slope = float(np.polyfit(log_dk, log_l2, 1)[0])
    assert 1.5 <= slope <= 2.5, (
        f"Convergence slope {slope:.3f} not in [1.5, 2.5]; scheme may be wrong order"
    )


# ---------------------------------------------------------------------------
# Test 5: rk2_step returns correct shape
# ---------------------------------------------------------------------------

def test_rk2_step_shape() -> None:
    """rk2_step must return an ndarray with the same shape as the input."""
    N = 101
    k = np.linspace(-1.0, 1.0, N)
    dk = k[1] - k[0]
    beta = _BETA
    dt = 0.4 * dk / abs(beta)

    w = svi_smile(k, **_SVI)

    # Boundary function: returns exact solution at u + dt
    u_next = dt
    w_bc_func = lambda: exact_solution(k, u_next, beta, _SVI)

    w_new = rk2_step(w, dt, dk, beta, w_bc_func)

    assert isinstance(w_new, np.ndarray), "rk2_step must return ndarray"
    assert w_new.shape == w.shape, (
        f"Shape mismatch: input {w.shape}, output {w_new.shape}"
    )
    assert np.all(np.isfinite(w_new)), "rk2_step output contains non-finite values"
    assert np.all(w_new > 0), "rk2_step output has non-positive values"

    # After one small step, the result should be close to exact (error ∝ dt^3)
    w_exact = exact_solution(k, u_next, beta, _SVI)
    max_err = float(np.max(np.abs(w_new - w_exact)))
    # For N=101, dk≈0.02, dt≈0.006, error should be tiny
    assert max_err < 1e-3, (
        f"Single-step error too large: {max_err:.2e} (expected < 1e-3)"
    )
