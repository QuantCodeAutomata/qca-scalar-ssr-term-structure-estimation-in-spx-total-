"""
Unit tests for Experiment 5: Transport-versus-Source Decomposition.

All tests use deterministic synthetic data or hard-coded parameters;
no market data or full panel build is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import K_GRID, KF_GRID
from src.exp5_source import (
    estimate_transport_source,
    find_smile_minimum,
    interpolate_smile_41nodes,
    local_skew_at_node,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _spx_like_w(k: np.ndarray) -> np.ndarray:
    """Synthetic SPX-like total variance: w = a0 + a1*k + (a2/2)*k^2.

    a0 = 0.04, a1 = -0.10 (negative skew), a2 = 0.30 (positive convexity)
    → minimum to the left of ATM (k ≈ a1/a2 = -0.33).
    """
    return 0.04 + (-0.10) * k + (0.30 / 2.0) * k ** 2


# ---------------------------------------------------------------------------
# Test 1: interpolate_smile_41nodes returns exactly 41 nodes
# ---------------------------------------------------------------------------

def test_interpolate_smile_41nodes_correct_length() -> None:
    """interpolate_smile_41nodes must return arrays of length 41."""
    k_q = K_GRID.copy()
    w_q = _spx_like_w(k_q)
    k_fine, w_fine = interpolate_smile_41nodes(k_q, w_q)

    assert len(k_fine) == 41, f"Expected 41 fine nodes, got {len(k_fine)}"
    assert len(w_fine) == 41, f"Expected 41 w values, got {len(w_fine)}"
    # k_fine must span the quoted range
    assert abs(k_fine[0] - k_q[0]) < 1e-12
    assert abs(k_fine[-1] - k_q[-1]) < 1e-12
    # Spacing must be uniform
    diffs = np.diff(k_fine)
    assert np.allclose(diffs, diffs[0], rtol=1e-6), "k_fine spacing not uniform"
    # Interpolated values must be positive (total variance > 0)
    assert np.all(w_fine > 0), "Some interpolated variances are non-positive"


# ---------------------------------------------------------------------------
# Test 2: local skew sign is negative at ATM for an SPX-like smile
# ---------------------------------------------------------------------------

def test_local_skew_sign_negative_for_spx() -> None:
    """Local skew at ATM should be negative for an SPX-like smile (a1 < 0)."""
    k_q = K_GRID.copy()
    w_q = _spx_like_w(k_q)
    k_fine, w_fine = interpolate_smile_41nodes(k_q, w_q)

    # ATM is k=0; find the nearest fine node
    atm_idx = int(np.argmin(np.abs(k_fine)))
    b1_atm = local_skew_at_node(k_fine, w_fine, center_idx=atm_idx, half_width=5)

    assert not np.isnan(b1_atm), "local_skew_at_node returned NaN"
    assert b1_atm < 0.0, (
        f"Expected negative skew at ATM for SPX-like smile, got b1={b1_atm:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: find_smile_minimum returns None for a monotone b1 array
# ---------------------------------------------------------------------------

def test_find_smile_minimum_returns_none_for_monotone() -> None:
    """find_smile_minimum must return None if b1 is strictly negative everywhere."""
    k_fine = np.linspace(-0.16, 0.14, 41)

    # Case 1: all-negative b1 (pure downward skew, no minimum)
    b1_all_neg = -0.1 * np.ones(41)
    result = find_smile_minimum(k_fine, b1_all_neg)
    assert result is None, f"Expected None for all-negative b1, got {result}"

    # Case 2: all-positive b1 (monotone increasing w, no interior minimum)
    b1_all_pos = 0.1 * np.ones(41)
    result = find_smile_minimum(k_fine, b1_all_pos)
    assert result is None, f"Expected None for all-positive b1, got {result}"

    # Sanity check: a sign change should be detected
    b1_crossing = np.linspace(-0.1, 0.1, 41)  # crosses zero in the interior
    result = find_smile_minimum(k_fine, b1_crossing)
    assert result is not None, "Expected a zero crossing to be found"
    assert k_fine[0] < result < k_fine[-1], "Zero crossing outside domain"


# ---------------------------------------------------------------------------
# Test 4: estimate_transport_source returns dict with all required keys
# ---------------------------------------------------------------------------

def test_transport_source_regression_keys() -> None:
    """estimate_transport_source must return a dict with the full set of keys."""
    rng = np.random.default_rng(0)
    N = 300

    # Synthetic regression data
    delta_u = rng.normal(0, 0.012, size=N)
    b1 = rng.uniform(-0.15, -0.05, size=N)       # typical SPX negative skew
    b1_du = b1 * delta_u

    # True model: Delta_w = 1.4*Delta_u + 0*b1_du + noise
    delta_w = 1.4 * delta_u + rng.normal(0, 5e-4, size=N)

    df_node = pd.DataFrame({
        "Delta_w_k0": delta_w,
        "Delta_u": delta_u,
        "b1_du": b1_du,
        "Delta_t": np.full(N, 1.0 / 252.0),
    })

    result = estimate_transport_source(df_node)

    required_keys = {"c", "s", "v", "theta", "t_c", "t_s", "t_v", "t_theta",
                     "n_obs", "p_s"}
    assert required_keys.issubset(set(result.keys())), (
        f"Missing keys: {required_keys - set(result.keys())}"
    )

    # s coefficient should be positive and significant for the DGP above
    assert result["n_obs"] > 0, "n_obs must be positive"
    assert not np.isnan(result["s"]), "s coefficient is NaN"
    assert result["s"] > 0, f"Expected positive s for DGP, got s={result['s']:.4f}"
    assert abs(result["t_s"]) > 2.0, (
        f"Expected significant t_s for N={N}, got t_s={result['t_s']:.2f}"
    )
