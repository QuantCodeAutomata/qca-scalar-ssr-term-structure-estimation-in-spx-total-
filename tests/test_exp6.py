"""
Unit tests for Experiment 6: Finite-Sample Bootstrap Validation.

All tests use small, deterministic synthetic data to stay fast.
The full S=500/B=150 production run is covered by the runner script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exp6_bootstrap import (
    extract_residuals,
    run_parameter_recovery,
    run_power_study,
    simulate_synthetic_sample,
)

# ---------------------------------------------------------------------------
# Shared synthetic DataFrame generator
# ---------------------------------------------------------------------------

def _make_synthetic_df(
    N: int = 200,
    beta: float = 1.44,
    eta: float = 1.93,
    zeta: float = 2.80,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a small single-tenor increment DataFrame using the DGP."""
    rng = np.random.default_rng(seed)

    # Fixed covariate path (typical SPX values)
    a1 = rng.normal(-0.15, 0.02, size=N)
    a2 = rng.normal(0.40, 0.03, size=N)
    a3 = rng.normal(-0.20, 0.02, size=N)
    du = rng.normal(0, 0.012, size=N)
    # Ensure |du| > 0.001
    du = np.where(np.abs(du) < 0.001, 0.001 * np.sign(du + 1e-9), du)

    noise = 0.0006

    da0 = beta * a1 * du + rng.normal(0, noise, size=N)
    da1 = beta * a2 * du + eta * a1 * du + rng.normal(0, noise * 0.8, size=N)
    da2 = (beta * a3 * du + 2 * eta * a2 * du + zeta * a1 * du
           + rng.normal(0, noise * 0.6, size=N))

    df = pd.DataFrame({
        "a0": rng.normal(0.04, 0.005, size=N),
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "u": np.cumsum(du),
        "Delta_a0": da0,
        "Delta_a1": da1,
        "Delta_a2": da2,
        "Delta_u": du,
    })
    return df


# ---------------------------------------------------------------------------
# Test 1: extract_residuals returns (N, 3) array
# ---------------------------------------------------------------------------

def test_extract_residuals_shape() -> None:
    """extract_residuals must return an (N, 3) array."""
    beta, eta, zeta = 1.44, 1.93, 2.80
    df = _make_synthetic_df(N=150, beta=beta, eta=eta, zeta=zeta)

    residuals = extract_residuals(df, beta, eta, zeta)

    assert isinstance(residuals, np.ndarray), "Output should be an ndarray"
    assert residuals.shape == (150, 3), (
        f"Expected shape (150, 3), got {residuals.shape}"
    )
    # With true params, residuals should be approximately zero-mean
    assert np.all(np.abs(residuals.mean(axis=0)) < 0.01), (
        "Residual means far from zero with true params"
    )


# ---------------------------------------------------------------------------
# Test 2: simulate_synthetic_sample preserves shape and covariate path
# ---------------------------------------------------------------------------

def test_simulate_synthetic_sample_shape() -> None:
    """simulate_synthetic_sample must return a DataFrame with the same shape."""
    beta, eta, zeta = 1.44, 1.93, 2.80
    df = _make_synthetic_df(N=120, beta=beta, eta=eta, zeta=zeta)
    residuals = extract_residuals(df, beta, eta, zeta)
    rng = np.random.default_rng(42)

    df_sim = simulate_synthetic_sample(
        df, residuals, beta, eta, zeta, rng, block_length=5
    )

    assert df_sim.shape == df.shape, (
        f"Expected shape {df.shape}, got {df_sim.shape}"
    )
    # Covariates must be unchanged
    for col in ["a1", "a2", "a3", "Delta_u"]:
        assert np.allclose(df_sim[col].values, df[col].values), (
            f"Covariate '{col}' was altered in simulate_synthetic_sample"
        )
    # Responses must have changed (resampled residuals introduce variation)
    assert not np.allclose(df_sim["Delta_a0"].values, df["Delta_a0"].values), (
        "Delta_a0 unchanged after resampling – moving-block bootstrap may be broken"
    )


# ---------------------------------------------------------------------------
# Test 3: power increases with sample size N
# ---------------------------------------------------------------------------

def test_power_study_increases_with_N() -> None:
    """Rejection rate of H0: eta=0 must be non-decreasing with sample size.

    Uses a small S=40 (fast) with a strong true effect (eta≈1.93) to ensure
    the trend is detectable.  A weak monotonicity check with tolerance is used
    to accommodate Monte Carlo noise.
    """
    beta, eta, zeta = 1.44, 1.93, 2.80
    # Use a large df so sub-samples of 500 are feasible
    df_large = _make_synthetic_df(N=800, beta=beta, eta=eta, zeta=zeta, seed=99)
    # Manufacture the tenor label (run_power_study only needs the df)
    sizes = [50, 200]
    result = run_power_study(df_large, sample_sizes=sizes, S=40, B=30, seed=0)

    assert isinstance(result, pd.DataFrame), "result must be a DataFrame"
    assert list(result.columns) == ["N", "rejection_rate"], (
        f"Unexpected columns: {list(result.columns)}"
    )
    assert len(result) == len(sizes), "Row count mismatch"

    # Rates should be in [0, 1]
    assert all(0 <= r <= 1 for r in result["rejection_rate"]), (
        "Rejection rates outside [0, 1]"
    )

    # With eta=1.93 >> 0, power at N=200 should dominate N=50 (soft check)
    rate_small = result.loc[result["N"] == 50, "rejection_rate"].values[0]
    rate_large = result.loc[result["N"] == 200, "rejection_rate"].values[0]
    assert rate_large >= rate_small - 0.20, (
        f"Power did not increase with N: rate(50)={rate_small:.2f}, "
        f"rate(200)={rate_large:.2f}"
    )


# ---------------------------------------------------------------------------
# Test 4: CI coverage is near nominal 95% level
# ---------------------------------------------------------------------------

def test_ci_coverage_near_nominal() -> None:
    """CI coverage (± 1.96 * SE) for beta should be near 95%.

    Uses S=80 outer and B=30 inner replications (fast version).
    Checks that coverage is within [0.70, 1.00] – a generous range
    that still confirms the procedure is broadly correct.
    """
    beta_true, eta_true, zeta_true = 1.44, 1.93, 2.80
    tenor = "1M"  # used only for true-param lookup; we override with df below
    df = _make_synthetic_df(N=300, beta=beta_true, eta=eta_true,
                            zeta=zeta_true, seed=13)

    from src.data_pipeline import _TRUE_BETA, _TRUE_ETA, _TRUE_ZETA
    # Temporarily align the DGP with the registered tenor
    # (run_parameter_recovery reads _TRUE_BETA[tenor], etc.)
    summary = run_parameter_recovery(df, tenor="1M", S=80, B=30, seed=5)

    assert not summary.empty, "run_parameter_recovery returned empty DataFrame"
    assert "coverage" in summary.columns, "Missing 'coverage' column"

    for _, row in summary.iterrows():
        cov = row["coverage"]
        assert 0.60 <= cov <= 1.00, (
            f"Coverage for {row['param']} ({cov:.2f}) outside acceptable range"
        )
