"""Pytest tests for Experiment 1: Scalar SSR Term Structure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    TENOR_LABELS,
    build_increments,
    build_panel,
    get_tenor_increments,
)
from src.exp1_ssr import estimate_rolling_ssr, estimate_scalar_ssr, run_exp1

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """Build synthetic panel and increments once for the whole module."""
    panel = build_panel(force_synthetic=True, seed=42)
    incs = build_increments(panel)
    return panel, incs


@pytest.fixture(scope="module")
def first_tenor_df(synthetic_data):
    """Single-tenor (1M) increment DataFrame."""
    _, incs = synthetic_data
    return get_tenor_increments(incs, "1M")


@pytest.fixture(scope="module")
def full_results(synthetic_data):
    """Full run_exp1 results (module-scoped to avoid recomputation)."""
    panel, incs = synthetic_data
    return run_exp1(panel=panel, incs=incs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_scalar_ssr_returns_correct_keys(first_tenor_df):
    result = estimate_scalar_ssr(first_tenor_df, "1M")
    expected_keys = {"tenor", "beta", "se_hc3", "t_stat", "p_value", "r_squared", "n_obs"}
    assert set(result.keys()) == expected_keys


def test_beta_positive(full_results):
    for tenor, v in full_results["full_sample"].items():
        assert v["beta"] > 0, f"Expected positive beta for tenor {tenor}, got {v['beta']}"


def test_beta_decreases_with_tenor(full_results):
    """Beta should be monotonically decreasing from 1M to 24M."""
    fs = full_results["full_sample"]
    betas = [fs[t]["beta"] for t in TENOR_LABELS if t in fs]
    assert len(betas) >= 2, "Need at least two tenors to test monotonicity"
    for i in range(len(betas) - 1):
        assert betas[i] > betas[i + 1], (
            f"beta[{TENOR_LABELS[i]}]={betas[i]:.4f} should exceed "
            f"beta[{TENOR_LABELS[i+1]}]={betas[i+1]:.4f}"
        )


def test_rolling_ssr_returns_dataframe(first_tenor_df):
    roll = estimate_rolling_ssr(first_tenor_df, "1M", window=60)
    assert isinstance(roll, pd.DataFrame)
    assert "beta" in roll.columns
    assert "n_obs" in roll.columns
    assert len(roll) > 0, "Rolling result should contain at least one row"


def test_r_squared_between_0_and_1(full_results):
    for tenor, v in full_results["full_sample"].items():
        r2 = v["r_squared"]
        assert 0.0 <= r2 <= 1.0, f"R2 out of range for tenor {tenor}: {r2}"


def test_with_minimal_synthetic_data():
    """Verify estimator on fully controlled synthetic data with known beta."""
    rng = np.random.default_rng(0)
    n = 200
    true_beta = 1.50

    a1 = rng.normal(-0.12, 0.01, n)
    Delta_u = rng.normal(0, 0.012, n)
    Delta_a0 = true_beta * a1 * Delta_u + rng.normal(0, 5e-5, n)

    df = pd.DataFrame({
        "a1": a1,
        "Delta_u": Delta_u,
        "Delta_a0": Delta_a0,
    })

    result = estimate_scalar_ssr(df, "1M")
    assert result["beta"] == pytest.approx(true_beta, abs=0.05), (
        f"Expected beta≈{true_beta}, got {result['beta']:.4f}"
    )
    assert result["r_squared"] > 0.9, "R2 should be high for near-perfect DGP"
    assert result["n_obs"] == n
