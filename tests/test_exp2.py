"""Pytest tests for Experiment 2: ATM Jet Hierarchy."""

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
from src.exp2_jets import estimate_jets, pairs_bootstrap_jets, run_exp2

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
def bootstrap_results(first_tenor_df):
    """Bootstrap results for 1M (module-scoped to avoid recomputation)."""
    return pairs_bootstrap_jets(first_tenor_df, B=200, seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_estimate_jets_returns_correct_keys(first_tenor_df):
    result = estimate_jets(first_tenor_df)
    expected_keys = {"beta", "eta", "zeta", "beta_se", "eta_se", "zeta_se", "n_obs"}
    assert set(result.keys()) == expected_keys


def test_pairs_bootstrap_returns_positive_se(bootstrap_results):
    for key in ("se_beta", "se_eta", "se_zeta"):
        val = bootstrap_results[key]
        assert np.isfinite(val), f"{key} should be finite, got {val}"
        assert val > 0, f"{key} should be positive, got {val}"


def test_wald_statistic_positive(bootstrap_results):
    w = bootstrap_results["wald_stat"]
    assert np.isfinite(w), f"wald_stat should be finite, got {w}"
    assert w >= 0, f"Wald statistic must be non-negative, got {w}"


def test_wald_pvalue_between_0_and_1(bootstrap_results):
    p = bootstrap_results["wald_pvalue"]
    assert np.isfinite(p), f"wald_pvalue should be finite, got {p}"
    assert 0.0 <= p <= 1.0, f"p-value out of [0,1]: {p}"


def test_sequential_substitution_with_known_data():
    """Verify the estimator recovers known (beta, eta, zeta) from synthetic DGP."""
    rng = np.random.default_rng(7)
    n = 500
    true_beta = 1.40
    true_eta = 1.80
    true_zeta = 2.50

    a1 = rng.normal(-0.13, 0.008, n)
    a2 = rng.normal(0.35, 0.010, n)
    a3 = rng.normal(-0.15, 0.005, n)
    Delta_u = rng.normal(0, 0.012, n)
    noise = 1e-5  # tiny noise to keep R2 high and estimates precise

    Delta_a0 = true_beta * a1 * Delta_u + rng.normal(0, noise, n)
    Delta_a1 = true_beta * a2 * Delta_u + true_eta * a1 * Delta_u + rng.normal(0, noise, n)
    Delta_a2 = (
        true_beta * a3 * Delta_u
        + 2.0 * true_eta * a2 * Delta_u
        + true_zeta * a1 * Delta_u
        + rng.normal(0, noise, n)
    )

    df = pd.DataFrame({
        "a1": a1, "a2": a2, "a3": a3,
        "Delta_u": Delta_u,
        "Delta_a0": Delta_a0,
        "Delta_a1": Delta_a1,
        "Delta_a2": Delta_a2,
    })

    result = estimate_jets(df)
    assert result["beta"] == pytest.approx(true_beta, abs=0.05), (
        f"Expected beta≈{true_beta}, got {result['beta']:.4f}"
    )
    assert result["eta"] == pytest.approx(true_eta, abs=0.10), (
        f"Expected eta≈{true_eta}, got {result['eta']:.4f}"
    )
    assert result["zeta"] == pytest.approx(true_zeta, abs=0.15), (
        f"Expected zeta≈{true_zeta}, got {result['zeta']:.4f}"
    )
    assert result["n_obs"] == n
