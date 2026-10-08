"""
Unit tests for Experiment 4: Out-of-Sample Forecast Comparison.

All tests use synthetic data (force_synthetic=True) – no external API calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import K_GRID, build_panel, build_increments, get_tenor_increments
from src.exp4_forecast import (
    fit_full_model,
    fit_rw_model,
    fit_ssr_model,
    fit_ssr_skew_model,
    make_folds,
    predict_model,
    score_forecasts,
)

_N_GRID = len(K_GRID)  # 11


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tenor_df():
    """Single-tenor increment DataFrame for a short synthetic window."""
    panel = build_panel(start="2022-01-03", end="2023-12-29", force_synthetic=True)
    incs = build_increments(panel)
    return get_tenor_increments(incs, "3M")


# ---------------------------------------------------------------------------
# Test 1: make_folds returns exactly n_folds tuples
# ---------------------------------------------------------------------------


def test_make_folds_correct_count(tenor_df):
    """
    make_folds must return a list of exactly 5 (train_idx, test_idx) tuples.
    Each test_idx must be non-empty; together they must cover all indices.
    """
    folds = make_folds(tenor_df, n_folds=5)

    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

    for i, (train_idx, test_idx) in enumerate(folds):
        assert len(test_idx) > 0, f"Fold {i} has empty test set"
        # Train and test should be non-overlapping
        assert len(np.intersect1d(train_idx, test_idx)) == 0, (
            f"Fold {i}: train and test indices overlap"
        )
        # Train should come before test (temporal ordering)
        if len(train_idx) > 0:
            assert train_idx.max() < test_idx.min(), (
                f"Fold {i}: training data leaks into future"
            )

    # All test indices together should partition 0..T-1
    all_test = np.concatenate([t for _, t in folds])
    assert len(all_test) == len(tenor_df), "Test indices don't cover all observations"


# ---------------------------------------------------------------------------
# Test 2: random-walk model forecasts all zeros
# ---------------------------------------------------------------------------


def test_random_walk_zero_forecast(tenor_df):
    """
    The RW model must forecast Δa0 = Δa1 = Δa2 = 0 and therefore
    Δŵ(k) = 0 at every node.
    """
    params = fit_rw_model(tenor_df)
    assert params == {}, "RW model should return an empty param dict"

    preds = predict_model(params, "RW", tenor_df)

    np.testing.assert_array_equal(
        preds["Delta_a0_hat"].values,
        np.zeros(len(tenor_df)),
        err_msg="RW Δa0_hat should be zero",
    )
    np.testing.assert_array_equal(
        preds["Delta_a1_hat"].values,
        np.zeros(len(tenor_df)),
        err_msg="RW Δa1_hat should be zero",
    )
    np.testing.assert_array_equal(
        preds["Delta_a2_hat"].values,
        np.zeros(len(tenor_df)),
        err_msg="RW Δa2_hat should be zero",
    )
    for j in range(_N_GRID):
        np.testing.assert_array_equal(
            preds[f"Delta_w_hat_{j}"].values,
            np.zeros(len(tenor_df)),
            err_msg=f"RW Δŵ_{j} should be zero",
        )


# ---------------------------------------------------------------------------
# Test 3: full model achieves lower RMSE than RW on synthetic data
# ---------------------------------------------------------------------------


def test_full_model_beats_rw_on_synthetic(tenor_df):
    """
    On synthetic data generated from the true transport DGP, the full
    three-stage model should achieve strictly lower smile RMSE than the
    naïve random walk when both are evaluated in a held-out test fold.

    We use a single train/test split (80 / 20) to keep the test fast.
    """
    T = len(tenor_df)
    split = int(0.80 * T)
    train = tenor_df.iloc[:split]
    test = tenor_df.iloc[split:]

    assert len(train) >= 20 and len(test) >= 10, "Not enough data for this test"

    rw_params = fit_rw_model(train)
    full_params = fit_full_model(train)

    rw_preds = predict_model(rw_params, "RW", test)
    full_preds = predict_model(full_params, "Full", test)

    rw_scores = score_forecasts(rw_preds, test)
    full_scores = score_forecasts(full_preds, test)

    assert full_scores["rmse_smile"] < rw_scores["rmse_smile"], (
        f"Full model smile RMSE ({full_scores['rmse_smile']:.6f}) should be "
        f"less than RW ({rw_scores['rmse_smile']:.6f})"
    )
    assert full_scores["rmse_a0"] < rw_scores["rmse_a0"], (
        "Full model should beat RW on rmse_a0"
    )


# ---------------------------------------------------------------------------
# Test 4: score_forecasts returns non-negative RMSE values
# ---------------------------------------------------------------------------


def test_score_forecasts_nonnegative_rmse(tenor_df):
    """
    score_forecasts must return finite, non-negative RMSE for all metrics
    regardless of which model produced the predictions.
    """
    # Use SSR+Skew model for a non-trivial prediction
    folds = make_folds(tenor_df, n_folds=5)
    # Pick the last fold (largest training set)
    train_idx, test_idx = folds[-1]
    if len(train_idx) == 0:
        train_idx, test_idx = folds[-2][0], folds[-1][1]

    train = tenor_df.iloc[train_idx]
    test = tenor_df.iloc[test_idx]

    params = fit_ssr_skew_model(train)
    preds = predict_model(params, "SSR+Skew", test)
    scores = score_forecasts(preds, test)

    expected_keys = {"rmse_a0", "rmse_a1", "rmse_a2", "rmse_smile"}
    assert set(scores.keys()) == expected_keys, f"Unexpected keys: {set(scores.keys())}"

    for key, val in scores.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"
        assert val >= 0.0, f"{key} is negative: {val}"
