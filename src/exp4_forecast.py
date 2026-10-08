"""
Experiment 4: Out-of-Sample Forecast Comparison.

Four models are evaluated on tenors 1M, 3M, 6M, 12M using five-fold
blocked time-series cross-validation.

Models
------
RW        : naïve random walk – all jet changes forecast as zero
SSR       : Δa0 = β·a1·Δu  (one-stage sequential estimator)
SSR+Skew  : SSR + Δa1 = β·a2·Δu + η·a1·Δu  (two-stage)
Full      : SSR+Skew + Δa2 = β·a3·Δu + 2η·a2·Δu + ζ·a1·Δu  (three-stage)

Smile forecast
--------------
Δŵ(k) = Δâ0 + Δâ1·k + ½·Δâ2·k²   evaluated on the 11-node K_GRID.

Evaluation metrics
------------------
RMSE for Δa0, Δa1, Δa2, and a composite smile RMSE averaged over all 11
strike nodes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

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

FORECAST_TENORS = ["1M", "3M", "6M", "12M"]
MODEL_NAMES = ["RW", "SSR", "SSR+Skew", "Full"]
RESULTS_DIR = Path("/workspace/project/results")
_N_GRID = len(K_GRID)  # 11


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------


def make_folds(df: pd.DataFrame, n_folds: int = 5) -> list:
    """
    Create n_folds contiguous blocked folds for time-series cross-validation.

    Splits the T observations into n_folds equal-sized blocks.  Fold i uses
    all observations before block i as training data and block i as the test
    set.  Fold 0 returns an empty training index (caller should skip it or
    treat the first block as training).

    Returns
    -------
    list of (train_idx, test_idx) tuples  – length == n_folds
    """
    T = len(df)
    fold_size = max(T // n_folds, 1)
    folds = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else T
        test_idx = np.arange(test_start, test_end)
        train_idx = np.arange(0, test_start)
        folds.append((train_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def fit_rw_model(train: pd.DataFrame) -> dict:
    """Fit the random-walk model. No parameters; forecasts all zeros."""
    return {}


def fit_ssr_model(train: pd.DataFrame) -> dict:
    """
    Stage 1: regress Δa0 = β·a1·Δu (OLS, no intercept).

    Returns dict with key 'beta'.
    """
    X = (train["a1"] * train["Delta_u"]).values
    y = train["Delta_a0"].values
    beta = float(np.atleast_1d(sm.OLS(y, X).fit().params)[0])
    return {"beta": beta}


def fit_ssr_skew_model(train: pd.DataFrame) -> dict:
    """
    Two-stage sequential estimator.

    Stage 1 : β   from Δa0 ~ a1·Δu
    Stage 2 : η   from (Δa1 − β·a2·Δu) ~ a1·Δu

    Returns dict with keys 'beta', 'eta'.
    """
    beta = fit_ssr_model(train)["beta"]
    y2 = (train["Delta_a1"] - beta * train["a2"] * train["Delta_u"]).values
    X2 = (train["a1"] * train["Delta_u"]).values
    eta = float(np.atleast_1d(sm.OLS(y2, X2).fit().params)[0])
    return {"beta": beta, "eta": eta}


def fit_full_model(train: pd.DataFrame) -> dict:
    """
    Three-stage sequential estimator.

    Stage 1 : β   from Δa0 ~ a1·Δu
    Stage 2 : η   from (Δa1 − β·a2·Δu) ~ a1·Δu
    Stage 3 : ζ   from (Δa2 − β·a3·Δu − 2η·a2·Δu) ~ a1·Δu

    Returns dict with keys 'beta', 'eta', 'zeta'.
    """
    params = fit_ssr_skew_model(train)
    beta, eta = params["beta"], params["eta"]
    y3 = (
        train["Delta_a2"]
        - beta * train["a3"] * train["Delta_u"]
        - 2.0 * eta * train["a2"] * train["Delta_u"]
    ).values
    X3 = (train["a1"] * train["Delta_u"]).values
    zeta = float(np.atleast_1d(sm.OLS(y3, X3).fit().params)[0])
    return {"beta": beta, "eta": eta, "zeta": zeta}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

_FIT_DISPATCH = {
    "RW": fit_rw_model,
    "SSR": fit_ssr_model,
    "SSR+Skew": fit_ssr_skew_model,
    "Full": fit_full_model,
}


def predict_model(
    params: dict,
    model_name: str,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate smile forecasts for a fitted model.

    Computes Δâ0, Δâ1, Δâ2 from `params` and then evaluates
        Δŵ(k) = Δâ0 + Δâ1·k + ½·Δâ2·k²
    at each of the 11 K_GRID nodes.

    Returns
    -------
    DataFrame indexed like `test`, columns:
        Delta_a0_hat, Delta_a1_hat, Delta_a2_hat,
        Delta_w_hat_0 … Delta_w_hat_10
    """
    n = len(test)
    du = test["Delta_u"].values

    # ----- jet forecasts -----
    if model_name == "RW":
        da0 = np.zeros(n)
        da1 = np.zeros(n)
        da2 = np.zeros(n)

    elif model_name == "SSR":
        beta = params["beta"]
        da0 = beta * test["a1"].values * du
        da1 = np.zeros(n)
        da2 = np.zeros(n)

    elif model_name == "SSR+Skew":
        beta, eta = params["beta"], params["eta"]
        da0 = beta * test["a1"].values * du
        da1 = beta * test["a2"].values * du + eta * test["a1"].values * du
        da2 = np.zeros(n)

    elif model_name == "Full":
        beta, eta, zeta = params["beta"], params["eta"], params["zeta"]
        da0 = beta * test["a1"].values * du
        da1 = beta * test["a2"].values * du + eta * test["a1"].values * du
        da2 = (
            beta * test["a3"].values * du
            + 2.0 * eta * test["a2"].values * du
            + zeta * test["a1"].values * du
        )

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    # ----- smile forecasts -----
    preds = pd.DataFrame(
        {"Delta_a0_hat": da0, "Delta_a1_hat": da1, "Delta_a2_hat": da2},
        index=test.index,
    )
    for j, k in enumerate(K_GRID):
        preds[f"Delta_w_hat_{j}"] = da0 + da1 * k + 0.5 * da2 * k ** 2

    return preds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_forecasts(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> dict:
    """
    Compute RMSE for each jet and composite smile RMSE.

    `actuals` must contain columns Delta_a0, Delta_a1, Delta_a2,
    w_t_0..w_t_10, w_t1_0..w_t1_10.

    Returns
    -------
    dict with keys: rmse_a0, rmse_a1, rmse_a2, rmse_smile
    """
    rmse_a0 = float(
        np.sqrt(np.mean((predictions["Delta_a0_hat"].values - actuals["Delta_a0"].values) ** 2))
    )
    rmse_a1 = float(
        np.sqrt(np.mean((predictions["Delta_a1_hat"].values - actuals["Delta_a1"].values) ** 2))
    )
    rmse_a2 = float(
        np.sqrt(np.mean((predictions["Delta_a2_hat"].values - actuals["Delta_a2"].values) ** 2))
    )

    # Actual smile changes: Delta_w_j = w_t1_j - w_t_j
    actual_dw = np.column_stack(
        [actuals[f"w_t1_{j}"].values - actuals[f"w_t_{j}"].values for j in range(_N_GRID)]
    )  # (T, 11)
    pred_dw = np.column_stack(
        [predictions[f"Delta_w_hat_{j}"].values for j in range(_N_GRID)]
    )  # (T, 11)
    rmse_smile = float(np.sqrt(np.mean((pred_dw - actual_dw) ** 2)))

    return {
        "rmse_a0": rmse_a0,
        "rmse_a1": rmse_a1,
        "rmse_a2": rmse_a2,
        "rmse_smile": rmse_smile,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_exp4(
    panel: Optional[pd.DataFrame] = None,
    incs: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Run blocked cross-validation for four tenors and four models.

    For each tenor, the increment dataset is split into five contiguous folds.
    Each model is fitted on the training portion and scored on the held-out
    test portion.  Scores are averaged across folds.

    Saves
    -----
    results/exp4_forecast_table.csv
    results/exp4_forecast_rmse.png
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if panel is None:
        panel = build_panel(force_synthetic=True)
    if incs is None:
        incs = build_increments(panel)

    all_results: dict = {}
    table_rows: list[dict] = []

    for tenor in FORECAST_TENORS:
        df_tenor = get_tenor_increments(incs, tenor)
        if len(df_tenor) < 10:
            continue

        folds = make_folds(df_tenor, n_folds=5)
        tenor_scores: dict[str, list[dict]] = {m: [] for m in MODEL_NAMES}

        for train_idx, test_idx in folds:
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train = df_tenor.iloc[train_idx]
            test = df_tenor.iloc[test_idx]

            for model_name in MODEL_NAMES:
                fit_fn = _FIT_DISPATCH[model_name]
                params = fit_fn(train)
                preds = predict_model(params, model_name, test)
                scores = score_forecasts(preds, test)
                tenor_scores[model_name].append(scores)

        # Average across folds
        tenor_avg: dict[str, dict] = {}
        for model_name in MODEL_NAMES:
            fold_list = tenor_scores[model_name]
            if not fold_list:
                continue
            avg = {
                k: float(np.mean([s[k] for s in fold_list])) for k in fold_list[0]
            }
            tenor_avg[model_name] = avg
            table_rows.append(
                {
                    "tenor": tenor,
                    "model": model_name,
                    **avg,
                }
            )

        all_results[tenor] = tenor_avg

    # Save CSV table
    pd.DataFrame(table_rows).to_csv(
        RESULTS_DIR / "exp4_forecast_table.csv", index=False
    )

    # Save plot
    _plot_rmse_comparison(all_results)

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    "RW": "#999999",
    "SSR": "#4477AA",
    "SSR+Skew": "#EE6677",
    "Full": "#228833",
}


def _plot_rmse_comparison(all_results: dict) -> None:
    metrics = ["rmse_a0", "rmse_a1", "rmse_a2", "rmse_smile"]
    metric_labels = ["RMSE Δa₀", "RMSE Δa₁", "RMSE Δa₂", "Smile RMSE"]

    tenors_present = [t for t in FORECAST_TENORS if t in all_results]
    n_tenors = len(tenors_present)
    n_models = len(MODEL_NAMES)
    x = np.arange(n_tenors)
    width = 0.18

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        for i, model in enumerate(MODEL_NAMES):
            vals = []
            for tenor in tenors_present:
                vals.append(all_results[tenor].get(model, {}).get(metric, float("nan")))
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=_MODEL_COLORS[model],
                label=model,
                alpha=0.85,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tenors_present)
        ax.set_xlabel("Tenor")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7)

    fig.suptitle("Experiment 4: Out-of-Sample Forecast RMSE", fontsize=12)
    plt.tight_layout()
    fig.savefig(
        RESULTS_DIR / "exp4_forecast_rmse.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
