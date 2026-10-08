# SPX SSR Term Structure – Agent Knowledge Base

## Project Overview
Seven-experiment research repository studying SPX smile transport dynamics using total implied variance
in forward log-moneyness coordinates. Two experiments are fully implemented (exp1 and exp2).

## Data Pipeline (`src/data_pipeline.py`)
- `build_panel(start, end, seed, force_synthetic)` → MultiIndex `(date, tenor)` DataFrame
  - Columns: `[u, a0, a1, a2, a3, R2, w_0..w_10]`
  - Primary source: Massive REST API (requires `MASSIVE_TOKEN` env var). Falls back to synthetic DGP.
  - `force_synthetic=True` bypasses the API (use in tests).
- `build_increments(panel)` → MultiIndex `(date, tenor)` DataFrame
  - Columns: `[a0, a1, a2, a3, u, Delta_a0, Delta_a1, Delta_a2, Delta_u, ...]`
  - Filters: R2 ≥ 0.95 and |Δu| ≥ 0.001
- `get_tenor_increments(incs, tenor)` → single-tenor DatetimeIndex DataFrame
- Constants: `TENOR_LABELS = ['1M','2M','3M','6M','9M','12M','24M']`, `K_GRID = log([0.85..1.15])`

## Experiment 1 (`src/exp1_ssr.py`)
**Scalar SSR Term Structure** – estimates beta for each tenor.

Regression (no intercept, HC3): `Δa0 = β * a1 * Δu + ε`

Key functions:
- `estimate_scalar_ssr(df, tenor)` → dict `{tenor, beta, se_hc3, t_stat, p_value, r_squared, n_obs}`
- `estimate_rolling_ssr(df, tenor, window=60)` → DataFrame `{beta, n_obs}` indexed by date; skips windows < 20 obs
- `run_exp1(panel, incs)` → `{full_sample: {tenor: ...}, rolling: {tenor: DataFrame}}`
  - Saves: `results/exp1_ssr_table.csv`, `results/exp1_beta_term_structure.png`, `results/exp1_rolling_beta.png`

Benchmarks (synthetic DGP, seed=42, 2021-07-12 to 2026-07-09):
- beta: ~[1.42, 1.37, 1.33, 1.21, 1.16, 1.12, 1.00] (1M→24M, monotone decreasing)
- R²: ~[0.97, 0.95, 0.95, 0.91, 0.89, 0.86, 0.79]
- N: 1213 per tenor

## Experiment 2 (`src/exp2_jets.py`)
**ATM Jet Hierarchy** – sequential forward-substitution for (β, η, ζ).

Stage 1: `Δa0 = β·a1·Δu + ε0`
Stage 2: `Y1 = Δa1 − β̂·a2·Δu = η·a1·Δu + ε1`
Stage 3: `Y2 = Δa2 − β̂·a3·Δu − 2η̂·a2·Δu = ζ·a1·Δu + ε2`

Key functions:
- `estimate_jets(df)` → dict `{beta, eta, zeta, beta_se, eta_se, zeta_se, n_obs}` (SEs are NaN until bootstrap)
- `pairs_bootstrap_jets(df, B=500, seed=42)` → dict `{se_beta, se_eta, se_zeta, cov_eta_zeta, wald_stat, wald_pvalue}`
  - Wald test: `W = θ'Σ⁻¹θ`, `θ=[η̂, ζ̂]`, chi²(2) → use `np.linalg.pinv` for near-singular Σ
- `run_exp2(panel, incs)` → `{estimates: {tenor: ...}, wald: {tenor: ...}}`
  - Saves: `results/exp2_jets_table.csv`, `results/exp2_jets_plot.png`

## Runner Scripts (`exp/`)
- `exp/run_exp1.py` and `exp/run_exp2.py` add project root to `sys.path` via `sys.path.insert`.
- Run from project root: `python exp/run_exp1.py`

## Tests (`tests/`)
All tests use `force_synthetic=True` – never touch the Massive API.
- `tests/test_exp1.py`: 6 tests (keys, beta positive, monotone decrease, rolling DataFrame, R² in [0,1], minimal synthetic)
- `tests/test_exp2.py`: 5 tests (keys, positive SEs, Wald ≥ 0, p-value in [0,1], known-DGP recovery)
- Run: `python -m pytest tests/test_exp1.py tests/test_exp2.py -v` (~8-11 seconds)

## Implementation Patterns
- All OLS regressions use `sm.OLS(y, X)` (no intercept; X already is `a1*Δu`)
- Results dir is `os.path.join(os.path.dirname(__file__), "..", "results")` – robust to CWD
- Plots use `matplotlib.use("Agg")` (non-interactive backend) imported before pyplot
- Bootstrap uses `np.random.default_rng(seed)` for reproducibility

## Experiment 3 (`src/exp3_local_vel.py`)
**Strike-Dependent Local Velocity Profile** – estimates v₀(k₀) at 7 expansion centers.

`K0_CENTERS = log([0.905, 0.928, 0.951, 0.975, 1.000, 1.025, 1.051])`

Key functions:
- `local_cubic_jets(k_grid, w_grid, k0)` → `(b0, b1, b2, r2)` – fits `w ≈ b0 + b1*(k−k0) + …`
- `build_local_velocity_data(panel, tenor, k0, skew_threshold=0.001)` → DataFrame `[date, Delta_b0, b1_du]`
  - Calls `build_increments(panel)` internally
  - Drops rows where |b1(t)| < 0.001
- `estimate_local_velocity(df_node)` → dict `{v0, se_hc3, t_stat, p_val}` (OLS, HC3, no intercept)
- `moving_block_bootstrap_cov(panel, tenor, k0_centers, B=500, block_length=5, seed=42)` → (7×7) ndarray
  - Pre-computes all (Delta_b0, b1_du) pairs, then samples `floor(T/block_length)` blocks with replacement
- `smooth_trend_wald_test(v_hat, k0_centers, Sigma)` → dict `{c1, c2, wald_stat, p_value}`
  - Fits `v0 = c_const + c1*k0 + c2*k0²`; Wald chi²(2) tests H0: c1=c2=0
- `run_exp3(panel=None, incs=None)` → dict `{tenor: {v_hat, se, Sigma, wald}}`
  - Saves: `results/exp3_local_velocity_table.csv`, `results/exp3_velocity_profiles.png`

Tests: `tests/test_exp3.py` (4 tests, ~3s combined)
- `test_local_cubic_jets_accuracy`: exact recovery on noise-free cubic
- `test_estimate_local_velocity_returns_valid_keys`: key set + finite values
- `test_smooth_trend_wald_test_shape`: key set + wald≥0 + p∈[0,1]
- `test_velocity_not_constant_with_known_data`: spread > 0.01 across k0 centers

## Experiment 4 (`src/exp4_forecast.py`)
**Out-of-Sample Forecast Comparison** – 5-fold blocked CV on tenors 1M, 3M, 6M, 12M.

Models (all OLS, no intercept, sequential estimator):
- `RW`: Δâ0=Δâ1=Δâ2=0
- `SSR`: β from `Δa0 ~ a1·Δu`; Δâ1=Δâ2=0
- `SSR+Skew`: β (stage 1), η from `(Δa1−β·a2·Δu) ~ a1·Δu`; Δâ2=0
- `Full`: β, η, ζ from `(Δa2−β·a3·Δu−2η·a2·Δu) ~ a1·Δu` (stage 3)

Smile forecast: `Δŵ(k) = Δâ0 + Δâ1·k + 0.5·Δâ2·k²` on 11 K_GRID nodes.

Key functions:
- `make_folds(df, n_folds=5)` → list of `(train_idx, test_idx)` – fold 0 has empty train (skipped in runner)
- `fit_rw_model / fit_ssr_model / fit_ssr_skew_model / fit_full_model(train)` → params dict
- `predict_model(params, model_name, test)` → DataFrame with `Delta_a0_hat, Delta_a1_hat, Delta_a2_hat, Delta_w_hat_0..10`
- `score_forecasts(predictions, actuals)` → dict `{rmse_a0, rmse_a1, rmse_a2, rmse_smile}`
- `run_exp4(panel=None, incs=None)` → nested dict `{tenor: {model: avg_scores}}`
  - Saves: `results/exp4_forecast_table.csv`, `results/exp4_forecast_rmse.png`

Tests: `tests/test_exp4.py` (4 tests, ~3s combined)
- `test_make_folds_correct_count`: len==5, no overlap, temporal ordering
- `test_random_walk_zero_forecast`: all hat columns == 0
- `test_full_model_beats_rw_on_synthetic`: rmse_smile(Full) < rmse_smile(RW)
- `test_score_forecasts_nonnegative_rmse`: all scores finite and ≥ 0

## Critical statsmodels pattern
When X is a **1-D numpy array**, `sm.OLS(y, X).fit().params` is a numpy ndarray, NOT a pandas Series.
Use `float(np.atleast_1d(result.params)[0])` – NOT `.iloc[0]`.
Same applies to `.bse`, `.tvalues`, `.pvalues`.

## Known Calibration
Synthetic DGP true values (see `data_pipeline.py`):
- `_TRUE_BETA`: 1.4375 → 1.0114 (1M→24M)
- `_TRUE_ETA`:  1.93 → -0.10
- `_TRUE_ZETA`: 2.80 → 0.10

## Experiment 5 (`src/exp5_source.py`)
**Transport-versus-Source Decomposition** — tests H0: s=0 in the regression
`Δw(k0,t) = c + s·Δu + v·b1(k0,t)·Δu + θ·Δt + ε` (HAC NW maxlags=5).

Pipeline: takes `panel` → interpolates each smile to 41 PCHIP fine nodes → computes local skew `b1`
via local cubic OLS (half-width=5) → HAC regression.

Key functions:
- `interpolate_smile_41nodes(k_quoted, w_quoted)` → `(k_fine, w_fine)` len-41 arrays; uses `PchipInterpolator`
- `local_skew_at_node(k_fine, w_fine, center_idx, half_width=5)` → float (c1 of local cubic)
- `build_fixednode_data(panel, tenor, k0, n_nodes=41)` → DataFrame `[Delta_w_k0, Delta_u, b1_du, Delta_t]`
- `estimate_transport_source(df_node)` → dict `{c, s, v, theta, t_c, t_s, t_v, t_theta, n_obs, p_s}`
  - Uses `sm.add_constant([Delta_u, b1_du])` (3-param: c, s, v); theta=NaN (absorbed into c)
- `find_smile_minimum(k_fine, b1_fine)` → k* or None; handles exact zeros via `b1[i]==0 and b1[i+1]>0`
- `run_tracked_minimum_test(panel, tenor)` → dict `{s_min, t_stat, p_value, n_obs}`
  - Uses Pchip derivative (not local cubic) for b1_fine efficiency
- `run_exp5(panel, incs)` → saves `results/exp5_fixednode_table.csv`, `results/exp5_tracked_min_table.csv`

**HAC call syntax**: `res.get_robustcov_results(cov_type='HAC', maxlags=5, use_t=True)` — NOT `cov_kwds={...}`

Typical results (1M synthetic): s ≈ -0.20 with |t_s| ≈ 12 (highly significant).
Tracked minimum: few obs for 3M because polynomial smile min often lies outside domain.

Tests: `tests/test_exp5.py` (4 tests, ~1.3s)

## Experiment 6 (`src/exp6_bootstrap.py`)
**Finite-Sample Bootstrap Validation** — parameter recovery study (S=500, B=150) and power study.

Sequential estimator (same 3-stage as Exp 2 but no-intercept OLS, HAC):
- Stage 0: β from `Δa0 ~ a1·Δu`
- Stage 1: η from `(Δa1−β·a2·Δu) ~ a1·Δu`
- Stage 2: ζ from `(Δa2−β·a3·Δu−2η·a2·Δu) ~ a1·Δu`

Key functions:
- `extract_residuals(df, beta, eta, zeta)` → `(N, 3)` ndarray `[eps0, eps1, eps2]`
- `simulate_synthetic_sample(df, residuals, beta_true, eta_true, zeta_true, rng, block_length=5)`
  - Moving-block bootstrap of residuals, fixed covariate path `[a1,a2,a3,Δu]`
- `run_parameter_recovery(df, tenor, S=500, B=150, seed=42)` → summary DataFrame
  - Uses `joblib.Parallel(n_jobs=-1, prefer='threads')` for outer loop
  - Inner B is pairs bootstrap (row resampling with replacement)
  - Returns cols: `[param, true_val, mean_est, bias, emp_std, mean_se, coverage]`
- `run_power_study(df_1m, sample_sizes, S=200, B=150, seed=42)` → DataFrame `[N, rejection_rate]`
  - Tests H0: η=0 at 5% level; uses HAC t-stat directly (not bootstrap SE)
- `run_exp6(panel, incs)` → saves `results/exp6_recovery_table.csv`, `results/exp6_power_table.csv`, `results/exp6_power_curve.png`

Typical results (1M, small S/B): near-zero bias, ~95% coverage, power≈1.0 for N≥50 (large η_true=1.93).

Tests: `tests/test_exp6.py` (4 tests, ~2.6s with S=20-80, B=10-30)

## Experiment 7 (`src/exp7_pde.py`)
**Numerical PDE Transport Illustration** — pure numerical, no market data.

PDE: `∂_u w = β ∂_k w`; Exact solution: `w(k,u) = w0(k + β·u)`.
Initial condition: SVI smile `w0(k) = a + b(ρ(k−m) + √((k−m)²+σ²))`.
Default params: `a=0.02, b=0.15, ρ=−0.4, m=0, σ=0.2, β=1.3`.

**Beam-Warming 2nd-order upwind scheme** (stable for CFL=0.4):
- β>0 (wave moves left): **forward stencil** `β·(−3w_i+4w_{i+1}−w_{i+2})/(2·dk)` at i=0..N−3
- β<0 (wave moves right): **backward stencil** `β·(3w_i−4w_{i-1}+w_{i-2})/(2·dk)` at i=2..N−1
- Inflow boundaries (rightmost 2 nodes for β>0) filled from exact solution at each sub-step.

**RK2 (Heun)**: `rk2_step(w, dt, dk, beta, w_boundary_func)` where `w_boundary_func()` is a
zero-argument closure returning the full-domain exact solution at `u+dt`.

Convergence study: L2 slope ≈ 1.994 on grids [151,301,601,1201].

Gatheral functional: `g[w](k) = (1−k·w_k/(2w))² − (w_k²/4)(1/w+1/4) + w_kk/2`
- g≥0 for no static arbitrage; both initial SVI and transported solution satisfy this.

Tests: `tests/test_exp7.py` (5 tests, ~0.7s)

## Test Suite Summary
- Total: 32 tests across 7 files, all pass in ~11s.
- `pytest tests/` from project root.
- All tests use `force_synthetic=True` or hard-coded parameters (no API needed).

## Master Runner
`exp/run_all.py` — runs all 7 experiments in sequence; imports `run_expN(panel, incs)` from each
`src/expN_*.py`; prints summary tables; saves all artefacts to `results/`.

## Final Empirical Results (synthetic DGP, seed=42)
- Exp 1 beta: [1.4248, 1.3750, 1.3329, 1.2072, 1.1577, 1.1189, 0.9986]  — monotone ✓
- Exp 2 Wald: all p≈0; η turns negative at 24M ✓
- Exp 3 Wald: strike constancy rejected at all 7 tenors ✓
- Exp 4: Full model beats RW; biggest gains in Δa₂ RMSE (3M–6M) ✓
- Exp 5: ATM source negative & significant; tracked-min source significant for 1M ✓
- Exp 6: bias≈0, coverage≈95%, power=1.0 for all N ≥ 50 (η_true large) ✓
- Exp 7: convergence slope=1.994 ≈ 2; Gatheral g≥0 ✓
