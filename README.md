# SPX Scalar SSR Term Structure and Transport Dynamics

## Overview

This repository implements seven experiments studying the smile transport dynamics of SPX options
using total implied variance in forward log-moneyness coordinates. The experiments estimate the
classical sticky-strike-ratio (SSR) transport coefficient beta, the full ATM jet hierarchy
(beta, eta, zeta), strike-dependent velocity profiles, out-of-sample forecasting, and
transport-versus-source decompositions.

## Structure

```
repo/
├── data/           # Data loading scripts (Massive API)
├── src/            # Core implementation
│   ├── data_pipeline.py    # SPX smile data fetching and preprocessing
│   ├── smile_fitting.py    # Cubic total-variance smile fitting
│   ├── exp1_ssr.py         # Experiment 1: Scalar SSR term structure
│   ├── exp2_jets.py        # Experiment 2: ATM jet hierarchy
│   ├── exp3_local_vel.py   # Experiment 3: Strike-dependent velocity
│   ├── exp4_forecast.py    # Experiment 4: OOS forecast comparison
│   ├── exp5_source.py      # Experiment 5: Transport-source decomposition
│   ├── exp6_bootstrap.py   # Experiment 6: Bootstrap validation
│   └── exp7_pde.py         # Experiment 7: Numerical PDE illustration
├── exp/            # Experiment runner scripts
├── tests/          # pytest unit and integration tests
├── results/        # Output metrics, plots, RESULTS.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Experiments

| ID | Title |
|----|-------|
| exp_1 | Scalar SSR Term Structure Estimation |
| exp_2 | ATM Jet Hierarchy and Self-Similarity Rejection |
| exp_3 | Strike-Dependent Local Velocity Profile |
| exp_4 | Out-of-Sample Forecast Comparison |
| exp_5 | Transport-versus-Source Decomposition |
| exp_6 | Finite-Sample Bootstrap Validation |
| exp_7 | Numerical PDE Transport Illustration |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set MASSIVE_TOKEN=<your api key>
```

## Running Experiments

```bash
# Run all experiments
python exp/run_all.py

# Run individual experiments
python exp/run_exp1.py
python exp/run_exp2.py
python exp/run_exp3.py
python exp/run_exp4.py
python exp/run_exp5.py
python exp/run_exp6.py
python exp/run_exp7.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Data

SPX option data is downloaded from the Massive REST API. Set `MASSIVE_TOKEN` in your `.env` file.

- Sample period: 2021-07-12 through 2026-07-09
- Tenors: 1M, 2M, 3M, 6M, 9M, 12M, 24M
- Strike grid: K/F ∈ {0.85, 0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10, 1.15}

## Results

See `results/RESULTS.md` for a summary of all experiment findings.
