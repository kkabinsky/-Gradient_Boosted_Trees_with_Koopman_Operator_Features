# Gradient-Boosted Trees with Koopman Operator Features

Code and results for the paper

> Kabin Kanjamapornkul and Theepakorn Jithitikulchai,
> **An Early-Warning System for Stock Market Crashes: Gradient-Boosted Trees with Koopman Operator Features**,
> manuscript submitted for publication, 2026.

The paper builds a leakage-free early-warning system for large stock market
losses and tests whether features derived from the Koopman operator improve
gradient-boosted trees (XGBoost). The Koopman features come from a Koopman
autoencoder: 16 latent coordinates, their projections on the eigenvectors of
the learned linear operator, and a lagged one-step residual. Every
data-dependent step (scaling, the Koopman autoencoder and the XGBoost model) is
re-estimated on an expanding window using only information available at the
forecast origin.

This repository contains the programs that compute the tables and figures of
the paper and its supplementary material, the saved results (CSV and Excel),
and the figures.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── koopman_core.py          helper functions (technical predictors, Koopman autoencoder, XGBoost settings)
├── koopman_revision.py      comparison of predictor sets for next-day returns (Section 5.3)
├── replot_importance.py     Figure 4 with readable predictor names
├── koopman_ews.py           crash early-warning system (Section 5.5)
├── daily_backtest.py        daily XGBoost-LSTM backtest after transaction costs (Table 9)
├── gen_supp_tables.py       LaTeX bodies of Supplementary Tables S1-S8
├── make_excel_results.py    collects all results into Excel workbooks
├── dataset/                 input data (see dataset/README.md; not distributed)
└── results/
    ├── excel/               paper_tables.xlsx and one *_raw.xlsx workbook per experiment
    ├── csv/                 raw outputs of each experiment
    │   ├── predictor_sets/
    │   ├── early_warning_alpha0.10/
    │   ├── early_warning_alpha0.20/
    │   └── daily_backtest/
    ├── figures/             Figure 4, Figure 5 and Supplementary Figure S1 (.eps and .jpg)
    └── latex_tables/        Supplementary Tables S1-S8 as LaTeX table bodies
```

## Which program produces which result

Table and figure numbers refer to the submitted manuscript and its
supplementary material. The sheet names are those of
`results/excel/paper_tables.xlsx`.

| Paper | Content | Program | Output in `results/csv/` | Excel sheet |
|---|---|---|---|---|
| Table 4 | Number of predictors in each predictor set | `koopman_revision.py` | `predictor_sets/feature_counts_and_sample.csv` | `Table4_predictor_sets` |
| Table 5 | Hyperparameters | `koopman_core.py`, `koopman_revision.py`, `koopman_ews.py` (fixed in the code) | - | - |
| Table 9 | Daily backtest after transaction costs, with bootstrap intervals | `daily_backtest.py` | `daily_backtest/table_netcost_daily.csv` | `Table9_netcost_daily` |
| Table 10 | RMSE of next-day return forecasts and Diebold-Mariano tests | `koopman_revision.py` | `predictor_sets/rmse_table.csv`, `predictor_sets/dm_tests.csv` | `Table10_RMSE_DM` |
| Figure 4 | Gain importance, technical + Koopman predictor set | `koopman_revision.py`, then `replot_importance.py` | `predictor_sets/importance_F3_*.csv` | `Fig4_gain_importance` |
| Section 5.5 | Early-warning results (AUC, Brier skill score, hit and false-alarm rates, bootstrap tests), alpha = 0.10 | `koopman_ews.py` | `early_warning_alpha0.10/ews_performance.csv`, `ews_auc_tests.csv` | `Sec5.5_EWS_alpha0.10` |
| Figure 5 | Early-warning signals, alpha = 0.10 | `koopman_ews.py` | `early_warning_alpha0.10/ews_signals.csv` | - |
| Table S1 | List of technical predictors | `koopman_revision.py`, `gen_supp_tables.py` | `predictor_sets/F1_features_*.csv` | `S1_predictors` |
| Table S2 | Samples and Koopman autoencoder estimation | `koopman_revision.py`, `gen_supp_tables.py` | `predictor_sets/refit_log.csv` | `S2_refit_samples` |
| Table S3 | All Diebold-Mariano tests | `koopman_revision.py`, `gen_supp_tables.py` | `predictor_sets/dm_tests.csv` | `S3_DM_all` |
| Table S4 | Eigenvalues of the Koopman operator (final window) | `koopman_revision.py`, `gen_supp_tables.py` | `predictor_sets/koopman_eigenvalues_final_window.csv` | `S4_eigenvalues` |
| Table S5 | Loss thresholds of the warning event | `koopman_ews.py` (both alpha), `gen_supp_tables.py` | `early_warning_alpha*/ews_refit_log.csv` | `S5_EWS_thresholds` |
| Table S6 | Early-warning results, alpha = 0.20 | `koopman_ews.py --alpha 0.20`, `gen_supp_tables.py` | `early_warning_alpha0.20/ews_performance.csv` | `S6_EWS_alpha0.20` |
| Table S7 | Bootstrap tests of AUC differences | `koopman_ews.py` (both alpha), `gen_supp_tables.py` | `early_warning_alpha*/ews_auc_tests.csv` | `S7_EWS_AUC_tests` |
| Table S8 | Daily backtest by index, episode, model and window | `daily_backtest.py`, `gen_supp_tables.py` | `daily_backtest/per_dataset.csv` | `S8_backtest_per_dataset` |
| Figure S1 | Early-warning signals, alpha = 0.20 | `koopman_ews.py --alpha 0.20` | `early_warning_alpha0.20/ews_signals.csv` | - |

### Not included

* **Tables 6-8 and Figure 3** (gross returns and Sharpe ratios of XGBoost and
  LSTM on the intraday and daily data around the three stress episodes) come
  from an earlier program and intraday data sets that are not part of this
  repository. Table 9 and Table S8 are a separate re-implementation on the
  daily data, described in the paper.
* **Figures 1 and 2** (index levels, crash indicator and drawdown
  distributions) and the workflow diagram in the appendix.
* **Tables 1-3** are a literature summary and a description of the data.
* The interactive deployment through the OpenClaw agent (Appendix A). OpenClaw
  is third-party software and is not needed to reproduce any result.

## Installation

The results were produced with Python 3.14.4 on a CPU (20 threads), without a
GPU.

```bash
pip install -r requirements.txt
```

## Data

Place the three daily Excel files in `dataset/`; see
[`dataset/README.md`](dataset/README.md) for the expected format. The index
data were obtained from commercial data providers and are not distributed
here. They are available from the corresponding author on reasonable request.
The saved results in `results/` can be inspected without the data.

## Reproducing the results

Run the programs from the repository folder in this order. Each experiment
writes to `results/csv/`, `results/figures/` or `results/latex_tables/`, so the
saved results are overwritten.

```bash
# Section 5.3: Table 4, Table 10, Figure 4 (about 20 minutes)
python koopman_revision.py
python replot_importance.py

# Section 5.5: early-warning system, alpha = 0.10 (about 28 minutes)
python koopman_ews.py

# Supplementary Table S6 and Figure S1: alpha = 0.20 (about 28 minutes)
python koopman_ews.py --alpha 0.20 --output-dir results/csv/early_warning_alpha0.20 --fig-stem results/figures/fig_ews_signals_alpha20

# Table 9 and Supplementary Table S8 (about 5 minutes)
python daily_backtest.py

# Supplementary Tables S1-S8 (LaTeX) and the Excel workbooks (seconds)
python gen_supp_tables.py
python make_excel_results.py
```

`koopman_revision.py` and `koopman_ews.py` process the three indices in
parallel (`--jobs 3` by default). For a quick check that everything runs, use
for example

```bash
python koopman_revision.py --datasets dax --max-refits 1 --epochs 2 --output-dir results/smoke_test
```

All random seeds are fixed (42). Rerunning `koopman_revision.py` and
`daily_backtest.py` on the same machine and package versions reproduces the
saved results; a full rerun of both programs gave
numerically identical output files.

## Method summary

* **Target.** Next-day log return `r_{t+1}` for the return forecasts; for the
  early-warning system, the event that the index loses more than a threshold
  within the next 20 trading days. The threshold is the 90th (or 80th)
  percentile of the 20-day forward loss in the training window.
* **Predictors.** 15 lagged returns; technical predictors (lagged prices,
  moving averages, rolling volatility, RSI, rate of change, MACD, Bollinger
  bands and source-file columns known at day `t`); Koopman features; and
  technical plus Koopman features.
* **Koopman autoencoder.** Encoder and decoder with two hidden layers of 64
  units, 16 latent coordinates, linear latent dynamics `z_{t+1} = K z_t` with
  `K = Q diag(mu) Q^T` and `mu` in [0.5, 1.1], trained for 200 epochs with
  AdamW on the training window only.
* **Protocol.** Initial training window: first 80% of each series. The scaler,
  the autoencoder and XGBoost are re-estimated every 20 observations on an
  expanding window. For the early-warning system, training samples are purged
  of observations whose 20-day outcome is not yet known. Hyperparameters are
  fixed in advance and never tuned on test data.
* **Evaluation.** RMSE and Diebold-Mariano tests with the
  Harvey-Leybourne-Newbold correction for return forecasts; AUC, Brier skill
  score, hit rate, false-alarm rate, precision and moving-block bootstrap tests
  for the warnings; turnover, cumulative returns after one-way costs of 0, 1
  and 5 basis points, and stationary-bootstrap intervals for the trading rule.

## Notes on the results

* `koopman_ews.py` also computes a trailing 20-day realised-volatility signal.
  It is not reported in the paper, except that the Delta AUC column of
  Supplementary Table S6 is the difference from it. Its rows are therefore
  left out of `paper_tables.xlsx` and `results/latex_tables/`, but they remain
  in the raw CSV and `*_raw.xlsx` files.
* Buy-and-hold pays no transaction costs, so its three cost columns in Table 9
  describe the same return series. The paper reports the bootstrap interval for
  `c = 0` in all three columns. `table_netcost_daily.csv` contains a separate
  bootstrap draw for each cost level, so its intervals for `c = 1` and `c = 5`
  differ slightly.
* Dates in the DAX outputs are Buddhist-era `yyyymmdd` integers, as in the
  source file (for example `25650826` is 26 August 2022).

## Contact

Theepakorn Jithitikulchai (corresponding author), Thammasat University Research
Unit in Deep Learning and Advanced Econometrics, Faculty of Economics,
Thammasat University, Bangkok, Thailand. Email: theepakorn@econ.tu.ac.th
