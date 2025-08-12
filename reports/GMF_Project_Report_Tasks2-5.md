# GMF Investments: Time Series Forecasting and Portfolio Optimization (Tasks 2–5)

Author: GMF Analytics Team  
Repo: `sentayhu19/Time-Series-Forecasting-for-Portfolio-Management-Optimization`

## Executive Summary
- Built, evaluated, and compared ARIMA/SARIMA and LSTM models for TSLA forecasting with strict chronological splits.
- Forecasted TSLA 6–12 months ahead with confidence intervals and interpretation guidance.
- Constructed Efficient Frontier for TSLA/BND/SPY using Modern Portfolio Theory (MPT), selected an optimal portfolio, and saved weights/metrics.
- Backtested the strategy over the last year against a 60/40 SPY/BND benchmark and saved plots/metrics.

All code is modular under `src/` and reproducible via the provided notebooks in `notebooks/` and artifacts in `data/processed/` and `reports/figures/`.

---

## Data & Scope
- Tickers: `TSLA`, `BND`, `SPY`  
- Period: `START_DATE` to `END_DATE` from `src/constants/config.py`  
- Frequency: Business days  
- Loader utilities: `src/utils/data_loader.py` with robust yfinance fetching and column handling.  
- Preprocessing utilities: `src/utils/preprocessing.py` (fill business dates, handle missing, series extraction, chronological splits).

Data folders:
- Raw/processed artifacts stored under `data/` subfolders.
- Figures stored under `reports/figures/`.

---

## Task 2: Model Development and Evaluation (TSLA)
Artifacts: `notebooks/Task_02_Forecasting_TSLA.ipynb`

- Models:  
  - ARIMA/SARIMA: `src/models/arima.py` supports `pmdarima` auto-ARIMA; falls back to `statsmodels` grid search to avoid binary incompatibility.  
  - LSTM: `src/models/lstm.py` builds and trains a Keras LSTM. Graceful error if TensorFlow missing.
- Evaluation: `src/utils/evaluation.py` with MAE, RMSE, MAPE and series alignment.
- Split: Chronological split using `time_split_by_date()` with `SPLIT_DATE` in `src/constants/config.py`.
- Outputs: Metrics printed in the notebook; plots saved as needed.

Key implementation notes:
- Optional seasonal SARIMA via `seasonal=True, m=5` (business week) when `pmdarima` is available.  
- LSTM hyperparameters configurable (lookback, epochs, units, dropout).

---

## Task 3: Future Forecasts with CIs (TSLA)
Artifacts:
- Notebook: `notebooks/Task_03_Forecast_TSLA.ipynb`
- Figures: `reports/figures/tsla_task3_forecast_6m.png` (and similar)
- CSV: `data/processed/tsla_task3_forecasts_6m.csv`

Highlights:
- Forecast horizon selectable (e.g., 6 or 12 months) using business days.  
- ARIMA forecasts with confidence intervals via `forecast_arima_with_ci()`.
- LSTM forecasts optional; approximate CIs derived from residual std (`normal_prediction_interval`).
- Interpretation guidance included (trend, band widening, uncertainty, opportunities/risks).

---

## Task 4: Portfolio Optimization (MPT)
Artifacts:
- Notebook: `notebooks/Task_04_Portfolio_Optimization.ipynb`
- Figures: `reports/figures/task4_efficient_frontier.png`
- CSVs:  
  - `data/processed/task4_portfolio_summary.csv`  
  - `data/processed/task4_portfolio_weights.csv`

Methodology:
- Expected returns:  
  - TSLA from Task 3 forecast (preferred `LSTM_mean` else `ARIMA_mean`), annualized from projected daily returns.  
  - BND, SPY from historical mean daily returns (annualized).
- Covariance: Historical daily return covariance (annualized), default 3-year lookback if available.
- Optimization: PyPortfolioOpt Efficient Frontier  
  - Max Sharpe (tangency) portfolio  
  - Minimum Volatility portfolio
- Deliverables: Frontier plot with markers; saved weights and performance summary.

---

## Task 5: Strategy Backtesting
Artifacts:
- Notebook: `notebooks/Task_05_Backtest.ipynb`
- Figure: `reports/figures/task5_backtest_cum.png`
- CSV: `data/processed/task5_backtest_metrics.csv`

Setup:
- Backtest window: last ~252 business days of available data.
- Strategy: Hold Task 4 optimal (default Max Sharpe) weights throughout the period (simple hold).  
- Benchmark: 60% SPY / 40% BND static portfolio.

Evaluation:
- Cumulative returns plotted for strategy vs benchmark.
- Metrics saved: annualized return, volatility, Sharpe ratio, and total return for both.

Notes:
- Simplified backtest; no transaction costs or monthly re-optimization (can be added).
- For monthly rebalancing, iterate over month-ends to reset weights.

---

## Reproducibility & How to Run
1. Environment:  
   - See `requirements.txt`.  
   - On Windows, `tensorflow-cpu` is conditionally included. If LSTM is desired, ensure TensorFlow is installed and restart the kernel.
2. Run notebooks in order:  
   - `notebooks/Task_02_Forecasting_TSLA.ipynb`  
   - `notebooks/Task_03_Forecast_TSLA.ipynb`  
   - `notebooks/Task_04_Portfolio_Optimization.ipynb`  
   - `notebooks/Task_05_Backtest.ipynb`
3. Outputs:  
   - Forecast CSVs in `data/processed/`  
   - Portfolio weights/summary in `data/processed/`  
   - Backtest metrics in `data/processed/`  
   - Plots in `reports/figures/`

---

## Limitations & Next Steps
- Seasonality: Consider SARIMA grid with `statsmodels` if `pmdarima` unavailable.
- Stationarity: Consider modeling returns or log-returns with reconstruction.
- LSTM: Try multiple layers, dropout, and hyperparameter search; use walk-forward validation.
- Risk modeling: Explore covariance shrinkage, CVaR optimization, and constraints (e.g., max TSLA weight, turnover limits).
- Backtesting: Add transaction costs, monthly re-optimization, and walk-forward rolling retrain.
- Reporting: Parameter logs and experiment tracking (e.g., MLflow) for traceability.

---

## File Map (Key Components)
- `src/constants/config.py` — global settings, including `TARGET_TICKER`, `SPLIT_DATE`.
- `src/utils/data_loader.py` — yfinance fetch and merge utilities.
- `src/utils/preprocessing.py` — date filling, missing handling, splits, scaling helpers.
- `src/utils/evaluation.py` — MAE, RMSE, MAPE, and `normal_prediction_interval()` helper.
- `src/models/arima.py` — auto-ARIMA or statsmodels fallback; `forecast_arima_with_ci()`.
- `src/models/lstm.py` — Keras LSTM with graceful TensorFlow import handling.
- `notebooks/Task_02_Forecasting_TSLA.ipynb` — model training/evaluation.
- `notebooks/Task_03_Forecast_TSLA.ipynb` — 6–12 month forecasts with CIs.
- `notebooks/Task_04_Portfolio_Optimization.ipynb` — Efficient Frontier and portfolio selection.
- `notebooks/Task_05_Backtest.ipynb` — strategy vs 60/40 benchmark backtest.
- `reports/figures/` — plots for forecasting, frontier, and backtest.
- `data/processed/` — CSV outputs for forecasts, portfolio weights/summary, and backtest metrics.

---

## Conclusion
We delivered a modular, reproducible workflow that:
- Forecasts TSLA prices using classical and deep learning models with proper evaluation and CIs.
- Converts forecasts into investable insights via MPT to build optimal portfolios.
- Validates the strategy against a benchmark through an interpretable backtest.

Extend with walk-forward rebalancing, constraints, and robust risk models to further harden the approach for production use.
