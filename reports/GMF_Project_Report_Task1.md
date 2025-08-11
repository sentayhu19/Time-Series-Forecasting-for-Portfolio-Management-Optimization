# Time Series Forecasting for Portfolio Management Optimization

Prepared by: Senathu Berhnau  
Organization: Guide Me in Finance (GMF) Investments  
Date: 2025-08-10

## Executive Summary
GMF Investments seeks to enhance portfolio management through robust time series forecasting and risk analysis across a representative triad of assets: TSLA (growth equity), BND (investment-grade bond ETF), and SPY (broad U.S. equity market ETF). In Task 1, we established a clean, modular analytics foundation to source high-quality market data (2015-07-01 to 2025-07-31), ensure data integrity (type checks, missing-value handling, and time indexing), and perform a focused exploratory data analysis (EDA). We computed daily returns, rolling volatility, stationarity diagnostics (ADF), and foundational risk metrics (Sharpe Ratio and Value at Risk). This groundwork enables sound model development (ARIMA/LSTM) and portfolio optimization in subsequent tasks.

From a business perspective, we adopt EMH-aware expectations: pure price-level prediction is challenging. Instead, we emphasize stable and interpretable features, such as returns and volatility dynamics, that inform asset allocation and risk management. The outputs of Task 1 guide investment judgment by highlighting each asset’s risk/return profile, volatility clustering, and distributional properties. These insights support downstream forecasting and portfolio construction (e.g., efficient frontier analysis via PyPortfolioOpt) and provide a transparent, repeatable analytics path to informed portfolio decisions.

---

## Business Objective
Leverage time series forecasting and risk analytics to inform asset allocation decisions, minimize downside risk, and improve risk-adjusted returns for GMF’s client portfolios. Models and metrics serve as inputs into a broader decision framework rather than standalone price predictors.

## Situational Overview (Business Need)
- Translate market data into actionable insights for portfolio management.  
- Emphasize volatility characterization, momentum cues, and stationarity-aware modeling.  
- Build a maintainable and auditable data and analytics stack that scales to additional assets and factors.

## Data
- Source: yfinance API.  
- Assets: TSLA, BND, SPY.  
- Window: 2015-07-01 to 2025-07-31.  
- Fields: Date, Open, High, Low, Close, Adj Close, Volume.  
- Usage: Focus on adjusted close (or close fallback) for returns; OHLCV retained for completeness.

## Expected Outcomes (Skills & Knowledge)
- Data Wrangling: time-based indexing, missing handling, normalization.  
- Feature Engineering: daily pct/log returns, rolling statistics.  
- Statistical Modeling Readiness: stationarity testing for ARIMA/SARIMA.  
- Deep Learning Readiness: scaled features for LSTM.  
- Risk & Performance: Sharpe Ratio, VaR; context on EMH, stationarity, and interpretation.  
- Portfolio Optimization Readiness: prerequisites for efficient frontier analysis (PyPortfolioOpt).  
- Backtesting Readiness: clean returns time series for simulation and benchmarking.

## Team
- Tutors: Mahlet, Rediet, Kerod, Rehmet.

## Key Dates
- Case Discussion: Wed, 06 Aug 2025 (#all-week11).  
- Interim Solution: Sun, 10 Aug 2025, 20:00 UTC.  
- Final Submission: Tue, 12 Aug 2025, 20:00 UTC.

## Instructions Summary
- Objective: Prepare clean data, conduct EDA, quantify volatility, assess stationarity, and compute baseline risk metrics for TSLA/BND/SPY.  
- Tools: pandas, numpy, yfinance, matplotlib/seaborn/plotly, statsmodels/pmdarima, scikit-learn, PyPortfolioOpt.

## Tutorials Schedule / Submission
- Follow cohort schedule for walkthroughs and Q&A.  
- Submit notebook outputs, code, and report according to program guidelines.

---

## Professional Folder Structure and Artifacts
- `requirements.txt`: unified dependencies.  
- `src/`: modular Python package.  
  - `src/constants/config.py`: tickers, date range, RF rate, plotting style, yfinance settings.  
  - `src/utils/data_loader.py`: `fetch_yfinance_data()`, `merge_adjusted_close()` with robust Adj Close/Close handling.  
  - `src/utils/preprocessing.py`: time index normalization, missing handling, returns, scaling.  
  - `src/utils/eda.py`: `basic_stats`, rolling stats, outlier detection, ADF test.  
  - `src/utils/metrics.py`: Sharpe, VaR (historic/parametric).  
  - `src/utils/plotting.py`: price/returns and rolling volatility plots.  
- `notebooks/Task_01_Preprocess_and_Explore.ipynb`: complete Task 1 workflow.  
- `data/{raw,interim,processed,external}/`: reproducible data lifecycle.  
- `reports/figures/`: output charts and visuals.

---

## Task 1: Detailed Methodology and Findings

### 1) Data Acquisition & Integrity
- Pulled OHLCV via yfinance for TSLA, BND, SPY with daily interval.  
- Canonicalized DateTime index; standardized column names.  
- For prices, preferred `Adj Close` (if available), else `Close` to ensure continuity when `auto_adjust=True`.

### 2) Cleaning & Preprocessing
- Reindexed to a business-day calendar to expose missing sessions explicitly.  
- Filled gaps via time interpolation (or forward/backward fill as options).  
- Verified dtypes, null counts, and summary statistics.  
- Computed returns: percentage (default) and optional log returns for modeling stability.

### 3) Exploratory Data Analysis
- Visualized adjusted close trajectories to qualitatively assess regimes and drawdowns.  
- Inspected return distributions and tails (descriptive stats with custom percentiles).  
- Computed rolling mean and standard deviation (21-day) to quantify short-term volatility and potential clustering.  
- Performed z-score outlier detection on TSLA daily returns to flag extreme moves for contextual review.

### 4) Stationarity Diagnostics
- Conducted ADF tests on price levels (typically non-stationary) and returns (generally closer to stationary).  
- Interpreted p-values vs. 1%, 5%, and 10% critical values.  
- Established differencing requirements for ARIMA/SARIMA readiness where needed.

### 5) Risk Metrics
- Calculated Sharpe Ratio (daily aggregation, annualized) using an RF proxy from `config.py`.  
- Estimated Value at Risk (95%): historical percentile and parametric normal approximation.  
- Contextualized TSLA (higher volatility, potential drawdowns) vs. BND (stability) vs. SPY (diversified core).

### Key Insights (Qualitative)
- Price levels exhibit trend and structural breaks; returns are more stationary and modeling-friendly.  
- Volatility is time-varying with clustering; TSLA shows larger dispersion than SPY and BND.  
- Outlier detection highlights event-driven extremes, reinforcing the need for robust risk controls.  
- Baseline Sharpe/VaR provide comparative risk-return framing for allocation decisions.

> Note: Exact numeric results depend on the latest data pull; the notebook computes and prints them reproducibly.

---

## Reproducibility & How to Run
1. Install dependencies: `pip install -r requirements.txt`.  
2. Open and run `notebooks/Task_01_Preprocess_and_Explore.ipynb` sequentially.  
3. Generated figures can be saved/exported to `reports/figures/` as needed.

## Next Steps (Preview)
- Modeling: ARIMA/SARIMA with stationarity-aware differencing; LSTM with scaled sequences.  
- Evaluation: rolling/blocked validation; error metrics and forecast diagnostics.  
- Portfolio: Efficient Frontier via PyPortfolioOpt; sensitivity to constraints and risk models.  
- Backtesting: simple allocation rules using forecast signals; benchmark vs. SPY.

---

## Methodological Details and Formulas

- Returns definitions (for price series P_t):  
  - Daily percentage return: r_t = (P_t − P_{t−1}) / P_{t−1}  
  - Daily log return: r_t^{log} = ln(P_t) − ln(P_{t−1})  
  - Choice: percentage returns by default for interpretability; log returns are available for modeling stability.

- Rolling statistics (window w, default 21 business days):  
  - Rolling mean: μ_t = mean(r_{t−w+1} … r_t)  
  - Rolling volatility (std): σ_t = std(r_{t−w+1} … r_t)

- ADF stationarity test (null H0: unit root / non-stationary):  
  - If p-value < α (e.g., 0.05), reject H0 → series likely stationary.  
  - Implication: price levels often non-stationary; differencing needed for ARIMA; returns usually closer to stationary.

- Sharpe Ratio (annualized):  
  - r_f,period = (1 + r_f,annual)^{1/k} − 1, where k = 252 (daily), 52 (weekly), or 12 (monthly).  
  - Excess return: e_t = r_t − r_f,period.  
  - Sharpe = mean(e_t) / std(e_t) × sqrt(k).

- Value at Risk (VaR, α = 95% unless noted):  
  - Historical: VaR_α = percentile(r, 1 − α).  
  - Parametric (Gaussian): VaR_α = μ + σ × Φ^{−1}(1 − α), where μ, σ are sample mean/std of returns.

## Data Quality, Assumptions, and Limitations

- Data source and coverage: yfinance daily data (2015-07-01 to 2025-07-31). Trading holidays and corporate actions are implicitly handled by `auto_adjust` (if enabled); we fall back to Close if Adj Close absent.  
- Missing data: rare but possible due to exchange closures/API hiccups; we interpolate time-wise by default (alternatives: ffill/bfill).  
- Survivorship and look-ahead: using current tickers; historical composition of indices handled by ETF providers; we avoid look-ahead by sequential computation and no future leakage.  
- EMH context: we refrain from overfitting price levels; features emphasize returns/volatility.  
- Normality caveat: VaR parametric assumes normality; heavy tails lead to conservative use; historical VaR used as complementary measure.  
- Scaling: only applied when required for model classes (e.g., LSTM); statistics computed on original scale unless stated.

## Governance and Reproducibility

- Versioning: Git branches (e.g., `task-1`) and atomic commits.  
- Environment: `requirements.txt` pinned to compatible ranges; warning noted for `scikit-learn` vs `sklearn-compat`. Can pin to 1.6.x if needed.  
- Structure: modular `src/`, data lifecycle folders under `data/`, and outputs under `reports/figures/`.  
- Determinism: where applicable, set random seeds during modeling tasks; Task 1 is deterministic except for plotting randomness (none here).  
- Line endings: Windows CRLF warnings acknowledged; Git will normalize as configured (.gitattributes optional).  
- Documentation: this report plus the executable notebook provide a full audit trail.

## Validation and Benchmarking Plan

- Baselines:  
  - Random walk / naive (r̂_t = 0 for next period).  
  - Historical mean or moving-average forecasts as simple comparators.  
- Error metrics: MAE, RMSE, MAPE (as appropriate), coverage for interval forecasts.  
- Robustness: rolling-origin evaluation to avoid look-ahead; blocked time-series splits.  
- Benchmarks: portfolio comparisons vs. SPY buy-and-hold, and risk-adjusted returns (Sharpe) vs. risk-free.

## Output Artifacts

- Cleaned price panel (TSLA, BND, SPY) and daily returns time series.  
- Rolling statistics tables and plots (means/stds).  
- ADF test summaries for prices and returns.  
- Outlier tables for extreme daily moves (TSLA highlighted).  
- Risk metrics snapshot: Sharpe and VaR estimates per asset.  
- Figures exportable to `reports/figures/`.

## Glossary

- Adjusted Close: price adjusted for dividends/splits.  
- Stationarity: statistical properties (mean/variance/autocorrelation) constant over time.  
- ADF Test: Augmented Dickey-Fuller unit-root test for stationarity.  
- VaR: Value at Risk; loss threshold not exceeded with confidence α over a horizon.  
- Sharpe Ratio: risk-adjusted return measure vs. risk-free rate.  
- Efficient Frontier: set of portfolios with maximal expected return for a given risk level.

## References
- yfinance Documentation: https://github.com/ranaroussi/yfinance  
- Statsmodels: https://www.statsmodels.org/  
- pmdarima: https://alkaline-ml.com/pmdarima/  
- PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/  
- EMH Overview: Fama, Eugene F. “Efficient Capital Markets.” Journal of Finance (1970).
