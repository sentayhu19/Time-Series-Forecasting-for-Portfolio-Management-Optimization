from datetime import datetime

# Core configuration values
TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-07-01"
END_DATE = "2025-07-31"
INTERVAL = "1d"
AUTO_ADJUST = True
TARGET_TICKER = "TSLA"
SPLIT_DATE = "2024-01-01"  # Train up to this date (exclusive), test from this date onward

# Finance constants
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02  # 2% annualized risk-free rate (approx)

# Plotting
STYLE = "seaborn-v0_8"

__all__ = [
    "TICKERS",
    "START_DATE",
    "END_DATE",
    "INTERVAL",
    "AUTO_ADJUST",
    "TARGET_TICKER",
    "SPLIT_DATE",
    "TRADING_DAYS_PER_YEAR",
    "RISK_FREE_RATE",
    "STYLE",
]
