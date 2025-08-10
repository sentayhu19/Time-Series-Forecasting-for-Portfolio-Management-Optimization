from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal

TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: float = 0.02,
    freq: Literal["daily", "weekly", "monthly"] = "daily",
) -> float:
    if isinstance(returns, pd.DataFrame):
        r = returns.mean(axis=1)
    else:
        r = returns
    rf_per_period = {
        "daily": (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1,
        "weekly": (1 + risk_free_rate) ** (1 / 52) - 1,
        "monthly": (1 + risk_free_rate) ** (1 / 12) - 1,
    }[freq]
    excess = r - rf_per_period
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    scale = {"daily": np.sqrt(TRADING_DAYS_PER_YEAR), "weekly": np.sqrt(52), "monthly": np.sqrt(12)}[freq]
    return float(mu / sigma * scale)


def value_at_risk_historic(returns: pd.Series, alpha: float = 0.95) -> float:
    return float(np.percentile(returns.dropna(), (1 - alpha) * 100))


def value_at_risk_parametric(returns: pd.Series, alpha: float = 0.95) -> float:
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    from scipy.stats import norm

    return float(mu + sigma * norm.ppf(1 - alpha))


__all__ = ["sharpe_ratio", "value_at_risk_historic", "value_at_risk_parametric"]
