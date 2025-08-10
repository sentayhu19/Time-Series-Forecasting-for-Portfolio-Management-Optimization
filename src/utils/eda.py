from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from statsmodels.tsa.stattools import adfuller


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T


def rolling_stats(df: pd.DataFrame, window: int = 21) -> Dict[str, pd.DataFrame]:
    mean_df = df.rolling(window).mean()
    std_df = df.rolling(window).std()
    return {"mean": mean_df, "std": std_df}


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    s = series.dropna()
    z = (s - s.mean()) / s.std(ddof=0)
    flags = np.abs(z) > threshold
    return pd.DataFrame({"value": s, "z": z, "is_outlier": flags})


def adf_test(series: pd.Series) -> Dict[str, float]:
    s = series.dropna()
    result = adfuller(s)
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "crit_1%": result[4]["1%"],
        "crit_5%": result[4]["5%"],
        "crit_10%": result[4]["10%"],
    }


__all__ = ["basic_stats", "rolling_stats", "detect_outliers_zscore", "adf_test"]
