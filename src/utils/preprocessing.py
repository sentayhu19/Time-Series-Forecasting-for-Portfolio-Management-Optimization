from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BUSINESS_FREQ = "B"


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def fill_missing_dates(df: pd.DataFrame, freq: str = BUSINESS_FREQ) -> pd.DataFrame:
    df = ensure_datetime_index(df)
    start, end = df.index.min(), df.index.max()
    full_index = pd.date_range(start=start, end=end, freq=freq)
    return df.reindex(full_index)


def handle_missing(df: pd.DataFrame, method: Literal["ffill", "bfill", "interpolate"] = "interpolate") -> pd.DataFrame:
    df = df.copy()
    if method == "ffill":
        return df.ffill()
    if method == "bfill":
        return df.bfill()
    return df.interpolate(method="time")


def compute_returns(prices: pd.DataFrame, kind: Literal["pct", "log"] = "pct") -> pd.DataFrame:
    prices = ensure_datetime_index(prices)
    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets


def scale_features(
    df: pd.DataFrame,
    method: Literal["standard", "minmax"] = "standard",
) -> Tuple[pd.DataFrame, object]:
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df.values), index=df.index, columns=df.columns)
    return scaled, scaler


__all__ = [
    "ensure_datetime_index",
    "fill_missing_dates",
    "handle_missing",
    "compute_returns",
    "scale_features",
]
