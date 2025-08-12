from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler

BUSINESS_FREQ = "B"


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df


def fill_missing_dates(df: pd.DataFrame, freq: str = BUSINESS_FREQ) -> pd.DataFrame:
    if df is None or df.empty:
        # Nothing to fill; return as-is
        return df
    df = ensure_datetime_index(df)
    if df.index.size == 0:
        return df
    start, end = df.index.min(), df.index.max()
    if pd.isna(start) or pd.isna(end):
        # Guard against NaT
        return df
    full_index = pd.date_range(start=start, end=end, freq=freq)
    return df.reindex(full_index)


def handle_missing(df: pd.DataFrame, method: Literal["drop", "ffill", "bfill", "interpolate"] = "ffill") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if method == "drop":
        return df.dropna()
    if method == "ffill":
        return df.ffill()
    if method == "bfill":
        return df.bfill()
    if method == "interpolate":
        return df.interpolate(method="time").ffill().bfill()
    raise ValueError(f"Unknown missing method: {method}")


def compute_returns(prices: pd.DataFrame, method: Literal["pct", "log"] = "pct") -> pd.DataFrame:
    if prices is None or prices.empty:
        return prices
    if method == "pct":
        return prices.pct_change().dropna(how="all")
    if method == "log":
        return np.log(prices).diff().dropna(how="all")
    raise ValueError("method must be 'pct' or 'log'")


def scale_features(
    X: pd.DataFrame,
    scaler: Literal["standard", "minmax", None] = None,
) -> Tuple[pd.DataFrame, object | None]:
    if X is None or X.empty or scaler is None:
        return X, None
    if scaler == "standard":
        sc = StandardScaler()
    elif scaler == "minmax":
        sc = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler")
    X_scaled = pd.DataFrame(sc.fit_transform(X), index=X.index, columns=X.columns)
    return X_scaled, sc


def extract_series(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Extract a single ticker column as a clean float series."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    if ticker not in prices.columns:
        raise KeyError(f"Ticker '{ticker}' not found in prices columns: {list(prices.columns)}")
    s = pd.to_numeric(prices[ticker], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s.astype(float)


def time_split_by_date(s: pd.Series, split_date: str) -> Tuple[pd.Series, pd.Series]:
    """Chronologically split a series at split_date (train: < split_date, test: >= split_date)."""
    if s is None or s.empty:
        return s, s
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    mask = s.index < pd.to_datetime(split_date)
    train, test = s[mask], s[~mask]
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty. Adjust split_date or dataset extent.")
    return train, test


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
