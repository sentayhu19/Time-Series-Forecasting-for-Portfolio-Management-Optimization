from __future__ import annotations
import yfinance as yf
import pandas as pd
from typing import Dict, List


def fetch_yfinance_data(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    auto_adjust: bool = True,
    progress: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical OHLCV data from yfinance for a list of tickers.

    Returns a dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Adj Close, Volume].
    """
    data: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, interval=interval, auto_adjust=auto_adjust, progress=progress)
        if df is None or df.empty:
            print(f"Warning: No data returned for {t}.")
            data[t] = pd.DataFrame()
        else:
            df.index = pd.to_datetime(df.index)
            # Normalize columns: if MultiIndex (field, ticker), keep only the field level.
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    # Common shape is 2 levels: (Field, Ticker)
                    field_level = 0
                    df.columns = df.columns.get_level_values(field_level)
                except Exception:
                    # Fallback: stringify each level and take the first non-empty as field
                    df.columns = [str(col[0]).strip() if isinstance(col, tuple) and len(col) > 0 else str(col) for col in df.columns]
            else:
                df.columns = [str(c).strip() for c in df.columns]
            data[t] = df
    return data


def merge_adjusted_close(data: Dict[str, pd.DataFrame], column: str = "Adj Close") -> pd.DataFrame:
    """Merge a price column per ticker into a wide DataFrame.

    Preference order per ticker: provided `column` if present, else 'Adj Close' if present,
    else fall back to 'Close'. This handles yfinance auto_adjust=True (which drops 'Adj Close').

    The resulting columns are the ticker symbols.
    """
    frames = []
    for t, df in data.items():
        if df is None or df.empty:
            continue
        col_to_use = None
        if column in df.columns:
            col_to_use = column
        elif "Adj Close" in df.columns:
            col_to_use = "Adj Close"
        elif "Close" in df.columns:
            col_to_use = "Close"
        if col_to_use is None:
            # Skip if no suitable column
            continue
        frames.append(df[[col_to_use]].rename(columns={col_to_use: t}))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=1).sort_index()
    return merged


__all__ = ["fetch_yfinance_data", "merge_adjusted_close"]
