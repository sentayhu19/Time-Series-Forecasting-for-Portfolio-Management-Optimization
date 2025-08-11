from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

sns.set(style="whitegrid")


def plot_prices(prices: pd.DataFrame, title: str = "Adjusted Close Prices") -> None:
    if prices is None or prices.empty:
        raise ValueError("plot_prices: received empty DataFrame")
    # Ensure datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    # Coerce to numeric and keep only numeric columns
    numeric = prices.apply(pd.to_numeric, errors="coerce").select_dtypes(include=["number"]).dropna(how="all", axis=1)
    if numeric.empty:
        raise ValueError("plot_prices: no numeric data to plot after coercion")
    ax = numeric.plot(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()


def plot_returns(returns: pd.DataFrame, title: str = "Daily Returns") -> None:
    if returns is None or returns.empty:
        raise ValueError("plot_returns: received empty DataFrame")
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns = returns.copy()
        returns.index = pd.to_datetime(returns.index, errors="coerce")
    numeric = returns.apply(pd.to_numeric, errors="coerce").select_dtypes(include=["number"]).dropna(how="all", axis=1)
    if numeric.empty:
        raise ValueError("plot_returns: no numeric data to plot after coercion")
    ax = numeric.plot(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Return")
    plt.tight_layout()
    plt.show()


def plot_rolling(mean_df: pd.DataFrame, std_df: pd.DataFrame, column: str, title: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    mean_df[column].plot(ax=ax, label=f"Rolling Mean ({column})")
    std_df[column].plot(ax=ax, label=f"Rolling Std ({column})")
    ax.set_title(title or f"Rolling Stats - {column}")
    ax.legend()
    plt.tight_layout()
    plt.show()


__all__ = ["plot_prices", "plot_returns", "plot_rolling"]
