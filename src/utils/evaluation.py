from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_all(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)
    y_true = pd.to_numeric(y_true, errors="coerce").dropna()
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    if y_pred.index.equals(y_true.index):
        return y_true, y_pred.loc[y_true.index]
    # align by intersection of indices
    common_idx = y_true.index.intersection(y_pred.index)
    return y_true.loc[common_idx], y_pred.loc[common_idx]


def normal_prediction_interval(
    y_true: pd.Series,
    y_pred_in_sample: pd.Series,
    steps: int,
    z: float = 1.96,
    index: pd.DatetimeIndex | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Construct normal-approx prediction intervals using residual std.

    This provides an approximate CI for models without native intervals (e.g., LSTM).
    Uses in-sample residuals from a validation/test segment to estimate sigma.
    """
    y_true, y_pred_in_sample = _align(y_true, y_pred_in_sample)
    resid = (y_true - y_pred_in_sample).astype(float)
    sigma = float(resid.std(ddof=1)) if len(resid) > 1 else float(np.nan)
    lower = pd.Series([-z * sigma] * steps)
    upper = pd.Series([+z * sigma] * steps)
    if index is not None:
        lower.index = index
        upper.index = index
    return lower, upper
