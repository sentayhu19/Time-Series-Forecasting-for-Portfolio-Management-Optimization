from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

try:
    # Optional: use pmdarima if available and compatible
    from pmdarima import auto_arima as _pmd_auto_arima  # type: ignore
    _PMDARIMA_AVAILABLE = True
except Exception:
    _PMDARIMA_AVAILABLE = False


def fit_auto_arima(
    y: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    stepwise: bool = True,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    max_P: int = 2,
    max_D: int = 1,
    max_Q: int = 2,
    trace: bool = False,
    suppress_warnings: bool = True,
):
    """Fit an auto-ARIMA-like model on a univariate series.

    If pmdarima is available, use it. Otherwise, fall back to a statsmodels
    grid search over (p, d, q) minimizing AIC (non-seasonal).
    Returns a model object compatible with forecast_arima().
    """
    y = _to_series(y)

    if seasonal:
        # Fallback implementation below is non-seasonal only.
        # If pmdarima is available, we can honor seasonality.
        if _PMDARIMA_AVAILABLE:
            try:
                model = _pmd_auto_arima(
                    y,
                    seasonal=True,
                    m=m,
                    stepwise=stepwise,
                    max_p=max_p,
                    max_d=max_d,
                    max_q=max_q,
                    max_P=max_P,
                    max_D=max_D,
                    max_Q=max_Q,
                    trace=trace,
                    error_action="ignore",
                    suppress_warnings=suppress_warnings,
                    information_criterion="aic",
                )
                return model
            except Exception:
                # Fall back to non-seasonal grid search
                pass

    if _PMDARIMA_AVAILABLE:
        try:
            model = _pmd_auto_arima(
                y,
                seasonal=False,
                m=1,
                stepwise=stepwise,
                max_p=max_p,
                max_d=max_d,
                max_q=max_q,
                trace=trace,
                error_action="ignore",
                suppress_warnings=suppress_warnings,
                information_criterion="aic",
            )
            return model
        except Exception:
            # If pmdarima import/runtime fails (e.g., binary mismatch), use grid search
            pass

    # Statsmodels grid search (non-seasonal)
    best_aic = np.inf
    best_order: Tuple[int, int, int] | None = None
    best_res: ARIMAResults | None = None
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                if p == d == q == 0:
                    continue
                try:
                    res = ARIMA(y, order=(p, d, q)).fit()
                    aic = res.aic
                    if trace:
                        print(f"ARIMA({p},{d},{q}) AIC={aic:.2f}")
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        best_res = res
                except Exception:
                    continue
    if best_res is None:
        # As a last resort, try (1,1,1)
        best_res = ARIMA(y, order=(1, 1, 1)).fit()
        best_order = (1, 1, 1)
    if trace:
        print(f"Selected ARIMA{best_order} with AIC={best_aic:.2f}")
    return best_res


def forecast_arima(
    model,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Forecast steps ahead for either pmdarima or statsmodels model."""
    preds: np.ndarray
    if hasattr(model, "predict") and "n_periods" in model.predict.__code__.co_varnames:  # pmdarima
        preds = np.asarray(model.predict(n_periods=steps))
    else:
        # statsmodels ARIMAResults: use get_forecast
        try:
            fc = model.get_forecast(steps=steps)
            preds = np.asarray(fc.predicted_mean)
        except Exception:
            # Fallback to in-sample one-step-ahead predictions extended
            preds = np.asarray(model.forecast(steps=steps))
    s = pd.Series(preds)
    if index is not None:
        s.index = index
    return s.astype(float)


def forecast_arima_with_ci(
    model,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Forecast with confidence intervals for pmdarima or statsmodels.

    Returns (mean, lower, upper) as Series aligned to `index` if provided.
    """
    if _PMDARIMA_AVAILABLE and hasattr(model, "predict"):
        try:
            mean = np.asarray(model.predict(n_periods=steps))
            ci = np.asarray(model.predict(n_periods=steps, return_conf_int=True)[1])
            lower, upper = ci[:, 0], ci[:, 1]
            m = pd.Series(mean)
            l = pd.Series(lower)
            u = pd.Series(upper)
            if index is not None:
                m.index = index; l.index = index; u.index = index
            return m.astype(float), l.astype(float), u.astype(float)
        except Exception:
            pass

    # statsmodels path
    try:
        fc = model.get_forecast(steps=steps)
        mean = pd.Series(np.asarray(fc.predicted_mean))
        conf = fc.conf_int(alpha=alpha)
        # conf is a DataFrame with columns like 'lower y', 'upper y'
        lower = pd.Series(np.asarray(conf.iloc[:, 0]))
        upper = pd.Series(np.asarray(conf.iloc[:, 1]))
    except Exception:
        preds = pd.Series(np.asarray(model.forecast(steps=steps)))
        # No CI available; use NaNs
        mean, lower, upper = preds, pd.Series([np.nan]*steps), pd.Series([np.nan]*steps)

    if index is not None:
        mean.index = index; lower.index = index; upper.index = index
    return mean.astype(float), lower.astype(float), upper.astype(float)


def _to_series(y) -> pd.Series:
    if isinstance(y, pd.Series):
        return pd.to_numeric(y, errors="coerce").dropna()
    s = pd.Series(y)
    return pd.to_numeric(s, errors="coerce").dropna()
