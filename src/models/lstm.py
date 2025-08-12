from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_TF_AVAILABLE = True
try:
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    _TF_AVAILABLE = False


def make_sequences(values: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y  # add feature dim


essential_keras_kwargs = dict(optimizer="adam", loss="mse")


def build_lstm(input_shape: Tuple[int, int], units: int = 64, dropout: float = 0.0) -> keras.Model:
    if not _TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not installed. Install with: pip install tensorflow-cpu==2.16.1"
        )
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(units, return_sequences=False),
        layers.Dropout(dropout) if dropout > 0 else layers.Activation("linear"),
        layers.Dense(1),
    ])
    # If dropout == 0, the Activation layer is identity; harmless
    model.compile(**essential_keras_kwargs)
    return model


def train_lstm(
    train: pd.Series,
    lookback: int = 60,
    epochs: int = 20,
    batch_size: int = 32,
    units: int = 64,
    dropout: float = 0.0,
) -> Tuple[keras.Model, MinMaxScaler, np.ndarray]:
    if not _TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not installed. Install with: pip install tensorflow-cpu==2.16.1"
        )
    s = pd.to_numeric(train, errors="coerce").dropna().astype(float)
    values = s.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X_train, y_train = make_sequences(scaled.squeeze(), lookback)
    model = build_lstm((lookback, 1), units=units, dropout=dropout)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    last_window = scaled[-lookback:].squeeze()
    return model, scaler, last_window


def forecast_lstm(
    model: keras.Model,
    scaler: MinMaxScaler,
    last_window: np.ndarray,
    steps: int,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    window = last_window.copy()
    preds_scaled = []
    for _ in range(steps):
        x = window[-len(last_window) :].reshape(1, -1, 1)
        yhat_scaled = float(model.predict(x, verbose=0).squeeze())
        preds_scaled.append(yhat_scaled)
        window = np.append(window, yhat_scaled)
    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled_arr).squeeze()
    s = pd.Series(preds)
    if index is not None:
        s.index = index
    return s.astype(float)
