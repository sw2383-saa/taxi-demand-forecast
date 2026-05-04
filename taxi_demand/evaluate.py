"""Evaluation utilities for hourly NYC taxi-zone demand forecasts."""

import numpy as np
import pandas as pd

from taxi_demand.model import DemandForecaster


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error between y_true and y_pred."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error between y_true and y_pred."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def naive_baseline(df: pd.DataFrame) -> np.ndarray:
    """Return lag_24h column as forecast after dropping NaN rows.

    Rows with NaN in lag_1h or lag_24h are dropped to match the model's
    prediction contract. The returned forecast is a 1D numpy array.
    """
    eval_df = df.dropna(subset=["lag_1h", "lag_24h"])
    return eval_df["lag_24h"].to_numpy().ravel()


def evaluate_model(model: DemandForecaster, df: pd.DataFrame) -> dict:
    """Evaluate model and naive baseline on df.

    Drop rows where lag_1h or lag_24h is NaN before computing all metrics.
    Return a dictionary with exactly these keys: model_mae, model_rmse,
    baseline_mae, baseline_rmse.
    """
    eval_df = df.dropna(subset=["lag_1h", "lag_24h"])
    y_true = eval_df["demand"].to_numpy()
    model_pred = model.predict(eval_df)
    baseline_pred = naive_baseline(eval_df)

    return {
        "model_mae": mae(y_true, model_pred),
        "model_rmse": rmse(y_true, model_pred),
        "baseline_mae": mae(y_true, baseline_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
    }
