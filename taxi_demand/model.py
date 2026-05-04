"""Model utilities for hourly NYC taxi-zone pickup demand forecasting."""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class DemandForecaster:
    """Linear-regression forecaster using lagged hourly demand features."""

    def fit(self, df: pd.DataFrame) -> None:
        """Train on df.

        Drop rows where lag_1h or lag_24h is NaN. The model uses
        [lag_1h, lag_24h] as features and demand as the target. The fitted
        estimator is sklearn.linear_model.LinearRegression.
        """
        train_df = df.dropna(subset=["lag_1h", "lag_24h"])
        X = train_df[["lag_1h", "lag_24h"]].to_numpy()
        y = train_df["demand"].to_numpy()

        self.model_ = LinearRegression()
        self.model_.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted demand as a 1D numpy array.

        Drop rows where lag_1h or lag_24h is NaN before predicting. This
        method must be called after fit or load.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted or loaded.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("DemandForecaster must be fitted before prediction.")

        pred_df = df.dropna(subset=["lag_1h", "lag_24h"])
        X = pred_df[["lag_1h", "lag_24h"]].to_numpy()
        return np.asarray(self.model_.predict(X)).ravel()

    def save(self, path: str) -> None:
        """Save model to path using joblib.dump."""
        joblib.dump(self.model_, path)

    def load(self, path: str) -> None:
        """Load model from path using joblib.load."""
        self.model_ = joblib.load(path)
