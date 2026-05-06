"""
tests/test_model.py
-------------------
Tests for taxi_demand.model — uses synthetic DataFrames only.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from taxi_demand.model import DemandForecaster



def make_df(n=100, seed=42):
    """Synthetic features DataFrame matching the input contract."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-01-01", periods=n, freq="h")
    lag_1h = rng.integers(10, 200, n).astype(float)
    lag_24h = rng.integers(10, 200, n).astype(float)
    demand = (0.6 * lag_1h + 0.4 * lag_24h + rng.normal(0, 5, n)).astype(int)
    return pd.DataFrame({
        "zone_id": rng.integers(1, 10, n),
        "hour": hours,
        "demand": demand,
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
    })


def make_df_with_nans(n=100, seed=42):
    """DataFrame with NaN in first few lag rows (realistic)."""
    df = make_df(n, seed)
    df.loc[:5, "lag_1h"] = np.nan
    df.loc[:24, "lag_24h"] = np.nan
    return df



def test_fit_runs_without_error():
    model = DemandForecaster()
    model.fit(make_df())


def test_fit_sets_model_attribute():
    model = DemandForecaster()
    model.fit(make_df())
    assert hasattr(model, "model_")


def test_fit_handles_nan_rows():
    """fit() should drop NaN rows and still train successfully."""
    model = DemandForecaster()
    model.fit(make_df_with_nans())
    assert hasattr(model, "model_")


def test_fit_with_all_nan_raises():
    """If all rows have NaN lags, sklearn will fail gracefully."""
    model = DemandForecaster()
    df = make_df(10)
    df["lag_1h"] = np.nan
    with pytest.raises(Exception):
        model.fit(df)



def test_predict_returns_ndarray():
    model = DemandForecaster()
    df = make_df()
    model.fit(df)
    result = model.predict(df)
    assert isinstance(result, np.ndarray)


def test_predict_is_1d():
    model = DemandForecaster()
    df = make_df()
    model.fit(df)
    result = model.predict(df)
    assert result.ndim == 1


def test_predict_length_matches_non_nan_rows():
    model = DemandForecaster()
    df = make_df_with_nans()
    model.fit(df)
    result = model.predict(df)
    expected_len = df.dropna(subset=["lag_1h", "lag_24h"]).shape[0]
    assert len(result) == expected_len


def test_predict_before_fit_raises_runtime_error():
    model = DemandForecaster()
    with pytest.raises(RuntimeError):
        model.predict(make_df())


def test_predict_values_are_numeric():
    model = DemandForecaster()
    df = make_df()
    model.fit(df)
    result = model.predict(df)
    assert np.isfinite(result).all()


def test_predict_reasonable_range():
    """Predictions should be in a ballpark range given the synthetic data."""
    model = DemandForecaster()
    df = make_df()
    model.fit(df)
    result = model.predict(df)
    assert result.min() > -1000
    assert result.max() < 10000



def test_save_creates_file():
    model = DemandForecaster()
    model.fit(make_df())
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        model.save(path)
        assert os.path.exists(path)


def test_load_restores_predictions():
    """Save and load should produce identical predictions."""
    df = make_df()
    model = DemandForecaster()
    model.fit(df)
    original_preds = model.predict(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        model.save(path)

        loaded_model = DemandForecaster()
        loaded_model.load(path)
        loaded_preds = loaded_model.predict(df)

    np.testing.assert_array_almost_equal(original_preds, loaded_preds)


def test_load_enables_predict_without_fit():
    """A loaded model should predict without calling fit first."""
    df = make_df()
    original = DemandForecaster()
    original.fit(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.joblib")
        original.save(path)

        fresh = DemandForecaster()
        fresh.load(path)
        result = fresh.predict(df)
        assert len(result) > 0
