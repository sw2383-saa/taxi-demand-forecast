"""
tests/test_evaluate.py
----------------------
Tests for taxi_demand.evaluate — uses synthetic data only.
"""

import pytest
import numpy as np
import pandas as pd
from taxi_demand.model import DemandForecaster
from taxi_demand.evaluate import mae, rmse, naive_baseline, evaluate_model



def make_df(n=100, seed=42):
    """Synthetic features DataFrame matching the input contract."""
    rng = np.random.default_rng(seed)
    lag_1h = rng.integers(10, 200, n).astype(float)
    lag_24h = rng.integers(10, 200, n).astype(float)
    demand = (0.6 * lag_1h + 0.4 * lag_24h + rng.normal(0, 5, n)).astype(int)
    return pd.DataFrame({
        "zone_id": rng.integers(1, 10, n),
        "hour": pd.date_range("2026-01-01", periods=n, freq="h"),
        "demand": demand,
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
    })


def make_df_with_nans(n=100):
    df = make_df(n)
    df.loc[:24, "lag_24h"] = np.nan
    return df


def fitted_model(df=None):
    if df is None:
        df = make_df()
    model = DemandForecaster()
    model.fit(df)
    return model



def test_mae_returns_float():
    result = mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert isinstance(result, float)


def test_mae_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # abs errors: 2, 2, 3 → mean = 7/3
    assert abs(mae(y_true, y_pred) - 7 / 3) < 1e-9


def test_mae_is_non_negative():
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([3.0, 12.0, 14.0])
    assert mae(y_true, y_pred) >= 0



def test_rmse_returns_float():
    result = rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert isinstance(result, float)


def test_rmse_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # squared errors: 9, 16 → mean = 12.5 → sqrt = 3.535...
    expected = np.sqrt(12.5)
    assert abs(rmse(y_true, y_pred) - expected) < 1e-9


def test_rmse_is_non_negative():
    y_true = np.array([5.0, 10.0, 15.0])
    y_pred = np.array([3.0, 12.0, 14.0])
    assert rmse(y_true, y_pred) >= 0


def test_rmse_geq_mae():
    """RMSE is always >= MAE for the same inputs."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 17.0, 35.0, 38.0])
    assert rmse(y_true, y_pred) >= mae(y_true, y_pred)



def test_naive_baseline_returns_ndarray():
    df = make_df()
    result = naive_baseline(df)
    assert isinstance(result, np.ndarray)


def test_naive_baseline_is_1d():
    df = make_df()
    result = naive_baseline(df)
    assert result.ndim == 1


def test_naive_baseline_equals_lag24h():
    df = make_df()
    result = naive_baseline(df)
    expected = df.dropna(subset=["lag_1h", "lag_24h"])["lag_24h"].to_numpy()
    np.testing.assert_array_equal(result, expected)


def test_naive_baseline_drops_nan_rows():
    df = make_df_with_nans()
    result = naive_baseline(df)
    clean_len = df.dropna(subset=["lag_1h", "lag_24h"]).shape[0]
    assert len(result) == clean_len


def test_naive_baseline_no_nans_in_output():
    df = make_df_with_nans()
    result = naive_baseline(df)
    assert np.isfinite(result).all()



def test_evaluate_model_returns_dict():
    df = make_df()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    assert isinstance(result, dict)


def test_evaluate_model_has_all_keys():
    df = make_df()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    assert set(result.keys()) == {"model_mae", "model_rmse", "baseline_mae", "baseline_rmse"}


def test_evaluate_model_all_values_are_float():
    df = make_df()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    for key, val in result.items():
        assert isinstance(val, float), f"{key} is not float"


def test_evaluate_model_all_values_non_negative():
    df = make_df()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    for key, val in result.items():
        assert val >= 0, f"{key} is negative"


def test_evaluate_model_handles_nan_rows():
    df = make_df_with_nans()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    assert set(result.keys()) == {"model_mae", "model_rmse", "baseline_mae", "baseline_rmse"}


def test_evaluate_model_rmse_geq_mae():
    df = make_df()
    model = fitted_model(df)
    result = evaluate_model(model, df)
    assert result["model_rmse"] >= result["model_mae"]
    assert result["baseline_rmse"] >= result["baseline_mae"]
