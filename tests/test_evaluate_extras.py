"""Tests for the new evaluation utilities added on top of the team's
original ``mae`` / ``rmse`` / ``naive_baseline`` / ``evaluate_model``.

This file focuses on:

* :func:`mape` -- mean absolute percentage error with epsilon-stability.
* :func:`time_series_cv_split` -- time-respecting train/test splitter.
* :func:`evaluate_model` with the ``include_mape=True`` flag.

The team's existing ``test_evaluate.py`` covers all the original
behaviour and confirms the four-key return contract; we deliberately
do not duplicate those tests.
"""

import math
import numpy as np
import pandas as pd
import pytest

from taxi_demand.evaluate import evaluate_model, mape, time_series_cv_split
from taxi_demand.model import DemandForecaster




def _eval_df(n=100, seed=42):
    """Build a feature DataFrame matching the model's expected columns."""
    rng = np.random.default_rng(seed)
    lag_1h = rng.integers(10, 200, n).astype(float)
    lag_24h = rng.integers(10, 200, n).astype(float)
    demand = (0.6 * lag_1h + 0.4 * lag_24h + rng.normal(0, 5, n)).astype(int)
    return pd.DataFrame(
        {
            "zone_id": rng.integers(1, 10, n),
            "hour": pd.date_range("2026-01-01", periods=n, freq="h"),
            "demand": demand,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
        }
    )




def test_mape_perfect_prediction_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert math.isclose(mape(y, y), 0.0, abs_tol=1e-9)


def test_mape_known_value():
    # |10-12|/10 + |20-18|/20 + |30-30|/30 = 0.2 + 0.1 + 0
    # mean = 0.3 / 3 = 0.1
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    assert abs(mape(y_true, y_pred) - 0.1) < 1e-4


def test_mape_handles_zero_in_y_true_via_epsilon():
    # Without epsilon this would be a divide-by-zero; the function
    # should return a finite value instead.
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 11.0])
    result = mape(y_true, y_pred)
    assert math.isfinite(result)


def test_mape_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mape(np.array([1, 2, 3]), np.array([1, 2]))


def test_mape_empty_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        mape(np.array([]), np.array([]))


def test_mape_negative_epsilon_raises():
    with pytest.raises(ValueError, match="non-negative"):
        mape(np.array([1.0]), np.array([1.0]), epsilon=-1.0)


def test_mape_non_negative():
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    p = rng.normal(size=20)
    assert mape(y, p) >= 0




def test_cv_split_basic():
    splits = list(time_series_cv_split(n_samples=24, n_splits=3))
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_cv_split_train_strictly_before_test():
    """The most important time-series CV invariant: no leakage."""
    splits = list(time_series_cv_split(n_samples=24, n_splits=3))
    for train_idx, test_idx in splits:
        assert int(train_idx.max()) < int(test_idx.min())


def test_cv_split_train_grows_monotonically():
    splits = list(time_series_cv_split(n_samples=24, n_splits=4))
    prev_len = 0
    for train_idx, _ in splits:
        assert len(train_idx) > prev_len
        prev_len = len(train_idx)


def test_cv_split_no_overlap_between_train_and_test():
    splits = list(time_series_cv_split(n_samples=30, n_splits=5))
    for train_idx, test_idx in splits:
        overlap = set(train_idx.tolist()) & set(test_idx.tolist())
        assert overlap == set()


def test_cv_split_invalid_n_samples_raises():
    with pytest.raises(ValueError, match="n_samples"):
        list(time_series_cv_split(n_samples=0))
    with pytest.raises(ValueError, match="n_samples"):
        list(time_series_cv_split(n_samples=-1))
    with pytest.raises(ValueError, match="n_samples"):
        list(time_series_cv_split(n_samples=3.5))


def test_cv_split_invalid_n_splits_raises():
    with pytest.raises(ValueError, match="n_splits"):
        list(time_series_cv_split(n_samples=10, n_splits=1))
    with pytest.raises(ValueError, match="n_splits"):
        list(time_series_cv_split(n_samples=10, n_splits=0))


def test_cv_split_too_few_samples_raises():
    with pytest.raises(ValueError, match="at least"):
        list(time_series_cv_split(n_samples=2, n_splits=5))


def test_cv_split_minimum_viable():
    # n_samples = n_splits + 1 → test_size collapses to 1
    splits = list(time_series_cv_split(n_samples=3, n_splits=2))
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0




def test_evaluate_model_default_returns_four_keys():
    """Existing behaviour must not change with default args."""
    df = _eval_df()
    model = DemandForecaster()
    model.fit(df)
    result = evaluate_model(model, df)
    assert set(result.keys()) == {
        "model_mae", "model_rmse", "baseline_mae", "baseline_rmse"
    }


def test_evaluate_model_with_mape_returns_six_keys():
    df = _eval_df()
    model = DemandForecaster()
    model.fit(df)
    result = evaluate_model(model, df, include_mape=True)
    assert set(result.keys()) == {
        "model_mae", "model_rmse", "baseline_mae", "baseline_rmse",
        "model_mape", "baseline_mape",
    }


def test_evaluate_model_mape_values_are_floats():
    df = _eval_df()
    model = DemandForecaster()
    model.fit(df)
    result = evaluate_model(model, df, include_mape=True)
    assert isinstance(result["model_mape"], float)
    assert isinstance(result["baseline_mape"], float)


def test_evaluate_model_mape_non_negative():
    df = _eval_df()
    model = DemandForecaster()
    model.fit(df)
    result = evaluate_model(model, df, include_mape=True)
    assert result["model_mape"] >= 0
    assert result["baseline_mape"] >= 0
