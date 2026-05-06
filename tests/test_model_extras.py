"""Tests for the new multi-model API in :mod:`taxi_demand.model`.

The team's existing ``test_model.py`` covers the simple
``DemandForecaster`` class. This file adds tests for:

* :func:`make_pipeline` -- the three supported model families and
  the optional StandardScaler step.
* :func:`train_models` -- subset training and input validation.
* :class:`MultiModelForecaster` -- end-to-end fit / predict.
* :data:`SUPPORTED_MODELS` and :data:`DEFAULT_RANDOM_STATE` constants.
* The shared ``_check_xy`` validator's error paths.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from taxi_demand.model import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_RANDOM_STATE,
    SUPPORTED_MODELS,
    MultiModelForecaster,
    make_pipeline,
    train_models,
)




def _toy_xy(n=100, seed=0):
    """Synthetic feature matrix and target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "lag_1h": rng.normal(size=n),
            "lag_24h": rng.normal(size=n),
            "extra": rng.normal(size=n),
        }
    )
    y = X["lag_1h"] * 2.0 + X["lag_24h"] * 0.5 - X["extra"] + rng.normal(scale=0.1, size=n)
    return X, y


def _toy_df(n=100, seed=0):
    """Synthetic DataFrame matching MultiModelForecaster's expected shape."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "zone_id": rng.integers(1, 5, n),
            "hour": pd.date_range("2026-01-01", periods=n, freq="h"),
            "lag_1h": rng.normal(size=n).astype(float),
            "lag_24h": rng.normal(size=n).astype(float),
            "demand": rng.integers(10, 200, n).astype(int),
        }
    )




def test_make_pipeline_returns_pipeline():
    pipe = make_pipeline("linear")
    assert isinstance(pipe, Pipeline)


def test_make_pipeline_linear_structure():
    pipe = make_pipeline("linear")
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["model"], LinearRegression)


def test_make_pipeline_ridge_structure():
    pipe = make_pipeline("ridge")
    assert isinstance(pipe.named_steps["model"], Ridge)


def test_make_pipeline_random_forest_structure():
    pipe = make_pipeline("random_forest")
    assert isinstance(pipe.named_steps["model"], RandomForestRegressor)


def test_make_pipeline_no_scaler():
    pipe = make_pipeline("linear", scaler=False)
    assert "scaler" not in pipe.named_steps
    assert "model" in pipe.named_steps


def test_make_pipeline_unknown_model_raises():
    with pytest.raises(ValueError, match="model_name"):
        make_pipeline("xgboost")


def test_supported_models_constant():
    # Locks the contract that all three families remain available.
    assert tuple(SUPPORTED_MODELS) == ("linear", "ridge", "random_forest")


def test_default_random_state_is_42():
    # Locks the value so the test suite stays deterministic.
    assert DEFAULT_RANDOM_STATE == 42


def test_default_feature_columns_constant():
    assert tuple(DEFAULT_FEATURE_COLUMNS) == ("lag_1h", "lag_24h")




def test_train_models_default_trains_all_three():
    X, y = _toy_xy()
    fitted = train_models(X, y)
    assert set(fitted.keys()) == set(SUPPORTED_MODELS)
    for pipe in fitted.values():
        assert isinstance(pipe, Pipeline)


def test_train_models_subset():
    X, y = _toy_xy()
    fitted = train_models(X, y, model_names=["linear"])
    assert list(fitted.keys()) == ["linear"]


def test_train_models_unknown_name_raises():
    X, y = _toy_xy()
    with pytest.raises(ValueError, match="Unknown model"):
        train_models(X, y, model_names=["bogus"])


def test_train_models_empty_names_raises():
    X, y = _toy_xy()
    with pytest.raises(ValueError, match="non-empty"):
        train_models(X, y, model_names=[])


def test_train_models_x_must_be_dataframe():
    X, y = _toy_xy()
    with pytest.raises(ValueError, match="DataFrame"):
        train_models(X.values, y)


def test_train_models_empty_x_raises():
    X, y = _toy_xy()
    with pytest.raises(ValueError, match="non-empty"):
        train_models(X.iloc[0:0], y[:0])


def test_train_models_shape_mismatch_raises():
    X, y = _toy_xy(n=10)
    with pytest.raises(ValueError, match="Shape mismatch"):
        train_models(X, y[:5])


def test_train_models_non_numeric_columns_raise():
    X, y = _toy_xy()
    X["bad"] = ["a"] * len(X)
    with pytest.raises(ValueError, match="non-numeric"):
        train_models(X, y)


def test_train_models_random_state_reproducible():
    X, y = _toy_xy(seed=1)
    a = train_models(X, y, model_names=["random_forest"], random_state=42)
    b = train_models(X, y, model_names=["random_forest"], random_state=42)
    np.testing.assert_allclose(
        a["random_forest"].predict(X),
        b["random_forest"].predict(X),
    )




def test_multi_forecaster_fit_and_predict():
    df = _toy_df()
    multi = MultiModelForecaster()
    multi.fit(df)
    preds = multi.predict(df, model_name="linear")
    assert preds.shape == (len(df),)


def test_multi_forecaster_all_three_models_trained():
    df = _toy_df()
    multi = MultiModelForecaster()
    multi.fit(df)
    assert set(multi.models_.keys()) == set(SUPPORTED_MODELS)


def test_multi_forecaster_unknown_model_raises():
    df = _toy_df()
    multi = MultiModelForecaster()
    multi.fit(df)
    with pytest.raises(ValueError, match="not trained"):
        multi.predict(df, model_name="bogus")


def test_multi_forecaster_predict_before_fit_raises():
    multi = MultiModelForecaster()
    with pytest.raises(RuntimeError, match="fitted"):
        multi.predict(_toy_df(), model_name="linear")


def test_multi_forecaster_subset_models():
    df = _toy_df()
    multi = MultiModelForecaster(model_names=["ridge"])
    multi.fit(df)
    assert list(multi.models_.keys()) == ["ridge"]


def test_multi_forecaster_drops_nan_rows():
    df = _toy_df()
    df.loc[:5, "lag_1h"] = np.nan
    multi = MultiModelForecaster()
    multi.fit(df)
    assert "linear" in multi.models_
    preds = multi.predict(df, model_name="linear")
    assert len(preds) == len(df.dropna(subset=["lag_1h", "lag_24h"]))


def test_multi_forecaster_returns_self_for_chaining():
    df = _toy_df()
    multi = MultiModelForecaster()
    result = multi.fit(df)
    assert result is multi
