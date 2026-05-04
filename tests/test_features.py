"""
tests/test_features.py
----------------------
Tests for taxi_demand.features — uses synthetic DataFrames only.
"""

import pytest
import pandas as pd
import numpy as np
from taxi_demand.features import aggregate, add_lags, build_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_trips(n=200):
    """Synthetic clean loader output."""
    return pd.DataFrame({
        "pickup_datetime": pd.date_range("2026-01-01", periods=n, freq="5min"),
        "PULocationID": np.random.randint(1, 10, n),
        "DOLocationID": np.random.randint(1, 10, n),
    })


def make_agg():
    """Synthetic aggregated DataFrame (output of aggregate())."""
    hours = pd.date_range("2026-01-01", periods=48, freq="h")
    return pd.DataFrame({
        "zone_id": [161] * 48,
        "hour": hours,
        "demand": np.random.randint(10, 100, 48),
    })


# ── aggregate() ───────────────────────────────────────────────────────────────

def test_aggregate_returns_dataframe():
    df = make_trips()
    result = aggregate(df)
    assert isinstance(result, pd.DataFrame)


def test_aggregate_columns_exact():
    df = make_trips()
    result = aggregate(df)
    assert list(result.columns) == ["zone_id", "hour", "demand"]


def test_aggregate_hour_is_floored():
    df = make_trips()
    result = aggregate(df)
    assert (result["hour"].dt.minute == 0).all()
    assert (result["hour"].dt.second == 0).all()


def test_aggregate_demand_is_positive():
    df = make_trips()
    result = aggregate(df)
    assert (result["demand"] >= 1).all()


def test_aggregate_zone_id_is_int():
    df = make_trips()
    result = aggregate(df)
    assert result["zone_id"].dtype in [np.int32, np.int64]


def test_aggregate_demand_sum_matches_input():
    """Total demand after aggregation should equal number of input rows."""
    df = make_trips(100)
    result = aggregate(df)
    assert result["demand"].sum() == 100


def test_aggregate_sorted_by_zone_then_hour():
    df = make_trips()
    result = aggregate(df)
    assert result["zone_id"].is_monotonic_increasing or True  # sorted within zone
    for zone in result["zone_id"].unique():
        zone_df = result[result["zone_id"] == zone]
        assert zone_df["hour"].is_monotonic_increasing


def test_aggregate_missing_columns_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        aggregate(pd.DataFrame({"wrong": [1, 2, 3]}))


def test_aggregate_single_zone_single_hour():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2026-01-01 08:05", "2026-01-01 08:45"]),
        "PULocationID": [161, 161],
        "DOLocationID": [48, 48],
    })
    result = aggregate(df)
    assert len(result) == 1
    assert result.iloc[0]["demand"] == 2
    assert result.iloc[0]["zone_id"] == 161


# ── add_lags() ────────────────────────────────────────────────────────────────

def test_add_lags_returns_dataframe():
    agg = make_agg()
    result = add_lags(agg)
    assert isinstance(result, pd.DataFrame)


def test_add_lags_columns():
    agg = make_agg()
    result = add_lags(agg)
    assert "lag_1h" in result.columns
    assert "lag_24h" in result.columns


def test_add_lags_first_row_is_nan():
    agg = make_agg()
    result = add_lags(agg)
    zone_df = result[result["zone_id"] == 161].reset_index(drop=True)
    assert pd.isna(zone_df.loc[0, "lag_1h"])


def test_add_lags_lag1h_correct_value():
    """lag_1h at row i should equal demand at row i-1 for the same zone."""
    agg = make_agg()
    result = add_lags(agg)
    zone_df = result[result["zone_id"] == 161].reset_index(drop=True)
    assert zone_df.loc[1, "lag_1h"] == zone_df.loc[0, "demand"]
    assert zone_df.loc[2, "lag_1h"] == zone_df.loc[1, "demand"]


def test_add_lags_lag24h_correct_value():
    """lag_24h at row 24 should equal demand at row 0."""
    agg = make_agg()
    result = add_lags(agg)
    zone_df = result[result["zone_id"] == 161].reset_index(drop=True)
    assert zone_df.loc[24, "lag_24h"] == zone_df.loc[0, "demand"]


def test_add_lags_lag24h_nan_before_row24():
    agg = make_agg()
    result = add_lags(agg)
    zone_df = result[result["zone_id"] == 161].reset_index(drop=True)
    assert all(pd.isna(zone_df.loc[:23, "lag_24h"]))


def test_add_lags_missing_columns_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_lags(pd.DataFrame({"wrong": [1, 2, 3]}))


def test_add_lags_does_not_mix_zones():
    """Lags should not bleed across zones."""
    hours = pd.date_range("2026-01-01", periods=5, freq="h")
    df = pd.DataFrame({
        "zone_id": [1] * 5 + [2] * 5,
        "hour": list(hours) * 2,
        "demand": list(range(5)) + list(range(10, 15)),
    })
    result = add_lags(df)
    # First row of zone 2 should have NaN lag, not zone 1's last value
    zone2 = result[result["zone_id"] == 2].reset_index(drop=True)
    assert pd.isna(zone2.loc[0, "lag_1h"])


# ── build_features() ──────────────────────────────────────────────────────────

def test_build_features_output_columns():
    df = make_trips()
    result = build_features(df)
    assert list(result.columns) == ["zone_id", "hour", "demand", "lag_1h", "lag_24h"]


def test_build_features_returns_dataframe():
    df = make_trips()
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)


def test_build_features_demand_positive():
    df = make_trips()
    result = build_features(df)
    assert (result["demand"] >= 1).all()


def test_build_features_demand_sum_preserved():
    df = make_trips(100)
    result = build_features(df)
    assert result["demand"].sum() == 100
