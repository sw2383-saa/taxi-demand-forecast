"""Tests for the new feature-engineering functions added on top of the
team's original ``aggregate``, ``add_lags``, ``build_features``.

The team's existing ``test_features.py`` covers the original API; this
file focuses on:

* ``add_calendar_features``  -- hour-of-day, day-of-week, is_weekend,
  is_holiday columns.
* ``add_rolling_features``   -- causal rolling-mean and rolling-std.
* ``add_lags`` with a custom lag list (the team's tests only used
  defaults, but the function now accepts arbitrary positive-integer
  lag offsets).
* ``build_features`` with ``rolling_windows=`` and ``add_calendar=``
  arguments.
"""

import math
import numpy as np
import pandas as pd
import pytest

from taxi_demand.features import (
    add_calendar_features,
    add_lags,
    add_rolling_features,
    build_features,
)




def _agg_frame(zone_ids, hours, demand):
    return pd.DataFrame(
        {
            "zone_id": zone_ids,
            "hour": pd.to_datetime(hours),
            "demand": demand,
        }
    )


def _trip_frame(n=200):
    return pd.DataFrame(
        {
            "pickup_datetime": pd.date_range("2026-01-01", periods=n, freq="5min"),
            "PULocationID": np.random.randint(1, 10, n),
            "DOLocationID": np.random.randint(1, 10, n),
        }
    )




def test_calendar_features_added():
    df = _agg_frame([1], ["2026-01-01 12:00"], [10])
    out = add_calendar_features(df)
    for col in ("hour_of_day", "day_of_week", "is_weekend", "is_holiday"):
        assert col in out.columns


def test_calendar_hour_of_day_extracted():
    df = _agg_frame(
        [1, 1, 1],
        ["2026-01-01 00:00", "2026-01-01 12:00", "2026-01-01 23:00"],
        [10, 20, 30],
    )
    out = add_calendar_features(df)
    assert list(out["hour_of_day"]) == [0, 12, 23]


def test_calendar_day_of_week_monday_zero():
    # 2026-01-05 is a Monday
    df = _agg_frame([1], ["2026-01-05 12:00"], [10])
    out = add_calendar_features(df)
    assert out["day_of_week"].iloc[0] == 0


def test_calendar_day_of_week_sunday_six():
    # 2026-01-04 is a Sunday
    df = _agg_frame([1], ["2026-01-04 12:00"], [10])
    out = add_calendar_features(df)
    assert out["day_of_week"].iloc[0] == 6


def test_calendar_is_weekend_correct():
    df = _agg_frame(
        [1, 1, 1, 1],
        # Mon, Sat, Sun, Tue
        ["2026-01-05 12:00", "2026-01-03 12:00", "2026-01-04 12:00", "2026-01-06 12:00"],
        [1, 2, 3, 4],
    )
    out = add_calendar_features(df)
    assert list(out["is_weekend"]) == [0, 1, 1, 0]


def test_calendar_default_us_holidays():
    # 2026-01-01 (New Year's Day), 2026-01-02 (not), 2026-12-25 (Christmas)
    df = _agg_frame(
        [1, 1, 1],
        ["2026-01-01 12:00", "2026-01-02 12:00", "2026-12-25 12:00"],
        [1, 2, 3],
    )
    out = add_calendar_features(df)
    assert list(out["is_holiday"]) == [1, 0, 1]


def test_calendar_custom_holiday_set():
    df = _agg_frame(
        [1, 1],
        ["2026-03-15 12:00", "2026-03-16 12:00"],
        [1, 2],
    )
    out = add_calendar_features(df, holidays=["2026-03-15"])
    assert list(out["is_holiday"]) == [1, 0]


def test_calendar_holiday_check_uses_date_only():
    df = _agg_frame(
        [1, 1],
        ["2026-01-01 23:59", "2026-01-01 00:00"],
        [1, 2],
    )
    out = add_calendar_features(df)
    # Both rows fall on the holiday date regardless of hour.
    assert list(out["is_holiday"]) == [1, 1]


def test_calendar_missing_hour_column_raises():
    with pytest.raises(ValueError, match="hour"):
        add_calendar_features(pd.DataFrame({"zone_id": [1], "demand": [1]}))


def test_calendar_unparseable_hour_raises():
    df = pd.DataFrame({"hour": ["not-a-date"], "zone_id": [1], "demand": [1]})
    with pytest.raises(ValueError, match="parsed"):
        add_calendar_features(df)


def test_calendar_does_not_mutate_input():
    df = _agg_frame([1], ["2026-01-01 12:00"], [10])
    original_cols = list(df.columns)
    add_calendar_features(df)
    assert list(df.columns) == original_cols




def test_rolling_mean_correctness():
    df = _agg_frame(
        [1, 1, 1],
        ["2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"],
        [10, 20, 30],
    )
    out = add_rolling_features(df, windows=[2])
    # The rolling window is STRICTLY CAUSAL (uses only past hours,
    # never the current one), so:
    #   row 0: no past data    -> NaN
    #   row 1: past = [10]     -> mean = 10
    #   row 2: past = [10, 20] -> mean = 15
    means = out["roll_mean_2h"].tolist()
    assert pd.isna(means[0])
    np.testing.assert_allclose(means[1:], [10.0, 15.0])


def test_rolling_std_zero_for_constant_window():
    df = _agg_frame(
        [1, 1, 1, 1],
        [
            "2026-01-01 00:00", "2026-01-01 01:00",
            "2026-01-01 02:00", "2026-01-01 03:00",
        ],
        [5, 5, 5, 5],
    )
    out = add_rolling_features(df, windows=[2])
    # Row 0 has no past data and gets NaN std filled to 0.0.
    # Rows 1+ have constant-5 past windows -> std = 0.0.
    for v in out["roll_std_2h"]:
        assert v == 0.0


def test_rolling_does_not_leak_across_zones():
    df = _agg_frame(
        [1, 1, 2, 2],
        [
            "2026-01-01 00:00", "2026-01-01 01:00",
            "2026-01-01 00:00", "2026-01-01 01:00",
        ],
        [10, 20, 100, 200],
    )
    out = add_rolling_features(df, windows=[2])
    zone2 = out[out["zone_id"] == 2].sort_values("hour")
    # The first row of zone 2 has no past history within zone 2,
    # so its rolling mean is NaN -- crucially NOT contaminated by
    # zone 1's earlier values.
    assert pd.isna(zone2["roll_mean_2h"].iloc[0])
    # The second row of zone 2 sees only zone 2's past (100), not
    # any of zone 1's values.
    assert zone2["roll_mean_2h"].iloc[1] == 100.0


def test_rolling_features_are_strictly_causal():
    # This is the central correctness test: at row i, the rolling
    # feature must NOT depend on demand[i] itself. We verify this
    # by changing demand[i] and confirming the rolling feature at
    # row i is unchanged.
    df = _agg_frame(
        [1, 1, 1, 1],
        [f"2026-01-01 0{h}:00" for h in range(4)],
        [10, 20, 30, 40],
    )
    out_a = add_rolling_features(df, windows=[2])

    df2 = df.copy()
    df2.loc[df2["hour"] == "2026-01-01 02:00", "demand"] = 999  # changed!
    out_b = add_rolling_features(df2, windows=[2])

    # The rolling features at row 2 must be identical between A and B,
    # because they only depend on rows 0 and 1, not on row 2 itself.
    a_row = out_a[out_a["hour"] == "2026-01-01 02:00"].iloc[0]
    b_row = out_b[out_b["hour"] == "2026-01-01 02:00"].iloc[0]
    assert a_row["roll_mean_2h"] == b_row["roll_mean_2h"]
    assert a_row["roll_std_2h"] == b_row["roll_std_2h"]


def test_rolling_default_window_present():
    df = _agg_frame([1], ["2026-01-01 00:00"], [10])
    out = add_rolling_features(df)
    assert "roll_mean_24h" in out.columns
    assert "roll_std_24h" in out.columns


def test_rolling_missing_columns_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        add_rolling_features(pd.DataFrame({"x": [1, 2, 3]}))


def test_rolling_invalid_window_raises():
    df = _agg_frame([1], ["2026-01-01 00:00"], [10])
    with pytest.raises(ValueError, match="positive integers"):
        add_rolling_features(df, windows=[0])
    with pytest.raises(ValueError, match="positive integers"):
        add_rolling_features(df, windows=[-1])
    with pytest.raises(ValueError, match="positive integers"):
        add_rolling_features(df, windows=[1.5])
    with pytest.raises(ValueError, match="at least one"):
        add_rolling_features(df, windows=[])




def test_add_lags_custom_lag_list():
    hours = pd.date_range("2026-01-01", periods=10, freq="h")
    df = _agg_frame([1] * 10, hours, list(range(10)))
    out = add_lags(df, lags=[1, 3])
    assert "lag_1h" in out.columns
    assert "lag_3h" in out.columns
    # lag_3 at row 5 should equal demand at row 2 (= 2)
    sorted_out = out.sort_values("hour").reset_index(drop=True)
    assert sorted_out["lag_3h"].iloc[5] == 2.0


def test_add_lags_invalid_lag_raises():
    df = _agg_frame([1], ["2026-01-01"], [1])
    with pytest.raises(ValueError, match="positive integers"):
        add_lags(df, lags=[0])
    with pytest.raises(ValueError, match="positive integers"):
        add_lags(df, lags=[-5])
    with pytest.raises(ValueError, match="positive integers"):
        add_lags(df, lags=[2.5])




def test_build_features_default_matches_team_contract():
    """Default args must reproduce the team's original output schema."""
    df = _trip_frame(50)
    out = build_features(df)
    assert list(out.columns) == ["zone_id", "hour", "demand", "lag_1h", "lag_24h"]


def test_build_features_with_rolling():
    df = _trip_frame(100)
    out = build_features(df, rolling_windows=[24])
    assert "roll_mean_24h" in out.columns
    assert "roll_std_24h" in out.columns


def test_build_features_with_calendar():
    df = _trip_frame(100)
    out = build_features(df, add_calendar=True)
    for col in ("hour_of_day", "day_of_week", "is_weekend", "is_holiday"):
        assert col in out.columns


def test_build_features_with_custom_lags():
    df = _trip_frame(200)
    out = build_features(df, lags=[1, 24, 168])
    assert "lag_168h" in out.columns
