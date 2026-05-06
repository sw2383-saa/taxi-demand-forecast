"""Tests for :mod:`taxi_demand.parallel`.

The most important property to verify is that the parallel and
sequential code paths produce **identical** output DataFrames for any
deterministic ``func``. This mirrors the HW3 satisfied-mark pattern
in which ``parallel_matvec`` was verified against a numpy reference.

Workers are spawned via ``multiprocessing.Pool`` so the worker
function and its arguments must be picklable. We define helpers at
module level for that reason.
"""

import pandas as pd
import pytest

from taxi_demand.parallel import _apply_one, parallel_apply_per_zone




def _add_double_count(zone_df):
    """Worker that adds a ``doubled`` column (= 2 * demand)."""
    out = zone_df.copy()
    out["doubled"] = out["demand"] * 2
    return out


def _identity(zone_df):
    """Worker that returns its input unchanged."""
    return zone_df.copy()




def _demand_frame():
    return pd.DataFrame(
        {
            "zone_id": [4, 4, 13, 13, 24, 24, 132, 132],
            "hour": pd.to_datetime(
                [
                    "2026-01-01 00:00", "2026-01-01 01:00",
                    "2026-01-01 00:00", "2026-01-01 01:00",
                    "2026-01-01 00:00", "2026-01-01 01:00",
                    "2026-01-01 00:00", "2026-01-01 01:00",
                ]
            ),
            "demand": [1, 2, 10, 20, 100, 200, 1000, 2000],
        }
    )




def test_single_worker_correct_output():
    df = _demand_frame()
    out = parallel_apply_per_zone(df, _add_double_count, n_workers=1)
    assert len(out) == len(df)
    assert int(out["doubled"].sum()) == int(df["demand"].sum() * 2)


def test_parallel_matches_sequential_exactly():
    """The single most important invariant: parallel == sequential."""
    df = _demand_frame()
    seq = parallel_apply_per_zone(df, _add_double_count, n_workers=1)
    par = parallel_apply_per_zone(df, _add_double_count, n_workers=4)
    pd.testing.assert_frame_equal(
        seq.reset_index(drop=True),
        par.reset_index(drop=True),
    )


def test_output_sorted_by_zone_then_hour():
    df = _demand_frame()
    out = parallel_apply_per_zone(df, _identity, n_workers=2)
    # Zone ids must be monotonically non-decreasing.
    zones = out["zone_id"].tolist()
    assert zones == sorted(zones)


def test_empty_input_returns_empty():
    df = _demand_frame().iloc[0:0].copy()
    out = parallel_apply_per_zone(df, _identity, n_workers=4)
    assert len(out) == 0


def test_single_zone_uses_sequential_fastpath():
    df = _demand_frame()
    one_zone = df[df["zone_id"] == 4].copy()
    out = parallel_apply_per_zone(one_zone, _add_double_count, n_workers=4)
    assert len(out) == 2


def test_invalid_n_workers_raises():
    df = _demand_frame()
    with pytest.raises(ValueError, match="positive integer"):
        parallel_apply_per_zone(df, _identity, n_workers=0)
    with pytest.raises(ValueError, match="positive integer"):
        parallel_apply_per_zone(df, _identity, n_workers=-1)
    with pytest.raises(ValueError, match="positive integer"):
        parallel_apply_per_zone(df, _identity, n_workers="four")


def test_missing_group_column_raises():
    df = _demand_frame().drop(columns=["zone_id"])
    with pytest.raises(ValueError, match="group column"):
        parallel_apply_per_zone(df, _identity, n_workers=1)


def test_missing_time_column_raises():
    df = _demand_frame().drop(columns=["hour"])
    with pytest.raises(ValueError, match="time column"):
        parallel_apply_per_zone(df, _identity, n_workers=1)


def test_n_workers_none_falls_back_to_cpu_count():
    df = _demand_frame()
    out = parallel_apply_per_zone(df, _identity, n_workers=None)
    assert len(out) == len(df)


def test_function_is_actually_applied():
    df = _demand_frame()
    out = parallel_apply_per_zone(df, _add_double_count, n_workers=4)
    for _, row in out.iterrows():
        assert row["doubled"] == row["demand"] * 2


def test_apply_one_helper_directly():
    """Cover the internal worker function without going through Pool.

    Without this test the worker body would only execute in
    subprocesses, where ``coverage`` cannot see it.
    """
    df = _demand_frame()
    one_zone = df[df["zone_id"] == 4].copy()
    result = _apply_one((_add_double_count, one_zone))
    assert "doubled" in result.columns
