"""Tests for the new loader functions: ``load_raw``, ``clean_trips``,
and the retry / atomic-write behaviour of ``download``.

The team's original ``test_loader.py`` already covers the
backwards-compatible ``load`` and ``download`` paths; this file
focuses on the new public surface so we do not duplicate those
checks.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from taxi_demand import loader as loader_module
from taxi_demand.loader import (
    NUM_TAXI_ZONES,
    REQUIRED_COLUMNS,
    clean_trips,
    download,
    load_raw,
)




def make_parquet(tmp_path, n=20, include_nulls=False, include_bad_zones=False):
    """Create a synthetic HVFHV-style parquet file."""
    df = pd.DataFrame({
        "pickup_datetime": pd.date_range("2026-01-01", periods=n, freq="1min"),
        "PULocationID": np.random.randint(1, 200, n),
        "DOLocationID": np.random.randint(1, 200, n),
    })
    if include_nulls:
        df.loc[0, "pickup_datetime"] = pd.NaT
        df.loc[1, "PULocationID"] = np.nan
    if include_bad_zones:
        df.loc[2, "PULocationID"] = 999
        df.loc[3, "DOLocationID"] = 0
    path = str(tmp_path / "test.parquet")
    df.to_parquet(path)
    return path




def test_load_raw_returns_dataframe(tmp_path):
    path = make_parquet(tmp_path)
    df = load_raw(path)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == REQUIRED_COLUMNS


def test_load_raw_preserves_nulls(tmp_path):
    path = make_parquet(tmp_path, include_nulls=True)
    df = load_raw(path)
    # Per forum #81 we deliberately keep the raw rows; the cleaning
    # step is responsible for dropping them.
    assert df.isnull().any().any()


def test_load_raw_preserves_bad_zones(tmp_path):
    path = make_parquet(tmp_path, include_bad_zones=True)
    df = load_raw(path)
    # We deliberately did not filter; the cleaning step will.
    assert (df["PULocationID"] == 999).any()


def test_load_raw_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_raw("nonexistent.parquet")


def test_load_raw_missing_columns_raises(tmp_path):
    bad = pd.DataFrame({"wrong_col": [1, 2, 3]})
    path = str(tmp_path / "bad.parquet")
    bad.to_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_raw(path)




def test_clean_trips_drops_nulls():
    df = pd.DataFrame({
        "pickup_datetime": [pd.Timestamp("2026-01-01"), pd.NaT, pd.Timestamp("2026-01-02")],
        "PULocationID": [1, 2, 3],
        "DOLocationID": [10, 20, 30],
    })
    out = clean_trips(df)
    assert len(out) == 2
    assert out.isnull().sum().sum() == 0


def test_clean_trips_filters_invalid_zones_265():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
        ),
        "PULocationID": [1, 265, 266, 0],
        "DOLocationID": [1, 1, 1, 1],
    })
    out = clean_trips(df)
    # 1 and 265 stay; 266 and 0 are dropped.
    assert sorted(out["PULocationID"].tolist()) == [1, 265]


def test_clean_trips_resets_index():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2026-01-01"] * 5),
        "PULocationID": [1, 2, 3, 4, 5],
        "DOLocationID": [1, 1, 1, 1, 1],
    }).set_index(pd.RangeIndex(start=100, stop=105))
    out = clean_trips(df)
    assert list(out.index) == list(range(len(out)))


def test_clean_trips_int64_zone_columns():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2026-01-01"] * 3),
        "PULocationID": [1, 2, 3],
        "DOLocationID": [4, 5, 6],
    })
    out = clean_trips(df)
    assert out["PULocationID"].dtype == np.int64
    assert out["DOLocationID"].dtype == np.int64


def test_clean_trips_missing_columns_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        clean_trips(pd.DataFrame({"x": [1]}))


def test_clean_trips_non_positive_num_zones_raises():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2026-01-01"]),
        "PULocationID": [1], "DOLocationID": [1],
    })
    with pytest.raises(ValueError, match="num_zones"):
        clean_trips(df, num_zones=0)


def test_clean_trips_does_not_mutate_input():
    df = pd.DataFrame({
        "pickup_datetime": pd.to_datetime(["2026-01-01"] * 3),
        "PULocationID": [1, 2, 3],
        "DOLocationID": [4, 5, 6],
    })
    original_columns = list(df.columns)
    clean_trips(df)
    assert list(df.columns) == original_columns


def test_num_taxi_zones_constant():
    # Asserting the constant catches the off-by-two bug from the
    # team's first version (which filtered to 263 instead of 265).
    assert NUM_TAXI_ZONES == 265




def test_download_year_too_high():
    with pytest.raises(ValueError, match="year"):
        download(2200, 1)


def test_download_max_retries_below_one():
    with pytest.raises(ValueError, match="max_retries"):
        download(2026, 1, max_retries=0)


def test_download_negative_backoff():
    with pytest.raises(ValueError, match="backoff_seconds"):
        download(2026, 1, backoff_seconds=-1)


def test_download_non_positive_timeout():
    with pytest.raises(ValueError, match="timeout"):
        download(2026, 1, timeout=0)




class _FakeResponse:
    """Minimal stand-in for a streaming requests.Response."""

    def __init__(self, status_code, body=b"", raises=None):
        self.status_code = status_code
        self._body = body
        self._raises = raises

    def raise_for_status(self):
        if self._raises is not None:
            raise self._raises
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self._body


def test_download_atomic_write_on_success(tmp_path, monkeypatch):
    expected_body = b"parquet bytes"

    def fake_get(url, headers=None, stream=None, timeout=None):
        return _FakeResponse(200, body=expected_body)

    monkeypatch.setattr(loader_module.requests, "get", fake_get)

    out = download(2026, 1, data_dir=str(tmp_path), backoff_seconds=0)
    assert os.path.exists(out)
    with open(out, "rb") as f:
        assert f.read() == expected_body
    # The .part file should not linger.
    assert not os.path.exists(out + ".part")


def test_download_retries_after_failure(tmp_path, monkeypatch):
    calls = []

    def flaky_get(url, headers=None, stream=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            return _FakeResponse(503)  # transient failure
        return _FakeResponse(200, body=b"finally")

    monkeypatch.setattr(loader_module.requests, "get", flaky_get)

    out = download(
        2026, 1, data_dir=str(tmp_path), max_retries=5, backoff_seconds=0
    )
    assert os.path.exists(out)
    assert len(calls) == 3


def test_download_raises_after_exhausting_retries(tmp_path, monkeypatch):
    def always_fail(url, headers=None, stream=None, timeout=None):
        return _FakeResponse(500)

    monkeypatch.setattr(loader_module.requests, "get", always_fail)

    with pytest.raises(Exception):
        download(2026, 1, data_dir=str(tmp_path), max_retries=2, backoff_seconds=0)
    # No leftover .part file.
    assert not any(p.suffix == ".part" for p in tmp_path.iterdir())
