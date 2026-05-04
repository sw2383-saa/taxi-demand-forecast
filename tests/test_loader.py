"""
tests/test_loader.py
--------------------
Tests for taxi_demand.loader — uses synthetic parquet files only.
No network calls are made.
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from taxi_demand.loader import load, download, REQUIRED_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_parquet(tmp_path, n=200, include_nulls=True, include_bad_zones=True):
    """Create a synthetic HVFHV-style parquet file for testing."""
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
    return path, df


# ── load() ────────────────────────────────────────────────────────────────────

def test_load_returns_dataframe(tmp_path):
    path, _ = make_parquet(tmp_path)
    df = load(path)
    assert isinstance(df, pd.DataFrame)


def test_load_columns_exact(tmp_path):
    path, _ = make_parquet(tmp_path)
    df = load(path)
    assert list(df.columns) == ["pickup_datetime", "PULocationID", "DOLocationID"]


def test_load_pickup_datetime_is_datetime(tmp_path):
    path, _ = make_parquet(tmp_path)
    df = load(path)
    assert pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"])


def test_load_zone_ids_are_int(tmp_path):
    path, _ = make_parquet(tmp_path)
    df = load(path)
    assert df["PULocationID"].dtype in [np.int32, np.int64]
    assert df["DOLocationID"].dtype in [np.int32, np.int64]


def test_load_drops_null_rows(tmp_path):
    path, _ = make_parquet(tmp_path, include_nulls=True)
    df = load(path)
    assert df.isnull().sum().sum() == 0


def test_load_filters_invalid_pu_zones(tmp_path):
    path, _ = make_parquet(tmp_path, include_bad_zones=True)
    df = load(path)
    assert df["PULocationID"].between(1, 263).all()


def test_load_filters_invalid_do_zones(tmp_path):
    path, _ = make_parquet(tmp_path, include_bad_zones=True)
    df = load(path)
    assert df["DOLocationID"].between(1, 263).all()


def test_load_resets_index(tmp_path):
    path, _ = make_parquet(tmp_path)
    df = load(path)
    assert list(df.index) == list(range(len(df)))


def test_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        load("nonexistent_file.parquet")


def test_load_missing_columns(tmp_path):
    # Parquet with wrong columns
    bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
    path = str(tmp_path / "bad.parquet")
    bad_df.to_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        load(path)


def test_load_clean_data_no_rows_dropped(tmp_path):
    """If data is already clean, all rows should survive."""
    df = pd.DataFrame({
        "pickup_datetime": pd.date_range("2026-01-01", periods=50, freq="1min"),
        "PULocationID": np.random.randint(1, 200, 50),
        "DOLocationID": np.random.randint(1, 200, 50),
    })
    path = str(tmp_path / "clean.parquet")
    df.to_parquet(path)
    result = load(path)
    assert len(result) == 50


# ── download() ────────────────────────────────────────────────────────────────

def test_download_invalid_year():
    with pytest.raises(ValueError, match="year"):
        download(2018, 1)


def test_download_invalid_month():
    with pytest.raises(ValueError, match="month"):
        download(2026, 13)


def test_download_invalid_month_zero():
    with pytest.raises(ValueError, match="month"):
        download(2026, 0)


def test_download_uses_cache(tmp_path, monkeypatch):
    """If file already exists, download() should return path without fetching."""
    # Pre-create the file
    cached = tmp_path / "fhvhv_tripdata_2026-01.parquet"
    cached.write_bytes(b"fake content")

    import taxi_demand.loader as loader_module
    called = []

    original_get = loader_module.requests.get
    def mock_get(*args, **kwargs):
        called.append(True)
        return original_get(*args, **kwargs)

    monkeypatch.setattr(loader_module.requests, "get", mock_get)

    result = download(2026, 1, data_dir=str(tmp_path))
    assert result == str(cached)
    assert len(called) == 0  # no HTTP request made
