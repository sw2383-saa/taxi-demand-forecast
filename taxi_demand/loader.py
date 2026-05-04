"""
loader.py
---------
Downloads and cleans the NYC TLC High Volume FHV (HVFHV) trip parquet file.
Returns a minimal, clean DataFrame ready for feature engineering.
"""

import os
import requests
import pandas as pd

HVFHV_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "fhvhv_tripdata_{year}-{month:02d}.parquet"
)

REQUIRED_COLUMNS = ["pickup_datetime", "PULocationID", "DOLocationID"]


def download(year: int, month: int, data_dir: str = "data") -> str:
    """
    Download the HVFHV parquet file for a given year and month if not cached.

    Parameters
    ----------
    year : int
        Year of the dataset (e.g. 2026).
    month : int
        Month of the dataset (1–12).
    data_dir : str
        Local directory to store downloaded files.

    Returns
    -------
    str
        Absolute path to the downloaded parquet file.

    Raises
    ------
    ValueError
        If year or month are out of valid range.
    requests.HTTPError
        If the TLC server returns a non-200 response.
    """
    if not (2019 <= year <= 2026):
        raise ValueError(f"year must be between 2019 and 2026, got {year}")
    if not (1 <= month <= 12):
        raise ValueError(f"month must be between 1 and 12, got {month}")

    os.makedirs(data_dir, exist_ok=True)
    filename = f"fhvhv_tripdata_{year}-{month:02d}.parquet"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        return filepath

    url = HVFHV_URL.format(year=year, month=month)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filepath


def load(filepath: str) -> pd.DataFrame:
    """
    Load and clean an HVFHV parquet file.

    Keeps only pickup_datetime, PULocationID, and DOLocationID.
    Drops rows with nulls in any of those columns.
    Ensures pickup_datetime is parsed as datetime64[ns].
    Filters to valid NYC taxi zone IDs (1–263).

    Parameters
    ----------
    filepath : str
        Path to a local HVFHV parquet file.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns:
        - pickup_datetime (datetime64[ns])
        - PULocationID (int)
        - DOLocationID (int)

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If the parquet file is missing required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Check columns exist before reading to avoid cryptic pyarrow errors
    schema = pd.read_parquet(filepath, columns=[]).columns.tolist()
    # read_parquet with empty columns list gives us schema; fall back to full read
    all_cols = pd.read_parquet(filepath).columns.tolist()
    missing = [c for c in REQUIRED_COLUMNS if c not in all_cols]
    if missing:
        raise ValueError(f"Parquet file missing required columns: {missing}")

    df = pd.read_parquet(filepath, columns=REQUIRED_COLUMNS)

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df = df.dropna(subset=REQUIRED_COLUMNS)
    df["PULocationID"] = df["PULocationID"].astype(int)
    df["DOLocationID"] = df["DOLocationID"].astype(int)

    # Filter to valid zone IDs
    df = df[(df["PULocationID"] >= 1) & (df["PULocationID"] <= 263)]
    df = df[(df["DOLocationID"] >= 1) & (df["DOLocationID"] <= 263)]

    return df.reset_index(drop=True)


def load_month(year: int, month: int, data_dir: str = "data") -> pd.DataFrame:
    """
    Convenience function: download (if needed) and load a month of HVFHV data.

    Parameters
    ----------
    year : int
        Year of the dataset.
    month : int
        Month of the dataset (1–12).
    data_dir : str
        Local directory for caching parquet files.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame as returned by load().
    """
    filepath = download(year, month, data_dir)
    return load(filepath)
