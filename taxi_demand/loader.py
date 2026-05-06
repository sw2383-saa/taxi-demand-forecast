"""
loader.py
---------
Downloads NYC TLC High Volume FHV (HVFHV) trip parquet files from the
official NYC TLC site and loads them into pandas DataFrames.

Two separate concerns are exposed as separate functions, in line with
the course-forum guidance that "we should also scrape all raw data
even if some appears to be erroneous, and then only filter them out
later during the analysis":

* ``download``       -- fetch a monthly parquet, with retry, atomic
                        write, and User-Agent header.
* ``load_raw``       -- read the parquet straight into a DataFrame
                        with no row-level cleaning.
* ``clean_trips``    -- apply the project-specific cleaning rules
                        (drop nulls, narrow to valid zone ids).
* ``load``           -- backwards-compatible convenience wrapper
                        that calls ``load_raw`` then ``clean_trips``.
* ``load_month``     -- highest-level helper that downloads (if needed)
                        and returns the cleaned DataFrame.
"""

import os
import time
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests


# Official NYC TLC CloudFront URL pattern. The HTML landing page at
# https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page links
# every monthly file at this URL pattern.
HVFHV_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "fhvhv_tripdata_{year}-{month:02d}.parquet"
)

# The pickup time, pickup zone, and dropoff zone are the three columns
# we actually need downstream. We keep the list as a module-level
# constant so callers can introspect it (and tests can monkey-patch
# it if a future schema change forces it).
REQUIRED_COLUMNS = ["pickup_datetime", "PULocationID", "DOLocationID"]

# NYC TLC publishes 265 distinct taxi-zone ids (1..265). The earlier
# version of this loader filtered to 1..263, which silently dropped
# zones 264 and 265; we corrected that here.
NUM_TAXI_ZONES = 265

# Default User-Agent. The TLC CDN has historically rejected the
# default ``python-requests/x.y`` UA; identifying ourselves cleanly
# is both more polite and more robust.
DEFAULT_USER_AGENT = (
    "taxi-demand-forecast/0.2 (academic project; ORIE 5270 final)"
)


def download(
    year: int,
    month: int,
    data_dir: str = "data",
    *,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float = 120.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> str:
    """
    Download the HVFHV parquet for a given year and month.

    The download is *atomic*: we stream into ``filepath + '.part'`` and
    rename only on success, so an interrupted download will not leave a
    truncated file masquerading as a complete one. We retry up to
    ``max_retries`` times with exponential backoff on transient HTTP
    errors. If the destination file already exists we skip the network
    entirely (useful for cached test fixtures).

    Parameters
    ----------
    year : int
        Year of the dataset. The HVFHV dataset starts in February 2019
        and we accept anything up to the current year (the upper bound
        is intentionally permissive so the function does not need a
        yearly maintenance bump).
    month : int
        Month of the dataset, in ``[1, 12]``.
    data_dir : str
        Local directory to store downloaded files.
    max_retries : int, keyword-only
        Maximum number of attempts per URL. Must be at least one.
    backoff_seconds : float, keyword-only
        Initial sleep before retrying. Doubled after each failure.
    timeout : float, keyword-only
        Per-request timeout, in seconds.
    user_agent : str, keyword-only
        ``User-Agent`` request header.

    Returns
    -------
    str
        Path to the downloaded (or already-cached) parquet file.

    Raises
    ------
    ValueError
        If ``year``/``month`` are out of range, or if any of the retry
        / timeout knobs are non-positive.
    requests.HTTPError
        If every attempt returns a non-200 response. The most recent
        underlying exception is chained as the cause.
    """
    if not (2019 <= year <= 2099):
        raise ValueError(f"year must be between 2019 and 2099, got {year}")
    if not (1 <= month <= 12):
        raise ValueError(f"month must be between 1 and 12, got {month}")
    if year == 2019 and month < 2:
        # Local Law 149 of 2018 went into effect February 1, 2019,
        # so the HVFHV trip-record series starts in February 2019.
        # The CDN does not host a January 2019 file; we fail early
        # with a clear message instead of letting the request 404.
        raise ValueError(
            "HVFHV trip records start in February 2019; "
            "no January 2019 file exists."
        )
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    if backoff_seconds < 0:
        raise ValueError(f"backoff_seconds must be >= 0, got {backoff_seconds}")
    if timeout <= 0:
        raise ValueError(f"timeout must be > 0, got {timeout}")

    os.makedirs(data_dir, exist_ok=True)
    filename = f"fhvhv_tripdata_{year}-{month:02d}.parquet"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        # Cached on disk -- skip the network entirely.
        return filepath

    url = HVFHV_URL.format(year=year, month=month)
    tmp_path = filepath + ".part"

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": user_agent},
                stream=True,
                timeout=timeout,
            )
            response.raise_for_status()

            # Atomic write: stream into .part, rename on success only.
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp_path, filepath)
            return filepath

        except Exception as exc:
            last_exc = exc
            # Clean up the partial file so a retry starts fresh.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if attempt < max_retries:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    raise requests.HTTPError(
        f"Failed to download {url} after {max_retries} attempts."
    ) from last_exc


def _read_schema_columns(filepath: str) -> list:
    """Read just the parquet schema (a few KiB), not the full file.

    This replaces the expensive double-read pattern in the previous
    version of the loader, which materialised the full DataFrame just
    to inspect column names.
    """
    return list(pq.read_schema(filepath).names)


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Read an HVFHV parquet into a DataFrame *without* any row filtering.

    Only column projection is performed (we keep just the three columns
    we use downstream, to keep memory bounded). We do not drop nulls,
    we do not filter zone ids, and we do not coerce dtypes beyond the
    parquet defaults. The forum guidance from Professor Zhang
    (#81: "we should also scrape all raw data even if some appears to
    be erroneous, and then only filter them out later during the
    analysis") motivates keeping a load function that is faithful to
    the on-disk data.

    Parameters
    ----------
    filepath : str
        Path to a local HVFHV parquet file.

    Returns
    -------
    pd.DataFrame
        The three required columns, with rows in the original on-disk
        order and *no* filtering applied.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If the parquet is missing any of the required columns.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    available = _read_schema_columns(filepath)
    missing = [c for c in REQUIRED_COLUMNS if c not in available]
    if missing:
        raise ValueError(f"Parquet file missing required columns: {missing}")

    return pd.read_parquet(filepath, columns=REQUIRED_COLUMNS)


def clean_trips(
    df: pd.DataFrame,
    *,
    num_zones: int = NUM_TAXI_ZONES,
) -> pd.DataFrame:
    """
    Apply project-level cleaning to a raw HVFHV DataFrame.

    Cleaning rules:

    1. Coerce ``pickup_datetime`` to ``datetime64[ns]`` (null values
       are kept as ``NaT`` for the next step).
    2. Drop rows where any of the three required columns is null.
    3. Cast ``PULocationID`` and ``DOLocationID`` to ``int64``.
    4. Restrict to NYC taxi-zone ids in ``[1, num_zones]``. The
       parameter ``num_zones`` defaults to ``NUM_TAXI_ZONES`` (265)
       so the function can be tested with a smaller synthetic universe.
    5. Reset the index so callers get sequential row positions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame, typically the output of :func:`load_raw`.
    num_zones : int, keyword-only
        Upper bound on valid zone ids (inclusive). Useful in tests
        that build a smaller synthetic universe.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with three columns, no nulls, all zone ids
        within ``[1, num_zones]``, and a fresh integer index.

    Raises
    ------
    ValueError
        If the input is missing any of the required columns or if
        ``num_zones`` is non-positive.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    if num_zones <= 0:
        raise ValueError(f"num_zones must be > 0, got {num_zones}")

    out = df.copy()
    out["pickup_datetime"] = pd.to_datetime(out["pickup_datetime"], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS)
    out["PULocationID"] = out["PULocationID"].astype("int64")
    out["DOLocationID"] = out["DOLocationID"].astype("int64")
    out = out[(out["PULocationID"] >= 1) & (out["PULocationID"] <= num_zones)]
    out = out[(out["DOLocationID"] >= 1) & (out["DOLocationID"] <= num_zones)]
    return out.reset_index(drop=True)


def load(filepath: str) -> pd.DataFrame:
    """
    Backwards-compatible wrapper: ``load_raw`` followed by ``clean_trips``.

    This preserves the original API used in the team's first round of
    tests. The two-step variant (``load_raw`` then ``clean_trips``) is
    recommended for new code because it keeps the raw and cleaned views
    of the data clearly separated.

    Parameters
    ----------
    filepath : str
        Path to a local HVFHV parquet file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame as returned by :func:`clean_trips`.

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If required columns are missing.
    """
    return clean_trips(load_raw(filepath))


def load_month(year: int, month: int, data_dir: str = "data") -> pd.DataFrame:
    """
    End-to-end helper: download (if not cached) and load a month of HVFHV.

    Parameters
    ----------
    year : int
        Year of the dataset.
    month : int
        Month of the dataset, in ``[1, 12]``.
    data_dir : str
        Local directory used for caching.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame as returned by :func:`load`.
    """
    return load(download(year, month, data_dir))
