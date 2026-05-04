"""
features.py
-----------
Aggregates cleaned HVFHV trip data into hourly demand by taxi zone
and computes lag features for model training.
"""

import pandas as pd


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip records into hourly pickup counts per zone.

    Groups by taxi zone and hour, counting the number of pickups.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from loader.load(), with columns:
        pickup_datetime (datetime64[ns]), PULocationID (int).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
        - zone_id (int): NYC taxi zone ID
        - hour (datetime64[ns]): timestamp floored to the hour
        - demand (int): number of pickups in that zone-hour

    Raises
    ------
    ValueError
        If required columns are missing from df.
    """
    required = ["pickup_datetime", "PULocationID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["hour"] = df["pickup_datetime"].dt.floor("h")

    agg = (
        df.groupby(["PULocationID", "hour"])
        .size()
        .reset_index(name="demand")
        .rename(columns={"PULocationID": "zone_id"})
    )

    agg["zone_id"] = agg["zone_id"].astype(int)
    agg["demand"] = agg["demand"].astype(int)
    agg = agg.sort_values(["zone_id", "hour"]).reset_index(drop=True)

    return agg


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features to the aggregated hourly demand DataFrame.

    For each zone, computes:
    - lag_1h: demand 1 hour prior
    - lag_24h: demand 24 hours prior

    Rows where lag values are unavailable are left as NaN
    (typically the first 24 hours of the dataset per zone).

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame as returned by aggregate(), with columns:
        zone_id (int), hour (datetime64[ns]), demand (int).

    Returns
    -------
    pd.DataFrame
        Same DataFrame with two additional columns:
        - lag_1h (float): demand 1 hour prior for that zone
        - lag_24h (float): demand 24 hours prior for that zone

    Raises
    ------
    ValueError
        If required columns are missing from df.
    """
    required = ["zone_id", "hour", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy().sort_values(["zone_id", "hour"]).reset_index(drop=True)

    df["lag_1h"] = (
        df.groupby("zone_id")["demand"]
        .shift(1)
        .astype(float)
    )
    df["lag_24h"] = (
        df.groupby("zone_id")["demand"]
        .shift(24)
        .astype(float)
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: aggregate raw trips into hourly demand and add lag features.

    Convenience function combining aggregate() and add_lags().

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from loader.load().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - zone_id (int)
        - hour (datetime64[ns])
        - demand (int)
        - lag_1h (float)
        - lag_24h (float)
    """
    agg = aggregate(df)
    return add_lags(agg)
