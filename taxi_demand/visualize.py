"""Plotting helpers for the demand-forecast pipeline.

The two functions here mirror the team's original API:

* :func:`plot_forecast`        -- actual vs. predicted demand for a
  single pickup zone.
* :func:`plot_demand_heatmap`  -- average hourly demand by hour-of-day
  for the top 20 busiest zones.

The only behavioural change relative to the team's first pass is that
:func:`plot_forecast` now validates that the prediction array has the
same length as the rows it will be plotted against. Without that
check, a length mismatch silently produced a misaligned plot rather
than an explicit error.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_forecast(
    df: pd.DataFrame,
    zone_id: int,
    y_pred: np.ndarray,
    save_path: str = None,
) -> None:
    """Plot actual vs. predicted demand for a single NYC taxi zone.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``zone_id``, ``hour``, ``demand``.
    zone_id : int
        NYC taxi zone ID to filter and plot.
    y_pred : np.ndarray
        Predicted demand values for the selected zone. Must have the
        same length as the rows of ``df`` after filtering on
        ``zone_id``.
    save_path : str, optional
        Path where the figure should be saved. If None, the figure is
        not saved but is still closed.

    Returns
    -------
    None
        Creates a matplotlib figure, optionally saves it, then closes
        it so callers do not leak figure handles.

    Raises
    ------
    ValueError
        If a required column is missing from ``df``, or if
        ``y_pred`` does not match the length of the filtered slice.
    """
    required = ["zone_id", "hour", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    zone_df = df[df["zone_id"] == zone_id].sort_values("hour")
    y_pred = np.asarray(y_pred).ravel()
    if len(y_pred) != len(zone_df):
        raise ValueError(
            f"y_pred has length {len(y_pred)} but the filter for "
            f"zone_id={zone_id} yielded {len(zone_df)} rows."
        )

    plt.figure()
    plt.plot(zone_df["hour"], zone_df["demand"], label="Actual")
    plt.plot(zone_df["hour"], y_pred, label="Predicted")
    plt.xlabel("Hour")
    plt.ylabel("Demand")
    plt.title(f"Demand Forecast – Zone {zone_id}")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_demand_heatmap(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot a heatmap of average hourly demand for the top 20 zones.

    Rows are zone ids and columns are hours of day (0..23). Zones are
    selected by total demand, descending; if fewer than 20 zones
    exist in ``df`` we plot all of them.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``zone_id``, ``hour``, ``demand``. The
        ``hour`` column must be a datetime-like dtype because we
        extract the hour-of-day from it via ``.dt.hour``.
    save_path : str, optional
        Path where the figure should be saved. If None, the figure is
        not saved but is still closed.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a required column is missing, or if ``hour`` is not a
        datetime-like dtype.
    """
    required = ["zone_id", "hour", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["hour"]):
        raise ValueError(
            "'hour' column must be datetime-like; got dtype "
            f"{df['hour'].dtype}. Pass df['hour'] = pd.to_datetime(...)"
            " before calling plot_demand_heatmap."
        )

    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour

    pivot = df.pivot_table(
        index="zone_id",
        columns="hour_of_day",
        values="demand",
        aggfunc="mean",
    )

    top_zones = (
        df.groupby("zone_id")["demand"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
    )
    pivot = pivot.loc[top_zones]

    plt.figure()
    plt.imshow(pivot, aspect="auto")
    plt.colorbar()
    plt.xlabel("Hour of Day")
    plt.ylabel("Zone ID")
    plt.title("Average Hourly Demand by Zone (Top 20)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)

    if save_path:
        plt.savefig(save_path)
    plt.close()
