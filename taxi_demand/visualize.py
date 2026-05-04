import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_forecast(df: pd.DataFrame, zone_id: int, y_pred: np.ndarray, save_path: str = None) -> None:
    """Plot actual vs predicted demand for a single NYC taxi zone.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least zone_id, hour, and demand columns.
    zone_id : int
        NYC taxi zone ID to filter and plot.
    y_pred : np.ndarray
        Predicted demand values for the selected zone.
    save_path : str, optional
        Path where the figure should be saved. If None, the figure is not saved.

    Returns
    -------
    None
        The function creates a matplotlib figure, optionally saves it, and closes it.
    """
    zone_df = df[df["zone_id"] == zone_id].sort_values("hour")

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
    """Plot a heatmap of average hourly demand for the top 20 NYC taxi zones.

    The heatmap shows zone IDs as rows and hour-of-day values from 0 to 23 as
    columns. Zones are selected based on total demand.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least zone_id, hour, and demand columns.
    save_path : str, optional
        Path where the figure should be saved. If None, the figure is not saved.

    Returns
    -------
    None
        The function creates a matplotlib heatmap, optionally saves it, and closes it.
    """
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour

    pivot = df.pivot_table(
        index="zone_id",
        columns="hour_of_day",
        values="demand",
        aggfunc="mean",
    )

    top_zones = df.groupby("zone_id")["demand"].sum().sort_values(ascending=False).head(20).index
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
