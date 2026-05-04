"""
tests/test_visualize.py
-----------------------
Tests for taxi_demand.visualize — uses synthetic DataFrames only.
No display window is opened during tests.
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, must be set before pyplot import
import matplotlib.pyplot as plt

from taxi_demand.visualize import plot_forecast, plot_demand_heatmap


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_df(zones=None, n_hours=48):
    """Synthetic aggregated DataFrame matching the features.py output contract."""
    if zones is None:
        zones = [161, 48, 79]
    rows = []
    for zone in zones:
        hours = pd.date_range("2026-01-01", periods=n_hours, freq="h")
        demand = np.random.randint(10, 200, n_hours)
        rows.append(pd.DataFrame({
            "zone_id": zone,
            "hour": hours,
            "demand": demand,
            "lag_1h": np.roll(demand.astype(float), 1),
            "lag_24h": np.roll(demand.astype(float), 24),
        }))
    return pd.concat(rows, ignore_index=True)


# ── plot_forecast() ───────────────────────────────────────────────────────────

def test_plot_forecast_runs_without_error():
    df = make_df()
    zone_df = df[df["zone_id"] == 161].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    plot_forecast(df, zone_id=161, y_pred=y_pred)


def test_plot_forecast_no_open_figures_after_call():
    """plot_forecast must call plt.close() — no lingering figures."""
    df = make_df()
    zone_df = df[df["zone_id"] == 161].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    plot_forecast(df, zone_id=161, y_pred=y_pred)
    assert len(plt.get_fignums()) == 0


def test_plot_forecast_saves_file(tmp_path):
    df = make_df()
    zone_df = df[df["zone_id"] == 161].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    save_path = str(tmp_path / "forecast.png")
    plot_forecast(df, zone_id=161, y_pred=y_pred, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def test_plot_forecast_no_file_without_save_path(tmp_path):
    df = make_df()
    zone_df = df[df["zone_id"] == 161].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    plot_forecast(df, zone_id=161, y_pred=y_pred)
    # No file should appear in tmp_path
    assert len(list(tmp_path.iterdir())) == 0


def test_plot_forecast_different_zone():
    df = make_df()
    zone_df = df[df["zone_id"] == 48].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    plot_forecast(df, zone_id=48, y_pred=y_pred)
    assert len(plt.get_fignums()) == 0


def test_plot_forecast_returns_none():
    df = make_df()
    zone_df = df[df["zone_id"] == 161].sort_values("hour")
    y_pred = np.random.randint(10, 200, len(zone_df))
    result = plot_forecast(df, zone_id=161, y_pred=y_pred)
    assert result is None


# ── plot_demand_heatmap() ─────────────────────────────────────────────────────

def test_plot_demand_heatmap_runs_without_error():
    df = make_df(zones=list(range(1, 25)))  # 24 zones
    plot_demand_heatmap(df)


def test_plot_demand_heatmap_no_open_figures_after_call():
    df = make_df(zones=list(range(1, 25)))
    plot_demand_heatmap(df)
    assert len(plt.get_fignums()) == 0


def test_plot_demand_heatmap_saves_file(tmp_path):
    df = make_df(zones=list(range(1, 25)))
    save_path = str(tmp_path / "heatmap.png")
    plot_demand_heatmap(df, save_path=save_path)
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def test_plot_demand_heatmap_no_file_without_save_path(tmp_path):
    df = make_df(zones=list(range(1, 25)))
    plot_demand_heatmap(df)
    assert len(list(tmp_path.iterdir())) == 0


def test_plot_demand_heatmap_returns_none():
    df = make_df(zones=list(range(1, 25)))
    result = plot_demand_heatmap(df)
    assert result is None


def test_plot_demand_heatmap_fewer_than_20_zones():
    """Should work fine with fewer than 20 zones."""
    df = make_df(zones=[1, 2, 3])
    plot_demand_heatmap(df)
    assert len(plt.get_fignums()) == 0
