"""
parallel.py
-----------
Apply a per-zone feature function in parallel across pickup zones.

Computing lag and rolling features per zone is embarrassingly parallel:
each zone's slice of the demand DataFrame is independent of the
others. The course covered ``multiprocessing.Pool`` for exactly this
kind of workload in Week 7-8 (Homework 3's ``approx_pi`` and
``compute_word_frequency`` followed the same template).

The wrapper here -- :func:`parallel_apply_per_zone` -- accepts an
arbitrary function ``func(zone_df) -> zone_df`` and applies it via
``Pool.map``. The output is sorted on ``(zone_id, hour)`` so the
parallel and sequential code paths produce *identical* DataFrames.
That equivalence property is asserted in the unit tests.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Callable, Optional

import pandas as pd

ZoneFunc = Callable[[pd.DataFrame], pd.DataFrame]


def _apply_one(args):
    """Worker entry point used by ``Pool.map``.

    The worker receives a tuple because ``Pool.map`` only passes a
    single argument per call.
    """
    func, sub_df = args
    return func(sub_df)


def parallel_apply_per_zone(
    df: pd.DataFrame,
    func: ZoneFunc,
    n_workers: Optional[int] = None,
    group_column: str = "zone_id",
    time_column: str = "hour",
) -> pd.DataFrame:
    """Apply ``func`` to each zone-slice in parallel and concatenate.

    The output is sorted on ``(group_column, time_column)`` so callers
    receive a deterministic ordering regardless of worker count or
    completion order.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format demand DataFrame.
    func : callable
        Function with signature ``func(zone_df) -> zone_df``. Must be
        importable at module level (multiprocessing pickles it).
    n_workers : int or None
        Number of worker processes. ``None`` falls back to
        ``os.cpu_count()`` (or 1 if unavailable). Must be positive
        when given explicitly.
    group_column : str
        Column to split the input on. Defaults to ``"zone_id"``.
    time_column : str
        Column used for the final sort. Defaults to ``"hour"``.

    Returns
    -------
    pd.DataFrame
        Concatenated and sorted DataFrame.

    Raises
    ------
    ValueError
        If ``df`` is missing the required columns or if ``n_workers``
        is non-positive.
    """
    if group_column not in df.columns:
        raise ValueError(f"DataFrame missing group column {group_column!r}.")
    if time_column not in df.columns:
        raise ValueError(f"DataFrame missing time column {time_column!r}.")

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    if not isinstance(n_workers, int) or n_workers <= 0:
        raise ValueError(
            f"n_workers must be a positive integer, got {n_workers!r}."
        )

    if df.empty:
        return df.copy()

    # Partition the DataFrame into per-zone slices in deterministic
    # order. We make explicit copies so worker processes never share
    # references with the parent.
    zone_dfs = [
        sub_df.copy()
        for _, sub_df in df.sort_values(group_column).groupby(
            group_column, sort=True
        )
    ]

    if n_workers == 1 or len(zone_dfs) <= 1:
        # Sequential fast-path: avoids the cost of spawning workers
        # for tiny inputs and is the only safe choice if the caller
        # is itself running inside a multiprocessing pool.
        results = [func(sub_df) for sub_df in zone_dfs]
    else:
        worker_count = min(n_workers, len(zone_dfs))
        with multiprocessing.Pool(processes=worker_count) as pool:
            results = pool.map(
                _apply_one, [(func, sub_df) for sub_df in zone_dfs]
            )

    if not results:
        return df.iloc[0:0].copy()

    combined = pd.concat(results, ignore_index=True)
    return combined.sort_values([group_column, time_column]).reset_index(drop=True)
