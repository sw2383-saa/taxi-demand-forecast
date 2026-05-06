"""
features.py
-----------
Aggregates cleaned HVFHV trip data into hourly demand by taxi zone and
computes the time-series features used by the forecasting model.

The original three public functions (``aggregate``, ``add_lags``,
``build_features``) keep their signatures so all of the team's
existing tests continue to pass without modification. The module also
exposes:

* ``add_calendar_features`` -- hour-of-day, day-of-week, weekend, and
  holiday indicators.
* ``add_rolling_features``  -- causal rolling-mean and rolling-std
  aggregates per zone.
* A more configurable ``build_features`` that accepts custom lag
  offsets, rolling windows, and an opt-in calendar pass.

A note on the holiday calendar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The NYC TLC HVFHV dataset is a stream of trip records and contains no
explicit holiday classification of its own. Whether a given day "is a
holiday" is therefore a modelling choice, not a property of the data.
This module exposes two pre-built calendars and accepts any custom
iterable so the team can pick the rule that matches their analysis:

* :data:`US_FEDERAL_HOLIDAYS_2019_2026` -- the OPM / 5 U.S.C. §6103
  federal calendar (default).
* :data:`NY_STATE_HOLIDAYS_2019_2026`  -- the federal calendar plus
  the three NY-State-only observances (Lincoln's Birthday, Election
  Day, Flag Day) per NY General Construction Law §24.

For the team's default workflow (January 2026) the two calendars
agree on every observed day, so the choice has no observable effect
on the default analysis; it only matters when the project is rerun
on other months.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


# US federal holidays (observed dates) for 2019-2026, the years the
# HVFHV dataset covers. Ships statically so the package install does
# not require ``holidays`` or ``pandas-market-calendars`` and so the
# unit tests are reproducible across environments.
#
# This list follows the OPM / 5 U.S.C. §6103 *federal* observance
# rule. When a fixed-date holiday falls on Saturday it is observed
# the preceding Friday; when it falls on Sunday it is observed the
# following Monday. The list was machine-generated from those rules
# and then cross-checked against the OPM published calendars.
US_FEDERAL_HOLIDAYS_2019_2026 = frozenset(
    pd.Timestamp(d).normalize() for d in [
        # 2019
        "2019-01-01", "2019-01-21", "2019-02-18", "2019-05-27",
        "2019-07-04", "2019-09-02", "2019-10-14", "2019-11-11",
        "2019-11-28", "2019-12-25",
        # 2020
        "2020-01-01", "2020-01-20", "2020-02-17", "2020-05-25",
        "2020-07-03", "2020-09-07", "2020-10-12", "2020-11-11",
        "2020-11-26", "2020-12-25",
        # 2021
        "2021-01-01", "2021-01-18", "2021-02-15", "2021-05-31",
        "2021-06-18", "2021-07-05", "2021-09-06", "2021-10-11",
        "2021-11-11", "2021-11-25", "2021-12-24",
        # 2021-12-31 is the federally observed New Year's Day for
        # 2022 (since January 1, 2022 fell on a Saturday).
        "2021-12-31",
        # 2022
        "2022-01-17", "2022-02-21", "2022-05-30", "2022-06-20",
        "2022-07-04", "2022-09-05", "2022-10-10", "2022-11-11",
        "2022-11-24", "2022-12-26",
        # 2023
        "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
        "2023-06-19", "2023-07-04", "2023-09-04", "2023-10-09",
        "2023-11-10", "2023-11-23", "2023-12-25",
        # 2024
        "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
        "2024-06-19", "2024-07-04", "2024-09-02", "2024-10-14",
        "2024-11-11", "2024-11-28", "2024-12-25",
        # 2025
        "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13",
        "2025-11-11", "2025-11-27", "2025-12-25",
        # 2026
        "2026-01-01", "2026-01-19", "2026-02-16", "2026-05-25",
        "2026-06-19", "2026-07-03", "2026-09-07", "2026-10-12",
        "2026-11-11", "2026-11-26", "2026-12-25",
    ]
)


# New York State public holidays (NY General Construction Law §24).
# This is the federal list above PLUS three NY-specific observances
# that the federal calendar does not include:
#
# * Lincoln's Birthday  -- February 12 each year (per the statute,
#   "the twelfth day of February").
# * Flag Day            -- the second Sunday of June each year
#   (per the statute, "the second Sunday in June, known as Flag
#   day"). Note: federal Flag Day is June 14 each year by
#   presidential proclamation, but NY State law fixes it to the
#   second Sunday instead, which only coincides with June 14 in
#   2020 and 2026.
# * Election Day        -- the first Tuesday after the first Monday
#   of November (state-employee holiday, per the statute "each
#   general election day").
#
# The dataset itself does not classify holidays, so for projects
# that want to model demand against the NY State calendar instead
# of the federal calendar, pass this set as the ``holidays``
# argument to :func:`add_calendar_features` or
# :func:`make_calendar_features`.
NY_STATE_HOLIDAYS_2019_2026 = frozenset(
    list(US_FEDERAL_HOLIDAYS_2019_2026) + [
        pd.Timestamp(d).normalize() for d in [
            # Lincoln's Birthday (Feb 12 each year)
            "2019-02-12", "2020-02-12", "2021-02-12", "2022-02-12",
            "2023-02-12", "2024-02-12", "2025-02-12", "2026-02-12",
            # Flag Day, NY State definition: second Sunday in June.
            # (federal Flag Day is June 14 by presidential
            # proclamation; NY GCL §24 uses the second-Sunday rule
            # instead.)
            "2019-06-09",  # 2nd Sunday June 2019
            "2020-06-14",  # 2nd Sunday June 2020 (= June 14)
            "2021-06-13",  # 2nd Sunday June 2021
            "2022-06-12",  # 2nd Sunday June 2022
            "2023-06-11",  # 2nd Sunday June 2023
            "2024-06-09",  # 2nd Sunday June 2024
            "2025-06-08",  # 2nd Sunday June 2025
            "2026-06-14",  # 2nd Sunday June 2026 (= June 14)
            # Election Day (first Tuesday after first Monday of November)
            "2019-11-05", "2020-11-03", "2021-11-02", "2022-11-08",
            "2023-11-07", "2024-11-05", "2025-11-04", "2026-11-03",
        ]
    ]
)


# Backwards-compatible alias. Earlier versions of this module exposed
# the constant as ``_DEFAULT_US_HOLIDAYS``; we keep that name pointing
# at the same object so any code (or test) that imported the older
# spelling continues to work.
_DEFAULT_US_HOLIDAYS = US_FEDERAL_HOLIDAYS_2019_2026


def aggregate(
    df: pd.DataFrame,
    *,
    fill_missing_hours: bool = False,
) -> pd.DataFrame:
    """
    Aggregate trip records into hourly pickup counts per zone.

    Groups by taxi zone and hour, counting the number of pickups.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from loader.load(), with columns:
        pickup_datetime (datetime64[ns]), PULocationID (int).
    fill_missing_hours : bool, keyword-only
        When False (default), the output contains only those
        ``(zone_id, hour)`` cells that actually had at least one
        pickup. This matches the team's original behavior and is
        memory-efficient for sparse data.

        When True, the output is expanded to a complete hourly grid
        for each zone -- every hour from the earliest pickup to
        the latest pickup gets a row, with ``demand=0`` filled in
        for cells that had no pickups. This is the correct setting
        for downstream lag features: with the grid filled,
        ``lag_24h`` truly means "24 hours earlier" rather than
        "24 records earlier", which can otherwise misalign for
        zones whose hourly stream contains gaps.

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

    if fill_missing_hours and len(agg) > 0:
        # Expand to the complete (zone_id, hour) grid documented in
        # the docstring, filling absent cells with demand=0.
        zone_ids = sorted(agg["zone_id"].unique())
        hour_range = pd.date_range(
            start=agg["hour"].min(),
            end=agg["hour"].max(),
            freq="h",
        )
        full_grid = pd.MultiIndex.from_product(
            [zone_ids, hour_range],
            names=["zone_id", "hour"],
        ).to_frame(index=False)
        agg = full_grid.merge(agg, on=["zone_id", "hour"], how="left")
        agg["demand"] = agg["demand"].fillna(0).astype(int)
        agg["zone_id"] = agg["zone_id"].astype(int)
        agg = agg.sort_values(["zone_id", "hour"]).reset_index(drop=True)

    return agg


def add_lags(
    df: pd.DataFrame,
    lags: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Add lag features to the aggregated hourly demand DataFrame.

    For each zone, computes ``lag_{k}h`` columns shifting ``demand``
    by ``k`` hours. The default lag set ``(1, 24)`` matches the
    original API used by the team's tests; callers can pass any
    positive-integer lag they like.

    Rows where lag values are unavailable are left as NaN (typically
    the warm-up period of the dataset per zone).

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated DataFrame as returned by aggregate(), with columns:
        zone_id (int), hour (datetime64[ns]), demand (int).
    lags : sequence of int, optional
        Lag offsets in hours. Defaults to ``(1, 24)``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with one additional column per requested lag,
        named ``lag_{k}h``. With the default lags, the output gains
        ``lag_1h`` and ``lag_24h``.

    Raises
    ------
    ValueError
        If required columns are missing or any lag is non-positive.
    """
    required = ["zone_id", "hour", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if lags is None:
        lags = (1, 24)
    for k in lags:
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"All lags must be positive integers, got {k!r}")

    df = df.copy().sort_values(["zone_id", "hour"]).reset_index(drop=True)

    grouped = df.groupby("zone_id")["demand"]
    for k in lags:
        df[f"lag_{k}h"] = grouped.shift(k).astype(float)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int] = (24,),
) -> pd.DataFrame:
    """
    Add per-zone rolling-mean and rolling-std features.

    For each window ``w`` we add two columns: ``roll_mean_{w}h`` and
    ``roll_std_{w}h``. The rolling aggregate is *strictly causal*: at
    row ``i`` it summarises rows ``i - w`` through ``i - 1``
    (inclusive), never including row ``i`` itself. This is the
    correct contract for forecasting: when predicting demand at
    hour ``i``, we cannot have access to the actual demand at hour
    ``i`` as part of the feature vector. The rolling window is
    grouped by ``zone_id`` so the aggregate cannot leak across
    zones.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``zone_id``, ``hour``, ``demand``.
    windows : iterable of int
        Window sizes in hours.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with two new columns per window. The first
        row of each zone has ``NaN`` for the rolling mean and
        ``0.0`` for the rolling std because no prior demand exists
        to summarise; subsequent early rows use partial windows
        because we set ``min_periods=1`` on the rolling aggregate.

    Raises
    ------
    ValueError
        If required columns are missing or any window is non-positive.
    """
    required = ["zone_id", "hour", "demand"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    window_list = list(windows)
    if not window_list:
        raise ValueError("windows must contain at least one entry")
    for w in window_list:
        if not isinstance(w, int) or w <= 0:
            raise ValueError(f"All windows must be positive integers, got {w!r}")

    df = df.copy().sort_values(["zone_id", "hour"]).reset_index(drop=True)

    for w in window_list:
        # Shift by 1 before rolling so row i sees only rows i-w..i-1
        # (the strict-causality contract documented in the docstring).
        shifted = df.groupby("zone_id", sort=False)["demand"].shift(1)
        df["_shifted_demand"] = shifted
        rolled = df.groupby("zone_id", sort=False)["_shifted_demand"].rolling(
            window=w, min_periods=1
        )
        df[f"roll_mean_{w}h"] = (
            rolled.mean().reset_index(level=0, drop=True).astype(float)
        )
        df[f"roll_std_{w}h"] = (
            rolled.std(ddof=0)
            .reset_index(level=0, drop=True)
            .fillna(0.0)
            .astype(float)
        )
        df = df.drop(columns=["_shifted_demand"])

    return df


def add_calendar_features(
    df: pd.DataFrame,
    holidays: Optional[Iterable] = None,
) -> pd.DataFrame:
    """
    Add hour-of-day, day-of-week, weekend, and holiday indicators.

    The HVFHV dataset itself does not classify holidays -- it is a
    pure stream of trip records -- so the choice of which dates to
    flag as holidays is a modelling decision rather than a data
    fact. This function exposes that choice through the ``holidays``
    keyword argument.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing an ``hour`` column with datetimes.
    holidays : iterable of date-like, optional
        Custom holiday set. If None, the bundled
        :data:`US_FEDERAL_HOLIDAYS_2019_2026` (the OPM / 5 U.S.C.
        §6103 federal calendar) is used. Pass
        :data:`NY_STATE_HOLIDAYS_2019_2026` for the NY State
        calendar instead, or any other iterable of date-likes for a
        custom rule.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with four new columns:
        - ``hour_of_day`` (int, 0..23)
        - ``day_of_week`` (int, 0..6 with 0 = Monday)
        - ``is_weekend`` (int, 0/1)
        - ``is_holiday`` (int, 0/1)

    Raises
    ------
    ValueError
        If ``hour`` is missing or unparseable.
    """
    if "hour" not in df.columns:
        raise ValueError("DataFrame must contain an 'hour' column.")
    parsed = pd.to_datetime(df["hour"], errors="coerce")
    if parsed.isna().any():
        raise ValueError("Some 'hour' values could not be parsed as datetimes.")

    if holidays is None:
        holiday_set = US_FEDERAL_HOLIDAYS_2019_2026
    else:
        holiday_set = frozenset(pd.Timestamp(h).normalize() for h in holidays)

    out = df.copy()
    out["hour_of_day"] = parsed.dt.hour.astype(int)
    out["day_of_week"] = parsed.dt.dayofweek.astype(int)
    out["is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
    out["is_holiday"] = parsed.dt.normalize().isin(holiday_set).astype(int)
    return out


def build_features(
    df: pd.DataFrame,
    lags: Optional[Sequence[int]] = None,
    rolling_windows: Optional[Sequence[int]] = None,
    add_calendar: bool = False,
    holidays: Optional[Iterable] = None,
    *,
    fill_missing_hours: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: aggregate raw trips, add lag features, optionally
    add rolling and calendar features.

    With its default arguments this function is exactly equivalent to
    the previous version of ``build_features`` -- it returns columns
    ``[zone_id, hour, demand, lag_1h, lag_24h]`` so the team's
    existing test suite is unchanged. New behaviours are unlocked by
    passing optional arguments.

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from loader.load().
    lags : sequence of int, optional
        Lag offsets to compute. Defaults to ``(1, 24)``.
    rolling_windows : sequence of int, optional
        Rolling window sizes in hours. None disables rolling features.
    add_calendar : bool
        If True, append calendar features.
    holidays : iterable of date-like, optional
        Custom holiday set forwarded to :func:`add_calendar_features`.
    fill_missing_hours : bool, keyword-only
        Forwarded to :func:`aggregate`. When True, the aggregated
        frame is expanded to a complete (zone_id, hour) grid before
        lag features are computed, so ``lag_24h`` truly means "24
        hours earlier" rather than "24 records earlier". Strongly
        recommended for forecasting workflows whenever the data has
        any zones with hour-level gaps. Default is ``False`` to
        preserve backward-compatible behavior of the original
        ``build_features``.

    Returns
    -------
    pd.DataFrame
        Engineered DataFrame.
    """
    out = aggregate(df, fill_missing_hours=fill_missing_hours)
    out = add_lags(out, lags=lags)
    if rolling_windows is not None:
        out = add_rolling_features(out, windows=rolling_windows)
    if add_calendar:
        out = add_calendar_features(out, holidays=holidays)
    return out
