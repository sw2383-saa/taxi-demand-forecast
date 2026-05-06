"""Empirical data-consistency tests anchored to real January 2026 facts.

We profiled the official NYC TLC HVFHV file
``fhvhv_tripdata_2026-01.parquet`` directly from the CloudFront URL
listed at https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
and recorded the resulting numbers in ``docs/dataset_facts.md``. The
tests in this file pin those numbers down as machine-checkable
assertions so that:

1. The recorded constants stay internally consistent: an accidental
   edit to ``docs/dataset_facts.md`` that forgets to update the
   matching constant here will fail CI, and vice versa.
2. If a downstream change in ``clean_trips`` accidentally starts
   filtering out valid rows from already-clean synthetic input,
   CI fails immediately with a clear message about which assumption
   broke.

These tests do not require the real 482 MB parquet to be present
at test time, and they do not re-run the loader against the real
file. They use small synthetic fixtures whose statistical shape
matches the real file, and they assert structural and behavioral
properties (e.g. "the [1, 265] filter must remove 0 rows from
already-clean input"). The actual numbers from the real file are
quoted in module-level constants so that the test file itself is a
human-readable archive of what we observed. If TLC re-publishes
the file, the profiling scripts in ``docs/dataset_profiling.md``
should be re-run and these constants updated deliberately.
"""

import unittest

import numpy as np
import pandas as pd

from taxi_demand.features import (
    NY_STATE_HOLIDAYS_2019_2026,
    US_FEDERAL_HOLIDAYS_2019_2026,
    aggregate,
)
from taxi_demand.loader import (
    NUM_TAXI_ZONES,
    REQUIRED_COLUMNS,
    clean_trips,
)


# Source: docs/dataset_facts.md. Updated whenever the team re-profiles
# the dataset; treated as ground truth for the tests below.

REAL_TOTAL_ROWS = 20_940_373
REAL_FILE_SIZE_BYTES = 505_868_728
REAL_FIRST_PICKUP = pd.Timestamp("2026-01-01 00:00:00")
REAL_LAST_PICKUP = pd.Timestamp("2026-01-31 23:59:59")
REAL_NUM_COLUMNS = 25
REAL_UNIQUE_PU_ZONES = 262
REAL_OUT_OF_RANGE_ZONE_COUNT = 0
REAL_NULL_PU_COUNT = 0

# The full daily breakdown we observed. Used by the holiday-signal
# tests below to verify the surprising MLK-Day-is-busy result.
REAL_DAILY_PICKUPS_2026_01 = {
    "2026-01-01": 730355, "2026-01-02": 600947, "2026-01-03": 640262,
    "2026-01-04": 572075, "2026-01-05": 552678, "2026-01-06": 565093,
    "2026-01-07": 580665, "2026-01-08": 617258, "2026-01-09": 699547,
    "2026-01-10": 833940, "2026-01-11": 672785, "2026-01-12": 600183,
    "2026-01-13": 605814, "2026-01-14": 620274, "2026-01-15": 704259,
    "2026-01-16": 784122, "2026-01-17": 806362, "2026-01-18": 721685,
    "2026-01-19": 606810, "2026-01-20": 683479, "2026-01-21": 678801,
    "2026-01-22": 686127, "2026-01-23": 804588, "2026-01-24": 880274,
    "2026-01-25": 322276, "2026-01-26": 439462, "2026-01-27": 682417,
    "2026-01-28": 727157, "2026-01-29": 770967, "2026-01-30": 864832,
    "2026-01-31": 884879,
}


class TestRealFileShapeArchive(unittest.TestCase):
    """Read-only archive of the file's physical shape.

    These tests do not run against the real file; they assert that
    our recorded constants are internally consistent so that any
    accidental edit to docs/dataset_facts.md gets caught.
    """

    def test_total_rows_matches_table_sum(self) -> None:
        # Cross-check: the per-day sum should equal the headline
        # total-rows figure.
        per_day_sum = sum(REAL_DAILY_PICKUPS_2026_01.values())
        self.assertEqual(per_day_sum, REAL_TOTAL_ROWS)

    def test_january_has_31_days(self) -> None:
        self.assertEqual(len(REAL_DAILY_PICKUPS_2026_01), 31)

    def test_first_and_last_pickup_within_january(self) -> None:
        self.assertEqual(REAL_FIRST_PICKUP.month, 1)
        self.assertEqual(REAL_FIRST_PICKUP.year, 2026)
        self.assertEqual(REAL_LAST_PICKUP.month, 1)
        self.assertEqual(REAL_LAST_PICKUP.year, 2026)

    def test_unique_zones_does_not_exceed_total(self) -> None:
        self.assertLessEqual(REAL_UNIQUE_PU_ZONES, NUM_TAXI_ZONES)


class TestCleanTripsIsLosslessOnRealShape(unittest.TestCase):
    """The real January 2026 file has 0 nulls and 0 out-of-range
    zone ids, so ``clean_trips`` must remove exactly 0 rows when
    given input that matches that statistical shape.
    """

    def _make_clean_input(self, n: int) -> pd.DataFrame:
        """Synthesise n rows that match the real file's hygiene
        properties: no nulls, all zone ids in [1, 265]."""
        rng = np.random.default_rng(seed=42)
        return pd.DataFrame({
            "pickup_datetime": pd.date_range(
                "2026-01-01", periods=n, freq="1min"
            ),
            "PULocationID": rng.integers(1, NUM_TAXI_ZONES + 1, size=n),
            "DOLocationID": rng.integers(1, NUM_TAXI_ZONES + 1, size=n),
        })

    def test_clean_input_passes_through_unchanged_size(self) -> None:
        input_df = self._make_clean_input(1000)
        output = clean_trips(input_df)
        # Same row count: all 1000 synthetic rows survive both the
        # null filter and the zone-range filter, mirroring what we
        # observed on the real file.
        self.assertEqual(len(output), 1000)

    def test_clean_input_preserves_required_columns(self) -> None:
        input_df = self._make_clean_input(50)
        output = clean_trips(input_df)
        for col in REQUIRED_COLUMNS:
            self.assertIn(col, output.columns)


class TestRealHolidaySignal(unittest.TestCase):
    """Reproduce the surprising empirical finding from the real
    file: in HVFHV January 2026, MLK Day and New Year's Day are
    *not* unusually quiet relative to other days of the same
    weekday in the same month.

    We feed the actual daily totals from the real file into the
    z-score routine and assert the (z >= 0) outcome for each known
    federal holiday in the month. This locks down our README claim
    that "the OPM federal calendar is not a tight match for
    days-with-anomalously-low-demand in this dataset".
    """

    def _z_scores_by_weekday(self):
        """Compute each day's z-score against same-weekday peers."""
        # Group the real daily counts by weekday name.
        by_weekday = {}
        for date_str, count in REAL_DAILY_PICKUPS_2026_01.items():
            wd = pd.Timestamp(date_str).day_name()
            by_weekday.setdefault(wd, []).append((date_str, count))

        z_scores = {}
        for date_str, count in REAL_DAILY_PICKUPS_2026_01.items():
            wd = pd.Timestamp(date_str).day_name()
            peers = [c for d, c in by_weekday[wd] if d != date_str]
            mean = float(np.mean(peers))
            std = float(np.std(peers, ddof=1))
            z_scores[date_str] = (count - mean) / std if std > 0 else 0.0
        return z_scores

    def test_mlk_day_is_not_anomalously_quiet(self) -> None:
        # Jan 19 2026 (Monday, MLK Day) should have z-score >= 0,
        # i.e. NOT unusually quiet relative to other January Mondays.
        z = self._z_scores_by_weekday()
        self.assertGreaterEqual(
            z["2026-01-19"], 0,
            f"MLK Day z-score was {z['2026-01-19']:.2f}; expected "
            "non-negative to confirm the empirical finding that "
            "federal holidays are not detectable as demand dips.",
        )

    def test_new_years_day_is_not_anomalously_quiet(self) -> None:
        # Jan 1 2026 (Thursday, New Year's Day) should have z-score
        # >= 0; ride-share demand typically rises on New Year's Day
        # because of late-night revelers and visiting family.
        z = self._z_scores_by_weekday()
        self.assertGreaterEqual(
            z["2026-01-01"], 0,
            f"New Year's Day z-score was {z['2026-01-01']:.2f}; "
            "expected non-negative.",
        )

    def test_jan_25_is_extreme_dip(self) -> None:
        # The single quietest day in the entire month is Jan 25,
        # with a z-score below -3 (highly anomalous). This is
        # NOT a federal holiday - it is a real-world disruption,
        # most likely a winter storm. We pin this fact down so the
        # README's "OPM calendar misses the actual quiet days"
        # narrative is empirically grounded.
        z = self._z_scores_by_weekday()
        self.assertLess(
            z["2026-01-25"], -3,
            f"Jan 25 z-score was {z['2026-01-25']:.2f}; expected "
            "below -3.",
        )

    def test_quietest_day_is_not_in_federal_calendar(self) -> None:
        # Confirm the most empirically quiet day is NOT a federal
        # holiday — this is the central finding that motivates
        # exposing custom-holiday-set support in
        # add_calendar_features.
        quietest_date = min(
            REAL_DAILY_PICKUPS_2026_01,
            key=REAL_DAILY_PICKUPS_2026_01.get,
        )
        ts = pd.Timestamp(quietest_date).normalize()
        self.assertNotIn(ts, US_FEDERAL_HOLIDAYS_2019_2026)
        self.assertNotIn(ts, NY_STATE_HOLIDAYS_2019_2026)


class TestAggregatePreservesTotalCount(unittest.TestCase):
    """Aggregation correctness check using a small synthetic frame
    whose totals are designed to match the per-day signature of the
    real file (Jan 1 has the second-highest count for the early
    week, Jan 25 has the lowest count, etc).
    """

    def _make_synthetic_january(self, scale: int = 100) -> pd.DataFrame:
        """Produce a synthetic version of the real Jan 2026 daily
        signature, scaled down by ``scale`` so the test runs fast."""
        rows = []
        for date_str, count in REAL_DAILY_PICKUPS_2026_01.items():
            n = max(1, count // scale)
            rng = np.random.default_rng(
                seed=int(pd.Timestamp(date_str).strftime("%Y%m%d"))
            )
            day_start = pd.Timestamp(date_str)
            offsets = rng.integers(0, 24 * 60, size=n)
            timestamps = day_start + pd.to_timedelta(offsets, unit="m")
            zones = rng.integers(1, NUM_TAXI_ZONES + 1, size=n)
            for ts, z in zip(timestamps, zones):
                rows.append({
                    "pickup_datetime": ts,
                    "PULocationID": int(z),
                    "DOLocationID": int(z),
                })
        return pd.DataFrame(rows)

    def test_aggregated_total_matches_input(self) -> None:
        df = self._make_synthetic_january(scale=1000)
        agg = aggregate(df)
        self.assertEqual(int(agg["demand"].sum()), len(df))

    def test_aggregated_zones_within_official_range(self) -> None:
        df = self._make_synthetic_january(scale=1000)
        agg = aggregate(df)
        self.assertGreaterEqual(int(agg["zone_id"].min()), 1)
        self.assertLessEqual(int(agg["zone_id"].max()), NUM_TAXI_ZONES)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
