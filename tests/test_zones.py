"""Tests for ``taxi_demand.zones``.

Three layers of coverage:

1. ``load_zone_lookup`` against the real bundled CSV and against
   synthetic malformed inputs.
2. ``ZoneResolver`` lookups, edge cases, and the ``annotate``
   workflow that joins zone metadata onto a trip DataFrame.
3. Cross-package invariants pinning the resolver's content to
   facts we verified against the official lookup CSV (e.g. zone
   132 is JFK, zone 138 is LaGuardia, total count is 265).
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from taxi_demand.zones import (
    DEFAULT_LOOKUP_PATH,
    EXPECTED_COLUMNS,
    TAXI_ZONE_LOOKUP_URL,
    ZoneResolver,
    load_zone_lookup,
)
from taxi_demand.loader import NUM_TAXI_ZONES


class TestLoadZoneLookupRealFile(unittest.TestCase):
    """Tests that exercise the actual bundled CSV.

    These tests effectively turn ``data/taxi_zone_lookup.csv``
    into part of the test fixture set: any change to that file
    that breaks the assumed schema will be caught here.
    """

    def test_default_path_loads_265_rows(self) -> None:
        df = load_zone_lookup()
        # The official table has 265 entries spanning IDs 1..265
        # (with 264 = "Unknown" and 265 = "Outside of NYC").
        self.assertEqual(len(df), 265)

    def test_default_path_has_canonical_columns(self) -> None:
        df = load_zone_lookup()
        self.assertEqual(list(df.columns), EXPECTED_COLUMNS)

    def test_location_id_dtype_is_int(self) -> None:
        df = load_zone_lookup()
        self.assertTrue(pd.api.types.is_integer_dtype(df["LocationID"]))

    def test_location_id_range_matches_num_taxi_zones(self) -> None:
        # The lookup's ID range must agree with the constant our
        # loader uses to filter trips. If the two ever drift, this
        # test fires immediately and tells us where to look.
        df = load_zone_lookup()
        self.assertEqual(int(df["LocationID"].min()), 1)
        self.assertEqual(int(df["LocationID"].max()), NUM_TAXI_ZONES)

    def test_returns_a_copy_not_a_shared_reference(self) -> None:
        df1 = load_zone_lookup()
        df2 = load_zone_lookup()
        # If load_zone_lookup were caching a single object, mutating
        # one returned DataFrame would affect the other -- we don't
        # want that.
        self.assertIsNot(df1, df2)


class TestLoadZoneLookupErrorHandling(unittest.TestCase):
    """Tests that exercise the error paths of load_zone_lookup."""

    def test_missing_file_raises_with_helpful_message(self) -> None:
        with self.assertRaises(FileNotFoundError) as cm:
            load_zone_lookup("/no/such/path.csv")
        # The error message should point the user at the official URL
        # so they know where to grab the file from.
        self.assertIn(TAXI_ZONE_LOOKUP_URL, str(cm.exception))

    def test_missing_columns_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.csv"
            # Synthetic CSV that is valid CSV but lacks the
            # required columns.
            pd.DataFrame(
                {"LocationID": [1, 2], "Borough": ["X", "Y"]}
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_zone_lookup(path)

    def test_non_integer_location_id_raises(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_dtype.csv"
            pd.DataFrame({
                "LocationID": ["alpha", "beta"],
                "Borough": ["X", "Y"],
                "Zone": ["Z1", "Z2"],
                "service_zone": ["S1", "S2"],
            }).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_zone_lookup(path)


class TestZoneResolverLookups(unittest.TestCase):
    """Tests of the ZoneResolver class against real lookup data."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_zone_lookup()
        cls.resolver = ZoneResolver(cls.df)

    def test_known_ids_count_matches_dataframe_size(self) -> None:
        self.assertEqual(len(self.resolver.known_ids()), len(self.df))

    def test_jfk_zone_is_resolvable(self) -> None:
        # We pin zone 132 = JFK Airport because this fact also
        # surfaces in our README's top-10 busiest zones table; if
        # the lookup ever gets restructured and zone 132 changes
        # meaning, both the test and the README need updating.
        self.assertEqual(self.resolver.zone(132), "JFK Airport")
        self.assertEqual(self.resolver.borough(132), "Queens")
        self.assertEqual(self.resolver.service_zone(132), "Airports")

    def test_laguardia_zone_is_resolvable(self) -> None:
        self.assertEqual(self.resolver.zone(138), "LaGuardia Airport")
        self.assertEqual(self.resolver.borough(138), "Queens")

    def test_unknown_id_returns_none(self) -> None:
        # "Returning None instead of raising" is part of the public
        # contract -- pin it down so a future refactor doesn't
        # accidentally start raising and break callers that rely on
        # the soft-fail behaviour.
        self.assertIsNone(self.resolver.borough(99999))
        self.assertIsNone(self.resolver.zone(99999))
        self.assertIsNone(self.resolver.service_zone(99999))

    def test_lookup_accepts_int_like_input(self) -> None:
        # A trip DataFrame may carry zone IDs as numpy int64 or
        # even floats; the resolver should be tolerant.
        self.assertEqual(self.resolver.borough(132.0), "Queens")

    def test_label_for_known_zone(self) -> None:
        label = self.resolver.label(132)
        self.assertIn("JFK Airport", label)
        self.assertIn("Queens", label)
        self.assertIn("132", label)

    def test_label_for_unknown_zone(self) -> None:
        label = self.resolver.label(99999)
        self.assertIn("unknown", label.lower())
        self.assertIn("99999", label)

    def test_constructor_rejects_missing_columns(self) -> None:
        partial = self.df.drop(columns=["service_zone"])
        with self.assertRaises(ValueError):
            ZoneResolver(partial)


class TestZoneResolverAnnotate(unittest.TestCase):
    """Tests for the ``annotate`` workflow that joins lookup
    metadata onto a trip-style DataFrame."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resolver = ZoneResolver(load_zone_lookup())

    def test_annotate_default_columns(self) -> None:
        trips = pd.DataFrame(
            {"PULocationID": [4, 132, 99999], "value": [1, 2, 3]}
        )
        out = self.resolver.annotate(trips)
        self.assertIn("PU_borough", out.columns)
        self.assertIn("PU_zone", out.columns)
        self.assertIn("PU_service_zone", out.columns)
        # Zone 4 is Alphabet City, Manhattan, Yellow Zone.
        self.assertEqual(out["PU_zone"].iloc[0], "Alphabet City")
        self.assertEqual(out["PU_borough"].iloc[0], "Manhattan")
        # Zone 132 = JFK as before.
        self.assertEqual(out["PU_zone"].iloc[1], "JFK Airport")
        # Unknown zone -> NaN, not a crash.
        self.assertTrue(pd.isna(out["PU_zone"].iloc[2]))

    def test_annotate_custom_id_and_prefix(self) -> None:
        # Apply the resolver to dropoff zones too, with a different
        # prefix so the new columns don't collide with PU columns.
        trips = pd.DataFrame({"DOLocationID": [4, 132]})
        out = self.resolver.annotate(
            trips, id_column="DOLocationID", prefix="DO"
        )
        self.assertIn("DO_borough", out.columns)
        self.assertEqual(out["DO_borough"].iloc[0], "Manhattan")
        self.assertEqual(out["DO_borough"].iloc[1], "Queens")

    def test_annotate_preserves_non_id_columns(self) -> None:
        trips = pd.DataFrame({
            "PULocationID": [4, 132],
            "demand": [10, 20],
            "hour": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 09:00"]),
        })
        out = self.resolver.annotate(trips)
        # Original columns must still be present and unchanged.
        self.assertIn("demand", out.columns)
        self.assertIn("hour", out.columns)
        self.assertEqual(list(out["demand"]), [10, 20])

    def test_annotate_does_not_mutate_input(self) -> None:
        trips = pd.DataFrame({"PULocationID": [4, 132]})
        original_columns = list(trips.columns)
        self.resolver.annotate(trips)
        # The input frame must not gain new columns -- annotate
        # returns a copy by contract.
        self.assertEqual(list(trips.columns), original_columns)

    def test_annotate_handles_nan_ids(self) -> None:
        trips = pd.DataFrame(
            {"PULocationID": pd.array([4, None], dtype="Int64")}
        )
        out = self.resolver.annotate(trips)
        self.assertEqual(out["PU_borough"].iloc[0], "Manhattan")
        self.assertTrue(pd.isna(out["PU_borough"].iloc[1]))

    def test_annotate_missing_id_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.resolver.annotate(pd.DataFrame({"other": [1]}))


class TestZoneResolverInvariants(unittest.TestCase):
    """Cross-package invariants linking zones, loader, and the
    real-data facts archived in docs/dataset_facts.md.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_zone_lookup()
        cls.resolver = ZoneResolver(cls.df)

    def test_known_id_count_matches_loader_constant(self) -> None:
        # The number of zone IDs the resolver knows about must
        # match the loader's NUM_TAXI_ZONES constant. This is the
        # invariant that justifies our [1, NUM_TAXI_ZONES] filter
        # in clean_trips.
        self.assertEqual(len(self.resolver.known_ids()), NUM_TAXI_ZONES)

    def test_special_zones_264_and_265(self) -> None:
        # Zone 264 and 265 are catch-alls in the official table.
        # We pin their meanings down so any future schema change
        # at NYC TLC produces a clear test failure.
        self.assertEqual(self.resolver.borough(264), "Unknown")
        self.assertEqual(self.resolver.borough(265), "N/A")
        self.assertEqual(self.resolver.zone(265), "Outside of NYC")

    def test_top_busiest_zones_from_real_data(self) -> None:
        # The two airports we identified as the top-2 busiest
        # pickup zones during our January 2026 profiling are zone
        # 132 (JFK) and zone 138 (LaGuardia). This test pairs the
        # README claim with a machine-checkable fact.
        self.assertEqual(self.resolver.zone(132), "JFK Airport")
        self.assertEqual(self.resolver.zone(138), "LaGuardia Airport")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
