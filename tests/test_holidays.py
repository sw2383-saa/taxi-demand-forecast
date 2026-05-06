"""Tests that pin down the holiday-calendar design decisions.

These tests exist to make the team's *modelling choice* about
holidays explicit and defensible. The HVFHV dataset itself does not
classify holidays, so the choice of which dates to flag is a
modelling decision; the assertions below document that decision in
machine-checkable form.
"""

import datetime as dt
import unittest

import pandas as pd

from taxi_demand.features import (
    NY_STATE_HOLIDAYS_2019_2026,
    US_FEDERAL_HOLIDAYS_2019_2026,
    add_calendar_features,
    _DEFAULT_US_HOLIDAYS,
)


class TestHolidayConstants(unittest.TestCase):
    """Tests for the two bundled holiday calendars."""

    def test_federal_calendar_size_matches_opm_rules(self) -> None:
        # Count by year of observed date within 2019-2026:
        #   2019: 10  (no Juneteenth yet)
        #   2020: 10  (no Juneteenth yet)
        #   2021: 12  (Juneteenth added; Christmas Day fell on Saturday
        #              and was observed Friday Dec 24; New Year's Day
        #              2022 fell on Saturday and was observed Friday
        #              Dec 31, 2021 -- so 2021 carries an extra entry)
        #   2022: 10  (Jan 1, 2022 observance already counted in 2021)
        #   2023: 11
        #   2024: 11
        #   2025: 11
        #   2026: 11
        # Total = 10 + 10 + 12 + 10 + 11 + 11 + 11 + 11 = 86.
        self.assertEqual(len(US_FEDERAL_HOLIDAYS_2019_2026), 86)

    def test_ny_state_is_strict_superset_of_federal(self) -> None:
        # NY State observes everything federal observes, plus three
        # additional holidays (Lincoln's Birthday, Flag Day, and
        # Election Day). The NY set must therefore *contain* the
        # federal set with no exceptions.
        self.assertTrue(
            US_FEDERAL_HOLIDAYS_2019_2026 <= NY_STATE_HOLIDAYS_2019_2026
        )

    def test_ny_state_adds_exactly_three_holidays_per_year(self) -> None:
        # Lincoln's Birthday + Flag Day + Election Day = 3 per year
        # for 2019-2026 = 24 extra observances.
        extras = NY_STATE_HOLIDAYS_2019_2026 - US_FEDERAL_HOLIDAYS_2019_2026
        self.assertEqual(len(extras), 8 * 3)

    def test_legacy_alias_preserved(self) -> None:
        # The previous version of this module exposed the constant as
        # ``_DEFAULT_US_HOLIDAYS``. We keep that name as an alias so
        # any old code that imported it continues to work.
        self.assertIs(_DEFAULT_US_HOLIDAYS, US_FEDERAL_HOLIDAYS_2019_2026)


class TestJanuary2026Agreement(unittest.TestCase):
    """The default workflow uses January 2026; for that month the
    federal and NY State calendars agree on every observed day, so
    the holiday-calendar choice has no observable effect on the
    default analysis. This test pins that fact down so a future
    change to either calendar is caught immediately.
    """

    def _january_2026_days(self):
        return pd.date_range("2026-01-01", "2026-01-31", freq="D")

    def test_federal_and_ny_agree_in_january_2026(self) -> None:
        days = self._january_2026_days()
        federal_flags = days.normalize().isin(US_FEDERAL_HOLIDAYS_2019_2026)
        ny_flags = days.normalize().isin(NY_STATE_HOLIDAYS_2019_2026)
        # Every January day is classified the same way under both
        # calendars (i.e. there are zero disagreements).
        disagreements = (federal_flags != ny_flags).sum()
        self.assertEqual(int(disagreements), 0)

    def test_january_2026_holiday_count_is_two(self) -> None:
        # New Year's Day (Jan 1) and MLK Day (Jan 19, third Monday).
        days = self._january_2026_days()
        flags = days.normalize().isin(US_FEDERAL_HOLIDAYS_2019_2026)
        self.assertEqual(int(flags.sum()), 2)
        # Verify the two expected dates explicitly.
        marked = days[flags].normalize().tolist()
        self.assertIn(pd.Timestamp("2026-01-01"), marked)
        self.assertIn(pd.Timestamp("2026-01-19"), marked)


class TestCustomHolidaysOverride(unittest.TestCase):
    """The ``add_calendar_features`` function lets callers override
    the holiday set entirely. This is the escape hatch that lets the
    team substitute any custom calendar (e.g. a list pulled from a
    third-party API) without changing the code.
    """

    def test_passing_ny_set_marks_lincoln_birthday(self) -> None:
        df = pd.DataFrame({"hour": pd.to_datetime(["2026-02-12 12:00"])})
        # February 12 is Lincoln's Birthday: NY State observes it,
        # federal does not. With the federal default, is_holiday
        # should be 0; with the NY State set, it should be 1.
        federal_out = add_calendar_features(df)
        self.assertEqual(int(federal_out["is_holiday"].iloc[0]), 0)

        ny_out = add_calendar_features(df, holidays=NY_STATE_HOLIDAYS_2019_2026)
        self.assertEqual(int(ny_out["is_holiday"].iloc[0]), 1)

    def test_arbitrary_custom_calendar(self) -> None:
        # Caller supplies their own list (e.g. a school calendar).
        df = pd.DataFrame({"hour": pd.to_datetime(["2026-03-15 12:00"])})
        out = add_calendar_features(df, holidays=["2026-03-15"])
        self.assertEqual(int(out["is_holiday"].iloc[0]), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
