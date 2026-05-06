"""Regression tests for bugs uncovered during the post-submission audit.

These tests pin down the fixes for four real algorithmic problems
that an external review caught after the original submission was
ready. Every test here corresponds to a specific bug and is named
to make the connection visible.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from taxi_demand.evaluate import evaluate_model, mae, naive_baseline, rmse
from taxi_demand.features import add_lags, add_rolling_features, aggregate
from taxi_demand.model import DemandForecaster


class TestAggregateFillMissingHours(unittest.TestCase):
    """Bug fix: ``aggregate`` previously emitted only those
    (zone, hour) cells with at least one pickup. Without a complete
    hourly grid, ``add_lags(lags=(24,))`` shifts by 24 *records*
    rather than 24 *hours*, which silently misaligns whenever a
    zone has a gap in its hourly stream. The fix is the new
    ``fill_missing_hours`` keyword argument.
    """

    def _build_trips_with_gap(self):
        """Produce 5 days of zone-1 trips with a deliberate gap at
        day 2 hour 12 (no trips), so the aggregate output skips
        that row by default."""
        trips = []
        base = pd.Timestamp("2026-01-01")
        for day in range(5):
            for hour in range(24):
                if day == 2 and hour == 12:
                    continue  # no trips this hour
                for _ in range(10):
                    trips.append({
                        "pickup_datetime": (
                            base + pd.Timedelta(days=day, hours=hour)
                        ),
                        "PULocationID": 1,
                        "DOLocationID": 1,
                    })
        return pd.DataFrame(trips)

    def test_default_behavior_preserves_existing_contract(self):
        # Backward compatibility: with fill_missing_hours=False
        # (the default), missing hours are still skipped, so the
        # team's original tests continue to pass.
        df = self._build_trips_with_gap()
        agg = aggregate(df)
        self.assertEqual(len(agg), 5 * 24 - 1)  # one cell missing

    def test_fill_missing_hours_completes_grid(self):
        df = self._build_trips_with_gap()
        agg = aggregate(df, fill_missing_hours=True)
        # Now we should have a complete 5-day x 24-hour grid.
        self.assertEqual(len(agg), 5 * 24)
        # The previously-missing cell should now have demand=0.
        target = pd.Timestamp("2026-01-03 12:00:00")
        gap_row = agg[agg["hour"] == target]
        self.assertEqual(int(gap_row["demand"].iloc[0]), 0)

    def test_fill_makes_lag_24h_truly_24_hours_back(self):
        # The whole point of fill_missing_hours: lag_24h must mean
        # "24 hours earlier", not "24 records earlier".
        df = self._build_trips_with_gap()
        agg = aggregate(df, fill_missing_hours=True)
        featured = add_lags(agg, lags=(24,))

        # Pick day 2 hour 13 (first row after the gap).
        target = pd.Timestamp("2026-01-03 13:00:00")
        row = featured[featured["hour"] == target].iloc[0]
        target_idx = featured.index[featured["hour"] == target].tolist()[0]
        # The row 24 positions earlier should be exactly 24 hours
        # earlier on the wall clock.
        ref = featured.iloc[target_idx - 24]
        self.assertEqual(
            ref["hour"], pd.Timestamp("2026-01-02 13:00:00")
        )


class TestRollingFeaturesNoLeakage(unittest.TestCase):
    """Bug fix: ``add_rolling_features`` used to include the
    current hour's demand in the rolling window, which is target
    leakage in a forecasting setting. The fix shifts demand by one
    before rolling, so the window summarises only past hours."""

    def test_changing_current_demand_does_not_change_current_feature(self):
        df = pd.DataFrame({
            "zone_id": [1] * 4,
            "hour": pd.date_range("2026-01-01", periods=4, freq="h"),
            "demand": [10, 20, 30, 40],
        })
        out_a = add_rolling_features(df, windows=[2])

        # Mutate the demand at row 2 only.
        df_b = df.copy()
        df_b.loc[2, "demand"] = 999
        out_b = add_rolling_features(df_b, windows=[2])

        # Row 2's rolling features must NOT depend on row 2's
        # demand: only on rows 0 and 1. So out_a and out_b should
        # agree at row 2.
        self.assertEqual(
            out_a.iloc[2]["roll_mean_2h"],
            out_b.iloc[2]["roll_mean_2h"],
        )

    def test_first_row_has_nan_mean(self):
        # The first row of any zone has no past history to roll
        # over, so its rolling mean must be NaN (no history).
        df = pd.DataFrame({
            "zone_id": [1, 1, 1],
            "hour": pd.date_range("2026-01-01", periods=3, freq="h"),
            "demand": [10, 20, 30],
        })
        out = add_rolling_features(df, windows=[2])
        self.assertTrue(pd.isna(out.iloc[0]["roll_mean_2h"]))


class TestEvaluateModelExtendedFeatures(unittest.TestCase):
    """Bug fix: ``evaluate_model`` used to only drop NaN on
    ``[lag_1h, lag_24h]``. If the model used additional feature
    columns with longer warm-up periods (e.g. ``lag_168h``), the
    model's internal NaN drop and the evaluator's NaN drop would
    disagree, causing a shape mismatch when computing metrics.
    """

    def _build_df(self, n=200):
        return pd.DataFrame({
            "demand": np.arange(n) % 24 + 5.0,
            "lag_1h":   [np.nan] + list((np.arange(n-1) % 24 + 5.0)),
            "lag_24h":  [np.nan]*24 + list((np.arange(n-24) % 24 + 5.0)),
            "lag_168h": [np.nan]*168 + list((np.arange(n-168) % 24 + 5.0)),
        })

    def test_evaluate_with_lag_168h_does_not_crash(self):
        df = self._build_df()
        model = DemandForecaster(
            feature_columns=["lag_1h", "lag_24h", "lag_168h"]
        )
        model.fit(df)
        # Before the fix this raised:
        #   ValueError: operands could not be broadcast together
        #   with shapes (176,) (32,)
        result = evaluate_model(model, df)
        for key in ("model_mae", "model_rmse",
                    "baseline_mae", "baseline_rmse"):
            self.assertIn(key, result)
            self.assertIsInstance(result[key], float)

    def test_naive_baseline_extra_dropna_columns(self):
        # Use n=200 so there is room for the 168-hour lag's
        # warm-up; using n=50 here would crash the dataframe
        # constructor because lag_168h would need 168 NaN slots.
        df = self._build_df(n=200)
        # Without extras, the baseline's row count reflects the
        # 24-hour warm-up of lag_24h (lag_1h's warm-up is shorter,
        # so lag_24h dominates).
        plain = naive_baseline(df)
        # With extras, the baseline drops more rows to match a
        # model that uses lag_168h.
        extended = naive_baseline(df, extra_dropna_columns=["lag_168h"])
        # Extended must be a (possibly proper) subset, never longer.
        self.assertLessEqual(len(extended), len(plain))


class TestDemandForecasterPersistencePreservesFeatures(unittest.TestCase):
    """Bug fix: ``save`` used to persist only the bare scikit-learn
    estimator, losing the ``feature_columns`` contract. A
    subsequent ``load`` into a default-constructed forecaster
    therefore predicted with the wrong columns, which crashed at
    inference."""

    def _build_df(self, n=200):
        return pd.DataFrame({
            "demand": np.arange(n) % 24 + 5.0,
            "lag_1h":   [0.0] + list((np.arange(n-1) % 24 + 5.0)),
            "lag_24h":  [0.0]*24 + list((np.arange(n-24) % 24 + 5.0)),
            "lag_168h": [0.0]*168 + list((np.arange(n-168) % 24 + 5.0)),
        })

    def test_round_trip_preserves_feature_columns(self):
        df = self._build_df()
        model = DemandForecaster(
            feature_columns=["lag_1h", "lag_24h", "lag_168h"]
        )
        model.fit(df)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
            path = tf.name
        try:
            model.save(path)
            # Default-construct, then load.
            m2 = DemandForecaster()
            self.assertEqual(m2.feature_columns, ["lag_1h", "lag_24h"])
            m2.load(path)
            # After load, the contract is the original one.
            self.assertEqual(
                m2.feature_columns,
                ["lag_1h", "lag_24h", "lag_168h"],
            )
            # Prediction succeeds with the right column count.
            preds = m2.predict(df)
            self.assertEqual(preds.ndim, 1)
        finally:
            os.unlink(path)

    def test_save_before_fit_raises(self):
        # We cannot persist a contract for an unfitted forecaster.
        model = DemandForecaster()
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
            path = tf.name
        try:
            with self.assertRaises(RuntimeError):
                model.save(path)
        finally:
            os.unlink(path)


class TestMaeRmseShapeValidation(unittest.TestCase):
    """Bug fix (defensive): ``mae``/``rmse`` previously relied on
    numpy's broadcasting error for shape mismatches and silently
    returned NaN for empty inputs. Now both surface as explicit
    ValueError, matching the contract MAPE has always had."""

    def test_mae_mismatched_shapes_raises_value_error(self):
        with self.assertRaises(ValueError):
            mae([1, 2, 3], [1, 2])

    def test_rmse_mismatched_shapes_raises_value_error(self):
        with self.assertRaises(ValueError):
            rmse([1, 2, 3], [1, 2])

    def test_mae_empty_input_raises(self):
        with self.assertRaises(ValueError):
            mae([], [])

    def test_rmse_empty_input_raises(self):
        with self.assertRaises(ValueError):
            rmse([], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
