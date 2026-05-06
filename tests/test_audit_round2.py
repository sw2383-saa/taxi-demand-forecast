"""Regression tests for the second-round audit fixes.

This file is a sibling of ``test_audit_fixes.py`` and follows the
same convention: each test is named after the bug or behavior it
pins down so the connection between code change and test is
visible.

The bugs covered here came from a second external review of the
package after the first round of audit fixes had been applied.
Each one was independently reproduced (with code, not just
inspection) before the fix was written.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import requests

from taxi_demand.features import aggregate, build_features
from taxi_demand.loader import download
from taxi_demand.model import MultiModelForecaster


class TestBuildFeaturesFillsHourGrid(unittest.TestCase):
    """Bug fix: ``build_features`` must forward ``fill_missing_hours``
    to ``aggregate`` so the recommended forecasting workflow can
    actually use the complete hourly grid. Without the wiring,
    enabling the feature on the underlying ``aggregate`` does not
    help users who go through the higher-level ``build_features``
    entry point."""

    def _trips_with_gap(self):
        """Build trips where one hour has zero pickups."""
        rows = []
        base = pd.Timestamp("2026-01-01")
        for hour in range(48):  # 2 days
            if hour == 12:
                continue  # zero-pickup hour
            for _ in range(3):
                rows.append({
                    "pickup_datetime": base + pd.Timedelta(hours=hour),
                    "PULocationID": 1,
                    "DOLocationID": 1,
                })
        return pd.DataFrame(rows)

    def test_default_does_not_fill(self):
        df = self._trips_with_gap()
        out = build_features(df)
        # Default behaviour preserved: the gap hour is missing.
        hours = out["hour"].nunique()
        self.assertEqual(hours, 47)

    def test_fill_missing_hours_true_fills(self):
        df = self._trips_with_gap()
        out = build_features(df, fill_missing_hours=True)
        # The grid is now complete: 48 hours, no gaps.
        hours = out["hour"].nunique()
        self.assertEqual(hours, 48)
        # The previously-missing hour now has demand=0 in the
        # raw aggregated column.
        gap_row = out[out["hour"] == pd.Timestamp("2026-01-01 12:00:00")]
        self.assertEqual(int(gap_row["demand"].iloc[0]), 0)


class TestPanelDataUniqueHourSplit(unittest.TestCase):
    """Bug fix: the example pipeline's split helper used to cut on
    raw row position, which on panel data (multiple zones sharing
    the same timestamps) produces a hour-overlap between train
    and test.

    The fix is to import the helper from the example script
    module and verify it returns disjoint hour sets.
    """

    def _import_helper(self):
        # Import the example script as a module. We do this
        # locally rather than at module top because importlib
        # paths depend on the test runner's working directory.
        import importlib.util
        from pathlib import Path
        path = (
            Path(__file__).resolve().parent.parent
            / "examples"
            / "run_pipeline.py"
        )
        spec = importlib.util.spec_from_file_location("rp_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._train_test_split_time_ordered

    def test_no_hour_appears_in_both_train_and_test(self):
        split_fn = self._import_helper()
        # 2 zones x 7 hours = 14 rows. Naive iloc[:11] would
        # split inside hour 5 because 11 = 7+4.
        df = pd.DataFrame({
            "zone_id": [1] * 7 + [2] * 7,
            "hour": list(pd.date_range("2026-01-01", periods=7, freq="h")) * 2,
            "demand": list(range(14)),
        })
        train, test = split_fn(df, train_frac=0.8)
        train_hours = set(train["hour"])
        test_hours = set(test["hour"])
        # The defining property: no hour is in both folds.
        self.assertEqual(len(train_hours & test_hours), 0)
        # And every input hour appears in at least one fold.
        self.assertEqual(
            train_hours | test_hours,
            set(df["hour"]),
        )

    def test_train_fold_strictly_precedes_test_fold(self):
        split_fn = self._import_helper()
        df = pd.DataFrame({
            "zone_id": [1] * 10 + [2] * 10,
            "hour": list(pd.date_range("2026-01-01", periods=10, freq="h")) * 2,
            "demand": list(range(20)),
        })
        train, test = split_fn(df, train_frac=0.7)
        # Time-respecting: max train hour < min test hour.
        self.assertLess(train["hour"].max(), test["hour"].min())


class TestMultiModelForecasterColumnCheck(unittest.TestCase):
    """Bug fix: MultiModelForecaster used to surface a pandas
    KeyError for missing columns. We want a clear ValueError
    with a self-explanatory message, matching the contract
    DemandForecaster has always had."""

    def test_missing_target_raises_value_error(self):
        # No "demand" column at all
        df = pd.DataFrame({
            "lag_1h": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lag_24h": [0.0, 0.5, 1.0, 1.5, 2.0],
        })
        m = MultiModelForecaster()
        with self.assertRaises(ValueError) as cm:
            m.fit(df)
        self.assertIn("missing required columns", str(cm.exception))
        self.assertIn("demand", str(cm.exception))

    def test_missing_feature_raises_value_error(self):
        # lag_24h missing, default features include it
        df = pd.DataFrame({
            "demand": [1.0, 2.0, 3.0, 4.0, 5.0],
            "lag_1h": [0.0, 1.0, 2.0, 3.0, 4.0],
        })
        m = MultiModelForecaster()
        with self.assertRaises(ValueError) as cm:
            m.fit(df)
        self.assertIn("missing required columns", str(cm.exception))
        self.assertIn("lag_24h", str(cm.exception))

    def test_predict_with_missing_feature_raises(self):
        # Train on full data, then try to predict on a frame
        # missing one of the feature columns.
        df = pd.DataFrame({
            "demand": np.arange(50, dtype=float),
            "lag_1h": np.arange(50, dtype=float),
            "lag_24h": np.arange(50, dtype=float),
        })
        m = MultiModelForecaster()
        m.fit(df)
        with self.assertRaises(ValueError):
            m.predict(df[["lag_1h"]], model_name="linear")

    def test_empty_after_dropna_raises(self):
        # Every row has at least one NaN feature -> dropna empties
        # the frame -> we want a clear error, not a silent crash
        # inside scikit-learn.
        df = pd.DataFrame({
            "demand": [1.0, 2.0, 3.0],
            "lag_1h": [np.nan, np.nan, np.nan],
            "lag_24h": [1.0, 2.0, 3.0],
        })
        m = MultiModelForecaster()
        with self.assertRaises(ValueError) as cm:
            m.fit(df)
        self.assertIn("empty", str(cm.exception).lower())


class TestDownloadJanuary2019Rejection(unittest.TestCase):
    """Bug fix: download() used to accept (year=2019, month=1) and
    only fail at the network layer with a 404. Since HVFHV trip
    records start in February 2019 (Local Law 149 of 2018 took
    effect Feb 1, 2019), we now reject this case at the input
    validation layer with a clear message."""

    def test_january_2019_rejected_before_network(self):
        with self.assertRaises(ValueError) as cm:
            download(2019, 1, data_dir="/tmp/never-used")
        self.assertIn("February 2019", str(cm.exception))

    def test_february_2019_passes_validation(self):
        # February 2019 is the earliest valid HVFHV month. We
        # patch ``requests.get`` so the test does not touch the
        # network -- we only care that the validation layer at
        # the top of ``download`` does not reject these inputs,
        # not that the real CDN responds. Mocking keeps the test
        # deterministic and fast in CI environments without
        # outbound network access.
        with patch("taxi_demand.loader.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError(
                "mocked network failure"
            )
            try:
                download(
                    2019, 2, data_dir="/tmp/never-used",
                    max_retries=1, backoff_seconds=0, timeout=1,
                )
            except ValueError as exc:
                # The whole point: validation must NOT reject Feb 2019.
                self.fail(f"February 2019 wrongly rejected: {exc}")
            except (requests.ConnectionError, requests.HTTPError):
                # Network-layer failure is the expected outcome
                # of getting past validation. Anything in this
                # branch confirms the test's positive assertion.
                pass


class TestDemandForecasterEmptyAfterDropna(unittest.TestCase):
    """Bug fix: DemandForecaster.fit used to let scikit-learn
    surface its own "Found array with 0 sample(s)" error when
    every training row was dropped due to NaN features. The fix
    raises a clear ValueError that names the actual cause."""

    def test_empty_after_dropna_raises_clear_value_error(self):
        from taxi_demand.model import DemandForecaster
        df = pd.DataFrame({
            "demand": [1.0, 2.0, 3.0],
            "lag_1h": [np.nan, np.nan, np.nan],
            "lag_24h": [1.0, 2.0, 3.0],
        })
        m = DemandForecaster()
        with self.assertRaises(ValueError) as cm:
            m.fit(df)
        msg = str(cm.exception).lower()
        # The new message names the actual root cause -- "training
        # set is empty" -- rather than referring to LinearRegression.
        self.assertIn("empty", msg)
        self.assertIn("training set", msg)


class TestEvaluateModelExplicitErrors(unittest.TestCase):
    """Bug fix: evaluate_model used to surface pandas KeyError
    for missing columns and scikit-learn's downstream error for
    empty eval sets. Now both surface as self-describing
    ValueError, matching the contract of the rest of the package."""

    def _trained_model(self):
        from taxi_demand.model import DemandForecaster
        df = pd.DataFrame({
            "demand": np.arange(50, dtype=float),
            "lag_1h": np.arange(50, dtype=float),
            "lag_24h": np.arange(50, dtype=float),
        })
        m = DemandForecaster()
        m.fit(df)
        return m, df

    def test_missing_column_raises_value_error(self):
        from taxi_demand.evaluate import evaluate_model
        m, df = self._trained_model()
        eval_df = df.drop(columns=["lag_24h"])
        with self.assertRaises(ValueError) as cm:
            evaluate_model(m, eval_df)
        self.assertIn("missing required columns", str(cm.exception))
        self.assertIn("lag_24h", str(cm.exception))

    def test_empty_eval_set_raises_value_error(self):
        from taxi_demand.evaluate import evaluate_model
        m, _ = self._trained_model()
        # Every row has NaN in lag_1h -> dropna empties the frame.
        eval_df = pd.DataFrame({
            "demand": [1.0, 2.0, 3.0],
            "lag_1h": [np.nan, np.nan, np.nan],
            "lag_24h": [1.0, 2.0, 3.0],
        })
        with self.assertRaises(ValueError) as cm:
            evaluate_model(m, eval_df)
        self.assertIn("empty", str(cm.exception).lower())


class TestPlotDemandHeatmapValidation(unittest.TestCase):
    """Bug fix: plot_demand_heatmap used to surface deep pandas
    AttributeError on missing columns or non-datetime hour. Now
    both raise a self-describing ValueError up front."""

    def test_missing_columns_raises(self):
        from taxi_demand.visualize import plot_demand_heatmap
        # Missing 'demand' column
        df = pd.DataFrame({
            "zone_id": [1, 2],
            "hour": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        })
        with self.assertRaises(ValueError) as cm:
            plot_demand_heatmap(df)
        self.assertIn("Missing required columns", str(cm.exception))
        self.assertIn("demand", str(cm.exception))

    def test_non_datetime_hour_raises(self):
        from taxi_demand.visualize import plot_demand_heatmap
        df = pd.DataFrame({
            "zone_id": [1, 2],
            "hour": ["2026-01-01", "2026-01-02"],  # strings, not datetime
            "demand": [10, 20],
        })
        with self.assertRaises(ValueError) as cm:
            plot_demand_heatmap(df)
        self.assertIn("datetime", str(cm.exception).lower())


class TestPlotForecastValidation(unittest.TestCase):
    """Style-consistency fix: plot_forecast used to surface a
    pandas KeyError on missing columns. Now it raises the same
    self-describing ValueError that plot_demand_heatmap has, so
    the two plotting functions present a unified contract.
    """

    def test_missing_zone_id_raises(self):
        from taxi_demand.visualize import plot_forecast
        df = pd.DataFrame({
            "hour": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "demand": [10, 20],
        })
        with self.assertRaises(ValueError) as cm:
            plot_forecast(df, zone_id=1, y_pred=np.array([10, 20]))
        self.assertIn("Missing required columns", str(cm.exception))
        self.assertIn("zone_id", str(cm.exception))

    def test_missing_demand_raises(self):
        from taxi_demand.visualize import plot_forecast
        df = pd.DataFrame({
            "zone_id": [1, 1],
            "hour": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        })
        with self.assertRaises(ValueError) as cm:
            plot_forecast(df, zone_id=1, y_pred=np.array([10, 20]))
        self.assertIn("Missing required columns", str(cm.exception))
        self.assertIn("demand", str(cm.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
