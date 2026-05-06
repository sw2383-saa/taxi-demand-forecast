"""Model utilities for hourly NYC taxi-zone pickup demand forecasting.

This module exposes two complementary APIs:

* :class:`DemandForecaster` -- the original simple linear-regression
  forecaster. Its API (``fit`` / ``predict`` / ``save`` / ``load``) is
  kept unchanged so the team's existing tests continue to pass.
* :func:`make_pipeline`, :func:`train_models`,
  :class:`MultiModelForecaster` -- the richer scikit-learn pipeline
  workflow that supports ``StandardScaler`` plus three different model
  families (linear regression, ridge regression, random forest), as
  agreed in the project plan. New code should prefer this API.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Names accepted by :func:`make_pipeline`. We keep the tuple at module
# scope so callers can introspect it (for example, when building a UI
# that lists the supported models) and so unit tests can assert that
# the constant has not silently shrunk.
SUPPORTED_MODELS: Tuple[str, ...] = ("linear", "ridge", "random_forest")

# Default fixed seed for the only model family that has stochastic
# behaviour (random forest). Centralising this makes every model run
# bit-reproducible by default, which keeps the test suite stable.
DEFAULT_RANDOM_STATE: int = 42

# Default feature columns the simple :class:`DemandForecaster` consumes.
# The team's tests use the two original lag columns and we keep them
# as the default to preserve backwards compatibility.
DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = ("lag_1h", "lag_24h")


class DemandForecaster:
    """Linear-regression forecaster using lagged hourly demand features.

    This class preserves the team's original API:

    * ``fit(df)``      -- train on a feature DataFrame.
    * ``predict(df)``  -- return predictions as a 1-D ``ndarray``.
    * ``save(path)``   -- persist the fitted estimator to disk.
    * ``load(path)``   -- restore a fitted estimator from disk.

    The defaults match the original code: features are
    ``[lag_1h, lag_24h]``, target is ``demand``, and the estimator is
    ``sklearn.linear_model.LinearRegression``.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
        target_column: str = "demand",
    ) -> None:
        # We store the feature/target column names so the same forecaster
        # instance can be retargeted (for example, to consume calendar
        # features as well) without changing its public methods.
        self.feature_columns: List[str] = list(feature_columns)
        self.target_column: str = target_column

    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        """Train on ``df`` and return ``self`` (sklearn-style chaining).

        Rows with NaN in any of the feature columns are dropped before
        fitting, mirroring the rule established in the team's original
        API: lag warm-up rows must not poison the fit.

        Raises
        ------
        ValueError
            If a required column is missing from ``df``, or if the
            training set is empty after the warm-up drop. The latter
            check (mirroring :class:`MultiModelForecaster`) replaces
            scikit-learn's less-informative "Found array with 0
            sample(s)" message with one that names the actual
            root cause.
        """
        self._check_columns(df, include_target=True)
        train_df = df.dropna(subset=self.feature_columns)
        if len(train_df) == 0:
            raise ValueError(
                "After dropping NaN rows on feature columns, the "
                "training set is empty. Check that lag warm-up rows "
                "have not removed all data."
            )
        X = train_df[self.feature_columns].to_numpy()
        y = train_df[self.target_column].to_numpy()
        self.model_ = LinearRegression()
        self.model_.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted demand as a 1-D numpy array.

        Rows with NaN in feature columns are dropped before prediction
        so the contract matches that of :func:`taxi_demand.evaluate.naive_baseline`.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted or loaded.
        ValueError
            If a required feature column is missing from ``df``.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError(
                "DemandForecaster must be fitted before prediction."
            )
        self._check_columns(df, include_target=False)
        pred_df = df.dropna(subset=self.feature_columns)
        X = pred_df[self.feature_columns].to_numpy()
        return np.asarray(self.model_.predict(X)).ravel()

    def save(self, path: str) -> None:
        """Persist the fitted estimator and its feature contract.

        We save a dict containing the underlying scikit-learn
        estimator together with the ``feature_columns`` and
        ``target_column`` that the forecaster was trained against.
        Without saving the feature contract, a subsequent ``load``
        call into a default-constructed DemandForecaster (which
        defaults to ``["lag_1h", "lag_24h"]``) would silently use
        the wrong feature set if the original training had used a
        custom one, which then crashes inside scikit-learn at
        prediction time with an opaque shape mismatch.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError(
                "DemandForecaster must be fitted before saving."
            )
        payload = {
            "model": self.model_,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
        }
        joblib.dump(payload, path)

    def load(self, path: str) -> None:
        """Restore a fitted estimator and its feature contract.

        Backwards-compatible with the older save format that
        persisted only the bare scikit-learn estimator: if the
        loaded artifact is not a dict, we assume it is the legacy
        format and only restore ``self.model_``, leaving the
        forecaster's ``feature_columns`` and ``target_column``
        attributes untouched (whatever the constructor set them to).
        """
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            self.model_ = loaded["model"]
            if "feature_columns" in loaded:
                self.feature_columns = list(loaded["feature_columns"])
            if "target_column" in loaded:
                self.target_column = loaded["target_column"]
        else:
            # Legacy format: just the estimator, no metadata.
            self.model_ = loaded

    def _check_columns(
        self, df: pd.DataFrame, *, include_target: bool
    ) -> None:
        """Raise if a feature (or target) column is absent.

        Catching missing columns up-front gives a much clearer error
        message than letting NumPy fail with a shape mismatch. The
        ``include_target`` flag lets us reuse this helper in both
        training (target required) and prediction (target optional).
        """
        required = list(self.feature_columns)
        if include_target:
            required.append(self.target_column)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")


def make_pipeline(
    model_name: str = "linear",
    *,
    scaler: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Build an *unfitted* scikit-learn ``Pipeline`` for demand forecasting.

    The pipeline structure is ``StandardScaler -> estimator`` for the
    two linear models and ``StandardScaler -> RandomForestRegressor``
    for the tree model. The scaler is harmless for the tree model but
    we keep the structure uniform so callers can swap models without
    rewiring downstream code.

    Parameters
    ----------
    model_name : str
        One of :data:`SUPPORTED_MODELS`.
    scaler : bool, keyword-only
        If False, omit the ``StandardScaler`` step.
    random_state : int, keyword-only
        Random seed for the random-forest estimator. Ignored for the
        deterministic linear models.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline.

    Raises
    ------
    ValueError
        If ``model_name`` is not a recognised family name.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"model_name must be one of {SUPPORTED_MODELS}, got {model_name!r}."
        )

    # Pick the estimator. The constants for each are deliberately small
    # (50 trees, depth 10) so the suite stays fast in CI; downstream
    # callers can replace the named step if they want to retune.
    if model_name == "linear":
        estimator = LinearRegression()
    elif model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=random_state)
    else:
        estimator = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=random_state,
            n_jobs=1,
        )

    steps: List[Tuple[str, object]] = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _check_xy(X: pd.DataFrame, y) -> Tuple[pd.DataFrame, np.ndarray]:
    """Validate model inputs and coerce them into numpy-friendly forms.

    We accept any DataFrame whose columns are numeric, and any
    array-like ``y``. This helper centralises the validation logic
    so :func:`train_models` does not have to repeat itself.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame.")
    if len(X) == 0:
        raise ValueError("X must be non-empty.")

    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.size == 0:
        raise ValueError("y must be non-empty.")
    if y_arr.shape[0] != X.shape[0]:
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} rows, y has "
            f"{y_arr.shape[0]} elements."
        )
    non_numeric = [
        c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
    ]
    if non_numeric:
        raise ValueError(
            f"X contains non-numeric columns: {non_numeric}. Encode "
            "them before fitting."
        )
    return X, y_arr


def train_models(
    X_train: pd.DataFrame,
    y_train,
    model_names: Optional[Sequence[str]] = None,
    *,
    scaler: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Pipeline]:
    """Fit one pipeline per requested model family.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : array-like
        Training target vector.
    model_names : sequence of str, optional
        Subset of :data:`SUPPORTED_MODELS` to train. Defaults to all
        three.
    scaler, random_state : forwarded to :func:`make_pipeline`.

    Returns
    -------
    dict of str to Pipeline
        Mapping from model name to fitted pipeline.

    Raises
    ------
    ValueError
        On invalid inputs (empty data, shape mismatch, unknown model
        name, or non-numeric feature columns).
    """
    X_clean, y_arr = _check_xy(X_train, y_train)

    if model_names is None:
        model_names = list(SUPPORTED_MODELS)
    if len(model_names) == 0:
        raise ValueError("model_names must be non-empty.")
    for name in model_names:
        if name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model name {name!r}. Allowed: {SUPPORTED_MODELS}."
            )

    fitted: Dict[str, Pipeline] = {}
    for name in model_names:
        pipe = make_pipeline(name, scaler=scaler, random_state=random_state)
        pipe.fit(X_clean, y_arr)
        fitted[name] = pipe
    return fitted


class MultiModelForecaster:
    """Train and serve all three model families behind a single object.

    This class is the recommended replacement for
    :class:`DemandForecaster` when the user wants the multi-model
    comparison the project plan calls for. The simpler class is kept
    for backwards compatibility with the team's original tests; new
    code should prefer this one.
    """

    def __init__(
        self,
        feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
        target_column: str = "demand",
        model_names: Optional[Sequence[str]] = None,
        *,
        scaler: bool = True,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.model_names = list(model_names) if model_names else list(SUPPORTED_MODELS)
        self.scaler = scaler
        self.random_state = random_state
        self.models_: Dict[str, Pipeline] = {}

    def _check_columns(
        self, df: pd.DataFrame, *, include_target: bool
    ) -> None:
        """Raise a clear ``ValueError`` if a required column is absent.

        Mirrors :meth:`DemandForecaster._check_columns` so both
        public model classes surface the same kind of message
        ("DataFrame missing required columns: [...]") rather than
        letting pandas raise its less-informative ``KeyError`` from
        deep inside the indexing path.
        """
        required = list(self.feature_columns)
        if include_target:
            required.append(self.target_column)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

    def fit(self, df: pd.DataFrame) -> "MultiModelForecaster":
        """Train every requested model on ``df``."""
        self._check_columns(df, include_target=True)
        train_df = df.dropna(subset=self.feature_columns)
        if len(train_df) == 0:
            raise ValueError(
                "After dropping NaN rows on feature columns, the "
                "training set is empty. Check that lag warm-up rows "
                "have not removed all data."
            )
        X = train_df[self.feature_columns]
        y = train_df[self.target_column].to_numpy()
        self.models_ = train_models(
            X,
            y,
            model_names=self.model_names,
            scaler=self.scaler,
            random_state=self.random_state,
        )
        return self

    def predict(
        self, df: pd.DataFrame, model_name: str = "linear"
    ) -> np.ndarray:
        """Return predictions from the requested model family."""
        if not self.models_:
            raise RuntimeError("MultiModelForecaster must be fitted first.")
        if model_name not in self.models_:
            raise ValueError(
                f"Model {model_name!r} not trained. Available: "
                f"{sorted(self.models_)}."
            )
        self._check_columns(df, include_target=False)
        pred_df = df.dropna(subset=self.feature_columns)
        X = pred_df[self.feature_columns]
        preds = self.models_[model_name].predict(X)
        return np.asarray(preds).ravel()
