"""Evaluation utilities for hourly NYC taxi-zone demand forecasts.

This module preserves the team's original four-function API
(:func:`mae`, :func:`rmse`, :func:`naive_baseline`,
:func:`evaluate_model`) and adds two further helpers:

* :func:`mape` -- mean absolute percentage error, useful for
  demand-volume forecasting where relative error is more interpretable
  than absolute error.
* :func:`time_series_cv_split` -- a time-respecting train/test
  splitter that yields ``(train_idx, test_idx)`` pairs in which the
  training prefix is *strictly* earlier than the test block, so we
  never leak information from the future into the past.

The :func:`evaluate_model` return dictionary still contains exactly
the four keys the team's tests assert against; we now optionally
include ``model_mape`` and ``baseline_mape`` only when ``include_mape``
is set, so adding the new metric does not change any existing test
expectations.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

from taxi_demand.model import DemandForecaster


def mae(y_true, y_pred) -> float:
    """Mean absolute error between ``y_true`` and ``y_pred``.

    Accepts any array-like inputs (numpy arrays, lists, pandas
    Series). Validates that the inputs have matching shapes and
    are non-empty so that a length mismatch surfaces as a clear
    ``ValueError`` rather than as numpy's much vaguer
    broadcasting error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Inputs must be non-empty.")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error between ``y_true`` and ``y_pred``.

    Accepts any array-like inputs; validates shape and emptiness
    in the same way as :func:`mae`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("Inputs must be non-empty.")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    epsilon: float = 1e-9,
) -> float:
    """Mean absolute percentage error, expressed as a fraction.

    Demand counts can include zeros (an empty zone-hour), which would
    blow up the standard MAPE formula via division by zero. We add a
    small ``epsilon`` to the denominator to keep the metric finite.
    A return value of ``0.12`` means the average forecast was off by
    12% of the observed value.

    Parameters
    ----------
    y_true, y_pred : array-like
        1-D arrays of the same length.
    epsilon : float, keyword-only
        Small constant added to ``|y_true|`` to avoid division by zero.

    Returns
    -------
    float
        Mean absolute percentage error as a fraction.

    Raises
    ------
    ValueError
        If the inputs have mismatched shapes or if ``epsilon`` is
        negative.
    """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs "
            f"y_pred {y_pred_arr.shape}."
        )
    if y_true_arr.size == 0:
        raise ValueError("Inputs must be non-empty.")
    denom = np.abs(y_true_arr) + epsilon
    return float(np.mean(np.abs(y_true_arr - y_pred_arr) / denom))


def naive_baseline(
    df: pd.DataFrame,
    *,
    extra_dropna_columns=None,
) -> np.ndarray:
    """Return ``lag_24h`` column as a naive forecast.

    Drops rows with NaN in ``lag_1h``, ``lag_24h``, and any
    additional columns the caller specifies via
    ``extra_dropna_columns``. The extra-columns argument lets
    :func:`evaluate_model` keep this function's output aligned with
    what a model with extended features (e.g. ``lag_168h``) would
    drop, so the baseline and the model see the same row set.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame including ``lag_1h`` and ``lag_24h``.
    extra_dropna_columns : iterable of str, optional
        Additional columns to include in the NaN drop.
    """
    drop_cols = ["lag_1h", "lag_24h"]
    if extra_dropna_columns:
        drop_cols = list(dict.fromkeys(drop_cols + list(extra_dropna_columns)))
    eval_df = df.dropna(subset=drop_cols)
    return eval_df["lag_24h"].to_numpy().ravel()


def evaluate_model(
    model: DemandForecaster,
    df: pd.DataFrame,
    *,
    include_mape: bool = False,
) -> dict:
    """Evaluate a fitted model and the naive baseline on ``df``.

    Drops rows with NaN in any column the model uses (i.e.
    ``model.feature_columns`` plus ``lag_24h`` for the baseline)
    before computing metrics, so that the model's predictions, the
    baseline's predictions, and the ground-truth ``y_true`` all have
    matching lengths. This is critical for models that use extended
    feature sets like ``lag_168h`` or ``roll_mean_24h``: without it,
    the model's internal NaN drop and the evaluator's NaN drop
    would disagree, causing a shape mismatch downstream.

    The returned dictionary always contains exactly four keys
    (``model_mae``, ``model_rmse``, ``baseline_mae``,
    ``baseline_rmse``) so the team's existing tests remain valid.
    Setting ``include_mape=True`` augments the dictionary with
    ``model_mape`` and ``baseline_mape``.

    Parameters
    ----------
    model : DemandForecaster
        A fitted forecaster.
    df : pd.DataFrame
        Evaluation set with at least ``demand``, ``lag_24h``, and
        every column listed in ``model.feature_columns``.
    include_mape : bool, keyword-only
        Whether to include MAPE in the returned dictionary.

    Returns
    -------
    dict
        Keys are the metric names, values are floats.
    """
    required = list(dict.fromkeys(
        list(model.feature_columns) + ["lag_24h", "demand"]
    ))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame missing required columns: {missing}"
        )
    eval_df = df.dropna(subset=required)
    if len(eval_df) == 0:
        raise ValueError(
            "Evaluation set is empty after dropping NaN rows on "
            f"required columns {required}. Check that the warm-up "
            "rows have not removed all data."
        )
    y_true = eval_df["demand"].to_numpy()
    model_pred = model.predict(eval_df)
    baseline_pred = eval_df["lag_24h"].to_numpy().ravel()

    result = {
        "model_mae": mae(y_true, model_pred),
        "model_rmse": rmse(y_true, model_pred),
        "baseline_mae": mae(y_true, baseline_pred),
        "baseline_rmse": rmse(y_true, baseline_pred),
    }
    if include_mape:
        result["model_mape"] = mape(y_true, model_pred)
        result["baseline_mape"] = mape(y_true, baseline_pred)
    return result


def time_series_cv_split(
    n_samples: int,
    n_splits: int = 5,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` pairs that respect time order.

    Time-series cross-validation must never use future observations
    to predict past ones. At fold ``i`` the training set is a strict
    prefix of the time-ordered series and the test block is a fixed-
    size window immediately after it. The splitter mirrors
    ``sklearn.model_selection.TimeSeriesSplit`` semantics but is
    implemented from scratch here so the package has no dependency
    on a private sklearn API.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the time-ordered dataset.
    n_splits : int
        Number of folds to generate. Must be at least 2.

    Yields
    ------
    tuple of numpy.ndarray
        ``(train_idx, test_idx)`` arrays of integer positions.

    Raises
    ------
    ValueError
        On non-positive ``n_samples``, ``n_splits < 2``, or
        ``n_samples < n_splits + 1``.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2.")
    if n_samples < n_splits + 1:
        raise ValueError(
            f"Need at least n_splits+1={n_splits + 1} samples, got "
            f"{n_samples}."
        )

    test_size = max(1, n_samples // (n_splits + 1))
    indices = np.arange(n_samples)
    for fold in range(n_splits):
        train_end = n_samples - (n_splits - fold) * test_size
        test_end = train_end + test_size
        train_idx = indices[:train_end]
        test_idx = indices[train_end:test_end]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx
