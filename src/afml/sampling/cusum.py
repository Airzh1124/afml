"""Symmetric CUSUM filter."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


def cusum_filter(
    series: pd.Series,
    threshold: float | pd.Series,
    *,
    use_returns: bool = False,
    log_returns: bool = False,
    max_gap: str | pd.Timedelta | None = None,
) -> pd.DatetimeIndex:
    """Apply the symmetric CUSUM filter.

    Parameters
    ----------
    series
        Price-like series indexed by timestamps. By default the filter is
        applied to first differences. If ``use_returns`` is true, the filter is
        applied to simple returns or log returns.
    threshold
        Positive scalar threshold, or a positive time-indexed series aligned to
        ``series``. Dynamic thresholds are forward-filled onto the input index.
    use_returns
        Use returns instead of price differences.
    log_returns
        Use log returns. Requires ``use_returns=True``.
    max_gap
        Optional maximum allowed time gap between adjacent observations. If the
        gap is larger, cumulative sums are reset and the cross-gap increment is
        ignored.

    Returns
    -------
    pandas.DatetimeIndex
        Event timestamps where the positive or negative cumulative sum crosses
        the threshold.
    """
    values = _prepare_series(series)
    thresholds = _prepare_threshold(threshold, values.index)

    if log_returns and not use_returns:
        raise ValueError("log_returns=True requires use_returns=True")

    max_gap = pd.Timedelta(max_gap) if max_gap is not None else None
    increments = _increments(values, use_returns=use_returns, log_returns=log_returns)
    if increments.empty:
        return pd.DatetimeIndex([], name=values.index.name)

    events = []
    positive_sum = 0.0
    negative_sum = 0.0
    increment_index = increments.index
    increment_values = increments.to_numpy(dtype=float, copy=False)
    threshold_values = thresholds.reindex(increment_index).to_numpy(dtype=float, copy=False)
    gap_values = None
    if max_gap is not None:
        gap_values = (
            values.index.to_series()
            .diff()
            .reindex(increment_index)
            .to_numpy()
        )

    for position, timestamp in enumerate(increment_index):
        if gap_values is not None:
            if gap_values[position] > max_gap:
                positive_sum = 0.0
                negative_sum = 0.0
                continue

        threshold_at_time = threshold_values[position]
        increment = increment_values[position]
        positive_sum = max(0.0, positive_sum + increment)
        negative_sum = min(0.0, negative_sum + increment)

        if negative_sum < -threshold_at_time:
            negative_sum = 0.0
            events.append(timestamp)
        elif positive_sum > threshold_at_time:
            positive_sum = 0.0
            events.append(timestamp)

    return pd.DatetimeIndex(events, name=values.index.name)


def _prepare_series(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a pandas DatetimeIndex")

    values = series.copy().sort_index(kind="mergesort")
    if values.index.hasnans:
        raise ValueError("series index contains missing timestamps")
    if values.isna().any():
        raise ValueError("series contains missing values")

    return values.astype(float)


def _prepare_threshold(
    threshold: float | pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    if isinstance(threshold, Real):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        return pd.Series(float(threshold), index=index)

    if not isinstance(threshold, pd.Series):
        raise TypeError("threshold must be a positive scalar or pandas Series")
    if not isinstance(threshold.index, pd.DatetimeIndex):
        raise TypeError("threshold series must be indexed by a pandas DatetimeIndex")
    if threshold.empty:
        raise ValueError("threshold series must not be empty")
    if threshold.isna().any():
        raise ValueError("threshold series contains missing values")
    if (threshold <= 0).any():
        raise ValueError("threshold must be positive")

    aligned = threshold.astype(float).sort_index(kind="mergesort").reindex(index, method="ffill")
    if aligned.isna().any():
        raise ValueError("threshold series must cover the input series start")
    return aligned


def _increments(
    values: pd.Series,
    *,
    use_returns: bool,
    log_returns: bool,
) -> pd.Series:
    if use_returns and log_returns:
        increments = np.log(values).diff()
    elif use_returns:
        increments = values.pct_change()
    else:
        increments = values.diff()

    return increments.dropna()
