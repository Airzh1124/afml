"""Dynamic volatility targets and thresholds."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def daily_volatility(close: pd.Series, *, span: int = 100) -> pd.Series:
    """Estimate daily volatility using AFML's one-day return method.

    For each timestamp, this finds the most recent observation at least one
    calendar day earlier, computes the one-day return, and applies an EWMA
    standard deviation.
    """
    close = _prepare_close(close)
    if span <= 0:
        raise ValueError("span must be positive")

    previous_positions = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    valid = previous_positions > 0
    if not valid.any():
        return pd.Series(dtype=float, index=close.index[:0], name="daily_volatility")

    current_positions = np.flatnonzero(valid)
    previous_positions = previous_positions[valid] - 1

    daily_returns = pd.Series(
        close.iloc[current_positions].to_numpy() / close.iloc[previous_positions].to_numpy() - 1,
        index=close.index[current_positions],
    )
    volatility = daily_returns.ewm(span=span).std()
    volatility.name = "daily_volatility"
    return volatility


def target_from_events(
    target: pd.Series,
    events: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    *,
    min_ret: float = 0.0,
) -> pd.Series:
    """Align a dynamic target series to event timestamps.

    ``min_ret`` filters out events whose target is too small to be useful for
    horizontal barriers.
    """
    target = _prepare_target(target)
    if min_ret < 0:
        raise ValueError("min_ret must be non-negative")

    event_index = pd.DatetimeIndex(events)
    aligned = target.reindex(event_index, method="ffill").dropna()
    aligned = aligned[aligned > min_ret]
    aligned.name = "trgt"
    return aligned


def _prepare_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must be indexed by a pandas DatetimeIndex")
    if close.empty:
        raise ValueError("close must not be empty")
    if close.isna().any():
        raise ValueError("close contains missing values")
    if (close <= 0).any():
        raise ValueError("close prices must be positive")
    return close.astype(float).sort_index(kind="mergesort")


def _prepare_target(target: pd.Series) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must be indexed by a pandas DatetimeIndex")
    if target.empty:
        raise ValueError("target must not be empty")
    if (target <= 0).dropna().any():
        raise ValueError("target values must be positive")
    return target.astype(float).sort_index(kind="mergesort")
