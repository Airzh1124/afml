"""Trend-following primary models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average_cross_side(
    close: pd.Series,
    *,
    fast_window: int,
    slow_window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Return trend-following side from a fast/slow moving-average cross.

    The side is ``1`` when the fast moving average is above the slow moving
    average, ``-1`` when it is below, and missing until both moving averages are
    available. Ties inherit the most recent non-zero side.
    """
    close = _prepare_close(close)
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("moving-average windows must be positive")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")
    if min_periods is not None and min_periods <= 0:
        raise ValueError("min_periods must be positive")

    fast_min_periods = min_periods if min_periods is not None else fast_window
    slow_min_periods = min_periods if min_periods is not None else slow_window
    fast = close.rolling(fast_window, min_periods=fast_min_periods).mean()
    slow = close.rolling(slow_window, min_periods=slow_min_periods).mean()

    raw_side = pd.Series(np.sign(fast - slow), index=close.index, name="side")
    side = raw_side.replace(0, np.nan).ffill()
    side = side.where(fast.notna() & slow.notna())
    return side.dropna().astype(float)


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
