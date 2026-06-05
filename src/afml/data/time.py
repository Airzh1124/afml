"""Time-index normalization for market data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _to_datetime(values: Any) -> pd.DatetimeIndex:
    """Parse market timestamps, accepting ISO strings with mixed precision."""
    try:
        parsed = pd.to_datetime(values, format="ISO8601")
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(values)
        except ValueError:
            parsed = pd.to_datetime(values, format="mixed")
    return pd.DatetimeIndex(parsed)


def normalize_time_index(
    values: Any,
    *,
    name: str = "timestamp",
    timezone: str | None = "UTC",
) -> pd.DatetimeIndex:
    """Return a normalized ``DatetimeIndex`` for market data.

    The project data-layer contract is UTC timezone-aware timestamps by default.
    Naive timestamps are interpreted as belonging to ``timezone``; timezone-aware
    timestamps are converted to ``timezone``.
    """
    if timezone is None:
        index = pd.DatetimeIndex(_to_datetime(values), name=name)
    else:
        index = pd.DatetimeIndex(_to_datetime(values), name=name)
        index = (
            index.tz_localize(timezone)
            if index.tz is None
            else index.tz_convert(timezone)
        )
    return index
