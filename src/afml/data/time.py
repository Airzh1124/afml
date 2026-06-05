"""Time-index normalization for market data."""

from __future__ import annotations

from typing import Any

import pandas as pd


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
        index = pd.DatetimeIndex(pd.to_datetime(values), name=name)
    else:
        index = pd.DatetimeIndex(pd.to_datetime(values), name=name)
        index = (
            index.tz_localize(timezone)
            if index.tz is None
            else index.tz_convert(timezone)
        )
    return index
