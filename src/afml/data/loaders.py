"""Market data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from afml.data.schemas import DEFAULT_TICK_SCHEMA, TickSchema
from afml.data.validation import validate_tick_data


def normalize_tick_data(
    data: pd.DataFrame,
    *,
    schema: TickSchema = DEFAULT_TICK_SCHEMA,
    timestamp_col: str | None = None,
    date_col: str | None = None,
    time_col: str | None = None,
    sort_index: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Normalize raw tick-like data to the project's internal tick contract.

    The internal contract is:

    - ``DatetimeIndex`` represents market time.
    - Required columns are ``price`` and ``volume``.
    - Optional metadata columns are preserved.

    ``timestamp_col`` defaults to ``schema.timestamp`` when that column exists.
    If separate date and time columns are available, they are combined into the
    index. If no timestamp columns are available, ``data`` must already have a
    ``DatetimeIndex``.
    """
    frame = data.copy()
    timestamp_col = timestamp_col or schema.timestamp
    date_col = date_col or schema.date
    time_col = time_col or schema.time

    if timestamp_col in frame.columns:
        timestamps = pd.to_datetime(frame.pop(timestamp_col))
        frame.index = pd.DatetimeIndex(timestamps, name=schema.timestamp)
    elif date_col in frame.columns and time_col in frame.columns:
        timestamps = pd.to_datetime(
            frame.pop(date_col).astype(str) + " " + frame.pop(time_col).astype(str)
        )
        frame.index = pd.DatetimeIndex(timestamps, name=schema.timestamp)
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(
            "tick data must have timestamp columns or a pandas DatetimeIndex"
        )
    elif frame.index.name is None:
        frame.index = frame.index.rename(schema.timestamp)

    if sort_index:
        frame = frame.sort_index(kind="mergesort")

    if (
        schema.dollar_value not in frame.columns
        and set(schema.required_columns) <= set(frame.columns)
    ):
        frame[schema.dollar_value] = frame[schema.price] * frame[schema.volume]

    if validate:
        validate_tick_data(frame, schema=schema, require_monotonic=sort_index)

    return frame


def read_tick_csv(
    path: str | Path,
    *,
    schema: TickSchema = DEFAULT_TICK_SCHEMA,
    timestamp_col: str | None = None,
    date_col: str | None = None,
    time_col: str | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a CSV file and normalize it as tick-like market data."""
    data = pd.read_csv(path, **read_csv_kwargs)
    return normalize_tick_data(
        data,
        schema=schema,
        timestamp_col=timestamp_col,
        date_col=date_col,
        time_col=time_col,
    )
