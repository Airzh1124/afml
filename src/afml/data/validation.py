"""Data validation helpers."""

from __future__ import annotations

import pandas as pd

from afml.data.schemas import DEFAULT_TICK_SCHEMA, TickSchema


def validate_tick_data(
    ticks: pd.DataFrame,
    *,
    schema: TickSchema = DEFAULT_TICK_SCHEMA,
    require_monotonic: bool = True,
    allow_duplicate_timestamps: bool = True,
) -> None:
    """Validate normalized tick-like data.

    Parameters
    ----------
    ticks
        DataFrame indexed by market timestamps.
    schema
        Column-name schema. Required columns are price and volume.
    require_monotonic
        Whether the index must be sorted increasingly.
    allow_duplicate_timestamps
        Whether multiple trades may share the same timestamp. This is common in
        futures tick data, so the default allows it.
    """
    missing = [column for column in schema.required_columns if column not in ticks.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if not isinstance(ticks.index, pd.DatetimeIndex):
        raise TypeError("ticks must be indexed by a pandas DatetimeIndex")

    if ticks.index.hasnans:
        raise ValueError("ticks index contains missing timestamps")

    if require_monotonic and not ticks.index.is_monotonic_increasing:
        raise ValueError("ticks index must be sorted increasingly")

    if not allow_duplicate_timestamps and ticks.index.has_duplicates:
        raise ValueError("ticks index contains duplicate timestamps")

    required = list(schema.required_columns)
    if ticks[required].isna().any().any():
        raise ValueError("ticks contain missing price or volume values")

    if (ticks[schema.price] <= 0).any():
        raise ValueError("price must be positive")

    if (ticks[schema.volume] < 0).any():
        raise ValueError("volume must be non-negative")

    if schema.bid in ticks.columns and schema.ask in ticks.columns:
        spread = ticks[schema.ask] - ticks[schema.bid]
        if spread.dropna().lt(0).any():
            raise ValueError("ask must be greater than or equal to bid")
