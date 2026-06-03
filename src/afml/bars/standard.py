"""Time, tick, volume, and dollar bars.

Core bar constructors use a pandas ``DatetimeIndex`` as the market time axis.
Raw data with a timestamp column should be normalized in the data layer before
calling these functions.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


BAR_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "dollar_value",
    "tick_count",
)


def time_bars(
    ticks: pd.DataFrame,
    freq: str,
    *,
    price_col: str = "price",
    volume_col: str = "volume",
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Build time bars from tick-like data.

    For time bars, the output index follows pandas resampling semantics and is
    the left edge of each time window by default.
    """
    frame = _prepare_ticks(ticks, price_col, volume_col)
    if frame.empty:
        return _empty_bars(index_name=frame.index.name)

    dollar_value = frame[price_col] * frame[volume_col]
    grouped = frame.assign(dollar_value=dollar_value).resample(freq)

    bars = pd.DataFrame(
        {
            "open": grouped[price_col].first(),
            "high": grouped[price_col].max(),
            "low": grouped[price_col].min(),
            "close": grouped[price_col].last(),
            "volume": grouped[volume_col].sum(),
            "dollar_value": grouped["dollar_value"].sum(),
            "tick_count": grouped[price_col].count(),
        }
    )

    if drop_empty:
        bars = bars[bars["tick_count"] > 0]

    return _finalize_bars(bars)


def tick_bars(
    ticks: pd.DataFrame,
    threshold: int,
    *,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial: bool = True,
) -> pd.DataFrame:
    """Build bars after a fixed number of ticks.

    The output index is each bar's ending timestamp.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    frame = _prepare_ticks(ticks, price_col, volume_col)
    boundaries = _threshold_boundaries(
        frame,
        values=pd.Series(1, index=frame.index),
        threshold=threshold,
        include_partial=include_partial,
    )
    return _bars_from_boundaries(frame, boundaries, price_col, volume_col)


def volume_bars(
    ticks: pd.DataFrame,
    threshold: float,
    *,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial: bool = True,
) -> pd.DataFrame:
    """Build bars after cumulative volume reaches ``threshold``.

    The output index is each bar's ending timestamp.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    frame = _prepare_ticks(ticks, price_col, volume_col)
    boundaries = _threshold_boundaries(
        frame,
        values=frame[volume_col],
        threshold=threshold,
        include_partial=include_partial,
    )
    return _bars_from_boundaries(frame, boundaries, price_col, volume_col)


def dollar_bars(
    ticks: pd.DataFrame,
    threshold: float,
    *,
    price_col: str = "price",
    volume_col: str = "volume",
    include_partial: bool = True,
) -> pd.DataFrame:
    """Build bars after cumulative dollar value reaches ``threshold``.

    The output index is each bar's ending timestamp.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    frame = _prepare_ticks(ticks, price_col, volume_col)
    dollar_value = frame[price_col] * frame[volume_col]
    boundaries = _threshold_boundaries(
        frame,
        values=dollar_value,
        threshold=threshold,
        include_partial=include_partial,
    )
    return _bars_from_boundaries(frame, boundaries, price_col, volume_col)


def _prepare_ticks(
    ticks: pd.DataFrame,
    price_col: str,
    volume_col: str,
) -> pd.DataFrame:
    """Validate and normalize tick-like input data."""
    missing = [column for column in (price_col, volume_col) if column not in ticks.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if not isinstance(ticks.index, pd.DatetimeIndex):
        raise TypeError("ticks must be indexed by a pandas DatetimeIndex")

    frame = ticks.loc[:, [price_col, volume_col]].copy()
    frame = frame.sort_index(kind="mergesort")

    if frame.index.hasnans:
        raise ValueError("ticks index contains missing timestamps")
    if frame[[price_col, volume_col]].isna().any().any():
        raise ValueError("ticks contain missing price or volume values")
    if (frame[volume_col] < 0).any():
        raise ValueError("volume must be non-negative")

    return frame


def _threshold_boundaries(
    frame: pd.DataFrame,
    values: pd.Series,
    threshold: float,
    include_partial: bool,
) -> list[tuple[int, int]]:
    """Return inclusive start/end row positions for threshold-triggered bars."""
    boundaries: list[tuple[int, int]] = []
    start = 0
    cumulative = 0.0

    for position, value in enumerate(values):
        cumulative += float(value)
        if cumulative >= threshold:
            boundaries.append((start, position))
            start = position + 1
            cumulative = 0.0

    if include_partial and start < len(frame):
        boundaries.append((start, len(frame) - 1))

    return boundaries


def _bars_from_boundaries(
    frame: pd.DataFrame,
    boundaries: Iterable[tuple[int, int]],
    price_col: str,
    volume_col: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    index = []

    for start, end in boundaries:
        chunk = frame.iloc[start : end + 1]
        index.append(chunk.index[-1])
        records.append(
            {
                "open": chunk[price_col].iloc[0],
                "high": chunk[price_col].max(),
                "low": chunk[price_col].min(),
                "close": chunk[price_col].iloc[-1],
                "volume": chunk[volume_col].sum(),
                "dollar_value": (chunk[price_col] * chunk[volume_col]).sum(),
                "tick_count": len(chunk),
            }
        )

    bars = pd.DataFrame.from_records(records, columns=BAR_COLUMNS)
    bars.index = pd.DatetimeIndex(index, name=frame.index.name)
    return _finalize_bars(bars)


def _empty_bars(index_name: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], name=index_name))


def _finalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return _empty_bars(index_name=bars.index.name)
    return bars.loc[:, BAR_COLUMNS]
