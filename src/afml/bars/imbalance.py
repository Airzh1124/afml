"""Tick, volume, and dollar imbalance bars."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from afml.bars.standard import BAR_COLUMNS, _prepare_ticks


IMBALANCE_COLUMNS = BAR_COLUMNS + (
    "theta",
    "threshold",
    "expected_ticks",
    "expected_imbalance",
)


def tick_rule(prices: pd.Series, *, initial_direction: int = 1) -> pd.Series:
    """Compute tick-rule signs from a price series.

    ``b_t`` is ``+1`` for an uptick, ``-1`` for a downtick, and inherits the
    previous non-zero direction when the price is unchanged.
    """
    if initial_direction not in (-1, 1):
        raise ValueError("initial_direction must be either -1 or 1")

    if prices.empty:
        return pd.Series(dtype="int8", index=prices.index, name="tick_rule")

    price_diff = prices.diff()
    signs = np.sign(price_diff.to_numpy(dtype=float, copy=False))
    signs[0] = initial_direction
    signs = pd.Series(signs, index=prices.index).replace(0, np.nan).ffill()
    return signs.astype("int8").rename("tick_rule")


def tick_imbalance_bars(
    ticks: pd.DataFrame,
    *,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    expected_ticks_window: int = 20,
    expected_imbalance_window: int = 20,
    min_expected_imbalance: float = 1e-6,
    price_col: str = "price",
    volume_col: str = "volume",
    initial_direction: int = 1,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Build tick imbalance bars.

    ``theta_T`` accumulates ``b_t`` and the stopping rule is:

    ``abs(theta_T) >= E[T] * max(abs(E[b_t]), min_expected_imbalance)``
    """
    _validate_imbalance_params(
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
        expected_imbalance_min=-1,
        expected_imbalance_max=1,
    )

    frame = _prepare_ticks(ticks, price_col, volume_col)
    b = tick_rule(frame[price_col], initial_direction=initial_direction).to_numpy()
    return _imbalance_bars(
        frame,
        imbalance_values=b.astype(float),
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
        price_col=price_col,
        volume_col=volume_col,
        include_partial=include_partial,
    )


def volume_imbalance_bars(
    ticks: pd.DataFrame,
    *,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    expected_ticks_window: int = 20,
    expected_imbalance_window: int = 20,
    min_expected_imbalance: float = 1e-6,
    price_col: str = "price",
    volume_col: str = "volume",
    initial_direction: int = 1,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Build volume imbalance bars.

    ``theta_T`` accumulates ``b_t * volume_t`` and the stopping rule is:

    ``abs(theta_T) >= E[T] * max(abs(E[b_t * volume_t]), min_expected_imbalance)``
    """
    _validate_imbalance_params(
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
    )

    frame = _prepare_ticks(ticks, price_col, volume_col)
    b = tick_rule(frame[price_col], initial_direction=initial_direction).to_numpy()
    volume = frame[volume_col].to_numpy(dtype=float, copy=False)
    return _imbalance_bars(
        frame,
        imbalance_values=b * volume,
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
        price_col=price_col,
        volume_col=volume_col,
        include_partial=include_partial,
    )


def dollar_imbalance_bars(
    ticks: pd.DataFrame,
    *,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    expected_ticks_window: int = 20,
    expected_imbalance_window: int = 20,
    min_expected_imbalance: float = 1e-6,
    price_col: str = "price",
    volume_col: str = "volume",
    initial_direction: int = 1,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Build dollar imbalance bars.

    ``theta_T`` accumulates ``b_t * price_t * volume_t`` and the stopping rule is:

    ``abs(theta_T) >= E[T] * max(abs(E[b_t * price_t * volume_t]), min_expected_imbalance)``
    """
    _validate_imbalance_params(
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
    )

    frame = _prepare_ticks(ticks, price_col, volume_col)
    b = tick_rule(frame[price_col], initial_direction=initial_direction).to_numpy()
    price = frame[price_col].to_numpy(dtype=float, copy=False)
    volume = frame[volume_col].to_numpy(dtype=float, copy=False)
    return _imbalance_bars(
        frame,
        imbalance_values=b * price * volume,
        expected_ticks_init=expected_ticks_init,
        expected_imbalance_init=expected_imbalance_init,
        expected_ticks_window=expected_ticks_window,
        expected_imbalance_window=expected_imbalance_window,
        min_expected_imbalance=min_expected_imbalance,
        price_col=price_col,
        volume_col=volume_col,
        include_partial=include_partial,
    )


def _imbalance_bars(
    frame: pd.DataFrame,
    *,
    imbalance_values: np.ndarray,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    expected_ticks_window: int,
    expected_imbalance_window: int,
    min_expected_imbalance: float,
    price_col: str,
    volume_col: str,
    include_partial: bool,
) -> pd.DataFrame:
    if frame.empty:
        return _empty_imbalance_bars(index_name=frame.index.name)

    price = frame[price_col].to_numpy()
    volume = frame[volume_col].to_numpy()
    dollar_value = price * volume

    expected_ticks = float(expected_ticks_init)
    expected_imbalance = float(expected_imbalance_init)
    ticks_alpha = _ewma_alpha(expected_ticks_window)
    imbalance_alpha = _ewma_alpha(expected_imbalance_window)

    records: list[dict[str, Any]] = []
    index = []
    start = 0
    theta = 0.0

    for position, imbalance in enumerate(imbalance_values):
        theta += float(imbalance)
        threshold = _imbalance_threshold(
            expected_ticks=expected_ticks,
            expected_imbalance=expected_imbalance,
            min_expected_imbalance=min_expected_imbalance,
        )

        if abs(theta) >= threshold:
            bar_length = position - start + 1
            _append_imbalance_bar(
                records=records,
                index=index,
                frame=frame,
                price=price,
                volume=volume,
                dollar_value=dollar_value,
                start=start,
                end=position,
                theta=theta,
                threshold=threshold,
                expected_ticks=expected_ticks,
                expected_imbalance=expected_imbalance,
            )
            expected_ticks = _ewma_update(expected_ticks, bar_length, ticks_alpha)
            expected_imbalance = _ewma_update(
                expected_imbalance,
                theta / bar_length,
                imbalance_alpha,
            )
            start = position + 1
            theta = 0.0

    if include_partial and start < len(frame):
        threshold = _imbalance_threshold(
            expected_ticks=expected_ticks,
            expected_imbalance=expected_imbalance,
            min_expected_imbalance=min_expected_imbalance,
        )
        _append_imbalance_bar(
            records=records,
            index=index,
            frame=frame,
            price=price,
            volume=volume,
            dollar_value=dollar_value,
            start=start,
            end=len(frame) - 1,
            theta=theta,
            threshold=threshold,
            expected_ticks=expected_ticks,
            expected_imbalance=expected_imbalance,
        )

    bars = pd.DataFrame.from_records(records, columns=IMBALANCE_COLUMNS)
    bars.index = pd.DatetimeIndex(index, name=frame.index.name)
    return _finalize_imbalance_bars(bars)


def _validate_imbalance_params(
    *,
    expected_ticks_init: float,
    expected_imbalance_init: float,
    expected_ticks_window: int,
    expected_imbalance_window: int,
    min_expected_imbalance: float,
    expected_imbalance_min: float | None = None,
    expected_imbalance_max: float | None = None,
) -> None:
    if expected_ticks_init <= 0:
        raise ValueError("expected_ticks_init must be positive")
    if (
        expected_imbalance_min is not None
        and expected_imbalance_init < expected_imbalance_min
    ) or (
        expected_imbalance_max is not None
        and expected_imbalance_init > expected_imbalance_max
    ):
        raise ValueError(
            "expected_imbalance_init must be between "
            f"{expected_imbalance_min} and {expected_imbalance_max}"
        )
    if expected_ticks_window <= 0:
        raise ValueError("expected_ticks_window must be positive")
    if expected_imbalance_window <= 0:
        raise ValueError("expected_imbalance_window must be positive")
    if min_expected_imbalance <= 0:
        raise ValueError("min_expected_imbalance must be positive")


def _append_imbalance_bar(
    *,
    records: list[dict[str, Any]],
    index: list[pd.Timestamp],
    frame: pd.DataFrame,
    price: np.ndarray,
    volume: np.ndarray,
    dollar_value: np.ndarray,
    start: int,
    end: int,
    theta: float,
    threshold: float,
    expected_ticks: float,
    expected_imbalance: float,
) -> None:
    price_slice = price[start : end + 1]
    volume_slice = volume[start : end + 1]
    dollar_slice = dollar_value[start : end + 1]

    index.append(frame.index[end])
    records.append(
        {
            "open": price_slice[0],
            "high": price_slice.max(),
            "low": price_slice.min(),
            "close": price_slice[-1],
            "volume": volume_slice.sum(),
            "dollar_value": dollar_slice.sum(),
            "tick_count": end - start + 1,
            "theta": theta,
            "threshold": threshold,
            "expected_ticks": expected_ticks,
            "expected_imbalance": expected_imbalance,
        }
    )


def _imbalance_threshold(
    *,
    expected_ticks: float,
    expected_imbalance: float,
    min_expected_imbalance: float,
) -> float:
    return expected_ticks * max(abs(expected_imbalance), min_expected_imbalance)


def _ewma_alpha(window: int) -> float:
    return 2.0 / (window + 1.0)


def _ewma_update(previous: float, observed: float, alpha: float) -> float:
    return alpha * observed + (1.0 - alpha) * previous


def _empty_imbalance_bars(index_name: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        columns=IMBALANCE_COLUMNS,
        index=pd.DatetimeIndex([], name=index_name),
    )


def _finalize_imbalance_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return _empty_imbalance_bars(index_name=bars.index.name)
    return bars.loc[:, IMBALANCE_COLUMNS]
