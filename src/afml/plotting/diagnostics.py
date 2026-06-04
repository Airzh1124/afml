"""Diagnostic plots for bars, CUSUM events, and triple-barrier labels."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes


def plot_bars(
    bars: pd.DataFrame,
    *,
    price_col: str = "close",
    ax: Axes | None = None,
    title: str | None = None,
    marker: str = ".",
) -> Axes:
    """Plot a bar price series.

    This is intended for quick visual inspection of time, tick, volume, dollar,
    imbalance, and runs bars. The DataFrame must use a ``DatetimeIndex``.
    """
    bars = _prepare_frame(bars, required_columns=[price_col], name="bars")
    ax = _get_ax(ax)
    ax.plot(bars.index, bars[price_col], marker=marker, linewidth=1.2)
    ax.set_xlabel("time")
    ax.set_ylabel(price_col)
    ax.set_title(title or f"{price_col} bars")
    ax.grid(True, alpha=0.25)
    return ax


def plot_cusum_events(
    close: pd.Series,
    events: pd.DatetimeIndex | pd.Series,
    *,
    ax: Axes | None = None,
    title: str | None = None,
    event_color: str = "tab:red",
) -> Axes:
    """Plot close prices with CUSUM event timestamps marked."""
    close = _prepare_series(close, name="close")
    event_index = _event_index(events)
    event_prices = close.reindex(event_index).dropna()

    ax = _get_ax(ax)
    ax.plot(close.index, close, linewidth=1.2, label=close.name or "close")
    ax.scatter(
        event_prices.index,
        event_prices,
        color=event_color,
        s=28,
        label="CUSUM event",
        zorder=3,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.set_title(title or "CUSUM events")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_triple_barrier_event(
    close: pd.Series,
    event_time: pd.Timestamp,
    event: pd.Series,
    *,
    pt_mult: float,
    sl_mult: float,
    ax: Axes | None = None,
    title: str | None = None,
    price_space: Literal["return", "price"] = "return",
) -> Axes:
    """Plot the triple-barrier box for one event.

    ``event`` should contain ``t1``, ``trgt``, and optionally ``side``. The plot
    shows the path from event start to vertical barrier plus the horizontal
    profit-taking and stop-loss barriers.
    """
    close = _prepare_series(close, name="close")
    if event_time not in close.index:
        raise ValueError("event_time must be present in close.index")
    if "trgt" not in event.index:
        raise ValueError("event must include 'trgt'")
    if pt_mult < 0 or sl_mult < 0:
        raise ValueError("pt_mult and sl_mult must be non-negative")
    if price_space not in {"return", "price"}:
        raise ValueError("price_space must be 'return' or 'price'")

    side = float(event["side"]) if "side" in event.index and pd.notna(event["side"]) else 1.0
    end_time = event["t1"] if "t1" in event.index and pd.notna(event["t1"]) else close.index[-1]
    path = close.loc[event_time:end_time]
    if path.empty:
        raise ValueError("event path is empty")

    event_price = close.loc[event_time]
    path_returns = (path / event_price - 1.0) * side
    target = float(event["trgt"])
    pt_level = pt_mult * target if pt_mult > 0 else None
    sl_level = -sl_mult * target if sl_mult > 0 else None

    ax = _get_ax(ax)
    if price_space == "return":
        y = path_returns
        ax.plot(y.index, y, linewidth=1.2, label="path return")
        if pt_level is not None:
            ax.axhline(pt_level, color="tab:green", linestyle="--", label="profit taking")
        if sl_level is not None:
            ax.axhline(sl_level, color="tab:red", linestyle="--", label="stop loss")
        ax.set_ylabel("side-adjusted return")
    else:
        y = path
        ax.plot(y.index, y, linewidth=1.2, label="price path")
        if pt_level is not None:
            ax.axhline(
                event_price * (1.0 + pt_level * side),
                color="tab:green",
                linestyle="--",
                label="profit taking",
            )
        if sl_level is not None:
            ax.axhline(
                event_price * (1.0 + sl_level * side),
                color="tab:red",
                linestyle="--",
                label="stop loss",
            )
        ax.set_ylabel("price")

    ax.axvline(event_time, color="black", linestyle=":", label="event")
    ax.axvline(path.index[-1], color="tab:blue", linestyle=":", label="vertical barrier")
    ax.set_xlabel("time")
    ax.set_title(title or "Triple-barrier event")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    required_columns: list[str],
    name: str,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by a pandas DatetimeIndex")
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    return frame.sort_index(kind="mergesort")


def _prepare_series(series: pd.Series, *, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must be indexed by a pandas DatetimeIndex")
    if series.empty:
        raise ValueError(f"{name} must not be empty")
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    return series.astype(float).sort_index(kind="mergesort")


def _event_index(events: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    if isinstance(events, pd.DatetimeIndex):
        return events
    if isinstance(events, pd.Series):
        return pd.DatetimeIndex(events.dropna())
    raise TypeError("events must be a DatetimeIndex or Series of timestamps")


def _get_ax(ax: Axes | None) -> Axes:
    if ax is not None:
        return ax
    _, created_ax = plt.subplots()
    return created_ax
