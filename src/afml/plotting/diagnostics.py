"""Diagnostic plots for bars, CUSUM events, and triple-barrier labels."""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

OHLC_COLUMNS = ["open", "high", "low", "close"]


def plot_bars(
    bars: pd.DataFrame,
    *,
    price_col: str = "close",
    ax: Axes | None = None,
    title: str | None = None,
    marker: str = ".",
    style: Literal["auto", "candlestick", "line"] = "auto",
    up_color: str = "tab:green",
    down_color: str = "tab:red",
    wick_color: str = "0.25",
    body_width: float | None = None,
    x_axis: Literal["time", "bar"] = "time",
) -> Axes:
    """Plot a bar price series or OHLC candlesticks.

    This is intended for quick visual inspection of time, tick, volume, dollar,
    imbalance, and runs bars. The DataFrame must use a ``DatetimeIndex``.
    If ``open``, ``high``, ``low``, and ``close`` are available, ``style="auto"``
    draws candlesticks. Otherwise it falls back to a close-price line plot.
    Use ``x_axis="bar"`` for event-based bars when real timestamps are too dense
    or uneven for readable candlestick spacing.
    """
    if style not in {"auto", "candlestick", "line"}:
        raise ValueError("style must be 'auto', 'candlestick', or 'line'")
    if x_axis not in {"time", "bar"}:
        raise ValueError("x_axis must be 'time' or 'bar'")

    required_columns = OHLC_COLUMNS if style == "candlestick" else [price_col]
    bars = _prepare_frame(bars, required_columns=required_columns, name="bars")
    ax = _get_ax(ax)

    has_ohlc = all(column in bars.columns for column in OHLC_COLUMNS)
    use_candlestick = style == "candlestick" or (style == "auto" and has_ohlc)
    if use_candlestick:
        _plot_candlesticks(
            ax,
            bars.loc[:, OHLC_COLUMNS].astype(float),
            up_color=up_color,
            down_color=down_color,
            wick_color=wick_color,
            body_width=body_width,
            x_axis=x_axis,
        )
        ax.set_ylabel("price")
        ax.set_title(title or "OHLC bars")
    else:
        bars = _prepare_frame(bars, required_columns=[price_col], name="bars")
        x_values = _x_values(bars.index, x_axis=x_axis)
        ax.plot(x_values, bars[price_col], marker=marker, linewidth=1.2)
        ax.set_ylabel(price_col)
        ax.set_title(title or f"{price_col} bars")

    ax.set_xlabel("time" if x_axis == "time" else "bar")
    ax.grid(True, alpha=0.25)
    _format_x_axis(ax, x_axis=x_axis)
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
    _format_time_axis(ax)
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
    _format_time_axis(ax)
    return ax


def _plot_candlesticks(
    ax: Axes,
    bars: pd.DataFrame,
    *,
    up_color: str,
    down_color: str,
    wick_color: str,
    body_width: float | None,
    x_axis: Literal["time", "bar"],
) -> None:
    x_values = _x_values(bars.index, x_axis=x_axis)
    width = body_width if body_width is not None else _infer_body_width(x_values, x_axis=x_axis)

    ax.vlines(
        x_values,
        bars["low"].to_numpy(),
        bars["high"].to_numpy(),
        color=wick_color,
        linewidth=0.8,
        alpha=0.85,
        zorder=1,
    )

    for x_value, row in zip(x_values, bars.itertuples(index=False), strict=True):
        open_ = row.open
        close = row.close
        color = up_color if close >= open_ else down_color
        body_bottom = min(open_, close)
        body_height = abs(close - open_)

        if body_height == 0:
            ax.hlines(
                close,
                x_value - width / 2,
                x_value + width / 2,
                color=color,
                linewidth=1.2,
                zorder=2,
            )
            continue

        ax.add_patch(
            Rectangle(
                (x_value - width / 2, body_bottom),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.85,
                zorder=2,
            )
        )

    ax.set_xlim(x_values[0] - width, x_values[-1] + width)
    ax.update_datalim(
        [
            (x_values[0], float(bars["low"].min())),
            (x_values[-1], float(bars["high"].max())),
        ]
    )
    ax.autoscale_view()


def _x_values(index: pd.DatetimeIndex, *, x_axis: Literal["time", "bar"]) -> Sequence[float]:
    if x_axis == "bar":
        return list(range(len(index)))
    return mdates.date2num(index.to_pydatetime())


def _infer_body_width(x_values: Sequence[float], *, x_axis: Literal["time", "bar"]) -> float:
    if x_axis == "bar":
        return 0.65
    if len(x_values) < 2:
        return 1.0 / (24 * 60) * 0.6
    deltas = pd.Series(x_values).diff().dropna()
    positive_deltas = deltas[deltas > 0]
    if positive_deltas.empty:
        return 1.0 / (24 * 60) * 0.6
    return float(positive_deltas.median() * 0.65)


def _format_x_axis(ax: Axes, *, x_axis: Literal["time", "bar"]) -> None:
    if x_axis == "bar":
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=4, nbins=8))
        return
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _format_time_axis(ax: Axes) -> None:
    _format_x_axis(ax, x_axis="time")


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
