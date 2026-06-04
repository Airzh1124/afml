"""Smoke tests for plotting diagnostics."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from afml.plotting import plot_bars, plot_cusum_events, plot_triple_barrier_event


def _close() -> pd.Series:
    return pd.Series(
        [100.0, 101.0, 99.0, 102.0],
        index=pd.date_range("2024-01-01 09:30:00", periods=4, freq="min"),
        name="close",
    )


def test_plot_bars_returns_axes() -> None:
    bars = pd.DataFrame({"close": _close()})

    ax = plot_bars(bars)

    assert ax.get_title() == "close bars"
    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_plot_cusum_events_returns_axes_with_markers() -> None:
    close = _close()
    events = pd.DatetimeIndex([close.index[1], close.index[3]])

    ax = plot_cusum_events(close, events)

    assert ax.get_title() == "CUSUM events"
    assert len(ax.collections) == 1
    plt.close(ax.figure)


def test_plot_triple_barrier_event_returns_axes() -> None:
    close = _close()
    event = pd.Series({"t1": close.index[-1], "trgt": 0.02, "side": 1.0})

    ax = plot_triple_barrier_event(
        close,
        close.index[0],
        event,
        pt_mult=1,
        sl_mult=1,
    )

    assert ax.get_title() == "Triple-barrier event"
    assert len(ax.lines) >= 4
    plt.close(ax.figure)
