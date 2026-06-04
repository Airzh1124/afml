"""Triple-barrier method."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


EVENT_COLUMNS = ("t1", "trgt", "side")
TOUCH_COLUMNS = ("t1", "sl", "pt")
LABEL_COLUMNS = ("t1", "trgt", "side", "pt", "sl", "ret", "label")


def add_vertical_barrier(
    events: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    close: pd.Series,
    *,
    num_days: int | None = None,
    num_hours: int | None = None,
    num_minutes: int | None = None,
) -> pd.Series:
    """Return vertical barrier timestamps for event start times.

    Each barrier is placed at the first close index at or after the requested
    horizon. Events whose horizon goes past the end of ``close`` receive
    ``NaT``.
    """
    close = _prepare_close(close)
    event_index = pd.DatetimeIndex(events)
    if event_index.empty:
        return pd.Series(dtype="datetime64[ns]", index=event_index, name="t1")

    delta = pd.Timedelta(0)
    if num_days is not None:
        delta += pd.Timedelta(days=num_days)
    if num_hours is not None:
        delta += pd.Timedelta(hours=num_hours)
    if num_minutes is not None:
        delta += pd.Timedelta(minutes=num_minutes)
    if delta <= pd.Timedelta(0):
        raise ValueError("vertical barrier horizon must be positive")

    positions = close.index.searchsorted(event_index + delta)
    values = [
        close.index[position] if position < len(close.index) else pd.NaT
        for position in positions
    ]
    return pd.Series(values, index=event_index, name="t1")


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float] | list[float],
    molecule: Iterable[pd.Timestamp] | pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Find first profit-taking or stop-loss touch before each event's ``t1``.

    This follows AFML Snippet 3.2. ``pt_sl`` is ``[pt, sl]``; a non-positive
    value disables that horizontal barrier.
    """
    close = _prepare_close(close)
    events = _prepare_events(events)
    pt, sl = _prepare_pt_sl(pt_sl)
    molecule_index = events.index if molecule is None else pd.DatetimeIndex(molecule)
    events_subset = events.loc[molecule_index]

    out = events_subset[["t1"]].copy(deep=True)
    out["sl"] = pd.NaT
    out["pt"] = pd.NaT

    pt_threshold = pt * events_subset["trgt"] if pt > 0 else None
    sl_threshold = -sl * events_subset["trgt"] if sl > 0 else None

    for event_time, event in events_subset.iterrows():
        end_time = event["t1"]
        if pd.isna(end_time):
            end_time = close.index[-1]

        path = close.loc[event_time:end_time]
        if path.empty:
            continue

        path_returns = path / close.loc[event_time] - 1.0
        path_returns = path_returns * float(event["side"])

        if sl_threshold is not None:
            touched_sl = path_returns[path_returns < sl_threshold.loc[event_time]]
            if not touched_sl.empty:
                out.loc[event_time, "sl"] = touched_sl.index[0]

        if pt_threshold is not None:
            touched_pt = path_returns[path_returns > pt_threshold.loc[event_time]]
            if not touched_pt.empty:
                out.loc[event_time, "pt"] = touched_pt.index[0]

    return out


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl_t1: tuple[float, float, int] | list[float],
) -> pd.DataFrame:
    """Compute labels using profit-taking, stop-loss, and vertical barriers.

    ``pt_sl_t1`` is ``[pt, sl, t1]`` where zero disables a barrier. ``events``
    must include ``trgt`` and may include ``t1`` and ``side``. If no ``side`` is
    supplied, side defaults to ``1``.
    """
    close = _prepare_close(close)
    if len(pt_sl_t1) != 3:
        raise ValueError("pt_sl_t1 must contain [pt, sl, t1]")
    pt, sl, use_t1 = pt_sl_t1
    events = _prepare_events(events)

    if not use_t1:
        events = events.copy()
        events["t1"] = pd.NaT

    touches = apply_pt_sl_on_t1(close, events, (pt, sl))
    first_touch = touches[["t1", "sl", "pt"]].min(axis=1, skipna=True)
    first_touch = first_touch.fillna(close.index[-1])

    out = events.copy()
    out["pt"] = touches["pt"]
    out["sl"] = touches["sl"]
    out["ret"] = np.nan
    out["label"] = 0

    for event_time, touch_time in first_touch.items():
        ret = close.loc[touch_time] / close.loc[event_time] - 1.0
        ret *= float(out.loc[event_time, "side"])
        out.loc[event_time, "ret"] = ret
        out.loc[event_time, "label"] = _label_from_return(ret)

    return out.loc[:, list(LABEL_COLUMNS)]


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


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame")
    if not isinstance(events.index, pd.DatetimeIndex):
        raise TypeError("events must be indexed by a pandas DatetimeIndex")
    if "trgt" not in events.columns:
        raise ValueError("events must include a 'trgt' column")
    if (events["trgt"] <= 0).any():
        raise ValueError("events targets must be positive")

    prepared = events.copy().sort_index(kind="mergesort")
    if "t1" not in prepared.columns:
        prepared["t1"] = pd.NaT
    prepared["t1"] = pd.to_datetime(prepared["t1"])
    if "side" not in prepared.columns:
        prepared["side"] = 1.0
    if prepared[["trgt", "side"]].isna().any().any():
        raise ValueError("events contain missing trgt or side values")
    return prepared


def _prepare_pt_sl(pt_sl: tuple[float, float] | list[float]) -> tuple[float, float]:
    if len(pt_sl) != 2:
        raise ValueError("pt_sl must contain [pt, sl]")
    pt, sl = float(pt_sl[0]), float(pt_sl[1])
    if pt < 0 or sl < 0:
        raise ValueError("pt and sl must be non-negative")
    return pt, sl


def _label_from_return(ret: float) -> int:
    if ret > 0:
        return 1
    if ret < 0:
        return -1
    return 0
