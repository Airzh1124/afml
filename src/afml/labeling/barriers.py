"""Triple-barrier method."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


EVENT_COLUMNS = ("t1", "trgt", "side")
TOUCH_COLUMNS = ("t1", "sl", "pt")


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


def get_events(
    close: pd.Series,
    t_events: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    pt_sl: float | tuple[float, float] | list[float],
    trgt: pd.Series,
    min_ret: float,
    *,
    t1: pd.Series | bool = False,
    num_threads: int = 1,
) -> pd.DataFrame:
    """Find the first barrier touch for sampled events.

    This follows AFML Snippet 3.3 for the case where the bet side is unknown.
    The horizontal barriers are therefore symmetric. ``num_threads`` is accepted
    for API compatibility with the book, but this implementation runs in-process.
    """
    close = _prepare_close(close)
    event_index = pd.DatetimeIndex(t_events)
    if min_ret < 0:
        raise ValueError("min_ret must be non-negative")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    pt, sl = _prepare_symmetric_pt_sl(pt_sl)
    target = _prepare_target(trgt)
    target = target.reindex(event_index).dropna()
    target = target[target > min_ret]

    if target.empty:
        return pd.DataFrame(
            {"t1": pd.Series(dtype="datetime64[ns]"), "trgt": pd.Series(dtype=float)},
            index=target.index,
        )

    t1_series = _prepare_t1(t1, target.index)
    side = pd.Series(1.0, index=target.index, name="side")
    events = pd.concat({"t1": t1_series, "trgt": target, "side": side}, axis=1)
    events = events.dropna(subset=["trgt"])

    touches = apply_pt_sl_on_t1(close, events, (pt, sl))
    events["t1"] = touches.dropna(how="all").min(axis=1, skipna=True)
    return events.drop(columns="side").loc[:, ["t1", "trgt"]]


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


def _prepare_symmetric_pt_sl(
    pt_sl: float | tuple[float, float] | list[float],
) -> tuple[float, float]:
    if isinstance(pt_sl, (int, float)):
        value = float(pt_sl)
        if value < 0:
            raise ValueError("pt_sl must be non-negative")
        return value, value

    pt, sl = _prepare_pt_sl(pt_sl)
    if pt != sl:
        raise ValueError("pt_sl must be symmetric when side is unknown")
    return pt, sl


def _prepare_target(trgt: pd.Series) -> pd.Series:
    if not isinstance(trgt, pd.Series):
        raise TypeError("trgt must be a pandas Series")
    if not isinstance(trgt.index, pd.DatetimeIndex):
        raise TypeError("trgt must be indexed by a pandas DatetimeIndex")
    if (trgt.dropna() <= 0).any():
        raise ValueError("trgt values must be positive")
    return trgt.astype(float).sort_index(kind="mergesort")


def _prepare_t1(t1: pd.Series | bool, event_index: pd.DatetimeIndex) -> pd.Series:
    if isinstance(t1, bool):
        if t1 is not False:
            raise ValueError("t1 must be a pandas Series or False")
        return pd.Series(pd.NaT, index=event_index, name="t1")
    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pandas Series or False")
    if not isinstance(t1.index, pd.DatetimeIndex):
        raise TypeError("t1 must be indexed by a pandas DatetimeIndex")
    return pd.to_datetime(t1.reindex(event_index)).rename("t1")

