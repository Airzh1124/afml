"""Return calculations used for labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Label events by realized return at the first touched barrier.

    This follows AFML Snippet 3.5. ``events`` must contain ``t1`` values already
    computed by ``get_events``. Events with missing ``t1`` are skipped.
    """
    if "t1" not in events.columns:
        raise ValueError("events must include a 't1' column")
    close = _prepare_close(close)

    events_ = events.dropna(subset=["t1"])
    if events_.empty:
        return pd.DataFrame(
            {"ret": pd.Series(dtype=float), "bin": pd.Series(dtype=float)},
            index=events_.index,
        )

    price_index = events_.index.union(pd.DatetimeIndex(events_["t1"])).drop_duplicates()
    prices = close.reindex(price_index, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["ret"] = prices.loc[events_["t1"].to_numpy()].to_numpy() / prices.loc[events_.index] - 1.0
    out["bin"] = np.sign(out["ret"])
    return out


def event_returns(close: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Return close-to-barrier returns for events with an end time ``t1``."""
    if "t1" not in events.columns:
        raise ValueError("events must include a 't1' column")

    returns = {}
    for event_time, t1 in events["t1"].items():
        if pd.isna(t1):
            continue
        returns[event_time] = close.loc[t1] / close.loc[event_time] - 1.0
    return pd.Series(returns, name="ret")


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
