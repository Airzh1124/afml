"""Return calculations used for labeling."""

from __future__ import annotations

import pandas as pd


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
