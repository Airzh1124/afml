"""Single future roll utilities."""

from __future__ import annotations

import pandas as pd


BLOOMBERG_FIELD_MAP = {
    "instrument": "FUT_CUR_GEN_TICKER",
    "open": "PX_OPEN",
    "close": "PX_LAST",
}


def roll_gaps(
    series: pd.DataFrame,
    *,
    field_map: dict[str, str] | None = None,
    match_end: bool = True,
) -> pd.Series:
    """Compute cumulative futures roll gaps.

    Gaps are computed at each contract roll as the new contract's open minus the
    previous contract's close. Subtracting this cumulative gap from raw prices
    produces a continuous rolled futures series.

    ``match_end=True`` rolls backward, so the adjusted series matches the raw
    series at the end. ``match_end=False`` rolls forward, so the adjusted series
    matches the raw series at the start.
    """
    fields = BLOOMBERG_FIELD_MAP | (field_map or {})
    _validate_roll_input(series, fields)

    frame = series.sort_index(kind="mergesort")
    instrument = fields["instrument"]
    open_col = fields["open"]
    close_col = fields["close"]

    roll_dates = frame[instrument].drop_duplicates(keep="first").index
    gaps = pd.Series(0.0, index=frame.index, name="roll_gap")

    if len(roll_dates) <= 1:
        return gaps

    previous_positions = [frame.index.get_loc(date) - 1 for date in roll_dates]
    gap_values = (
        frame.loc[roll_dates[1:], open_col].to_numpy(dtype=float)
        - frame.iloc[previous_positions[1:]][close_col].to_numpy(dtype=float)
    )
    gaps.loc[roll_dates[1:]] = gap_values
    gaps = gaps.cumsum()

    if match_end:
        gaps = gaps - gaps.iloc[-1]

    return gaps


def roll_futures_series(
    series: pd.DataFrame,
    *,
    price_columns: tuple[str, ...] = ("PX_LAST", "VWAP"),
    field_map: dict[str, str] | None = None,
    match_end: bool = True,
) -> pd.DataFrame:
    """Return a futures series with cumulative roll gaps subtracted from prices."""
    fields = BLOOMBERG_FIELD_MAP | (field_map or {})
    _validate_roll_input(series, fields)

    missing_prices = [column for column in price_columns if column not in series.columns]
    if missing_prices:
        raise ValueError(f"missing price columns: {missing_prices}")

    adjusted = series.sort_index(kind="mergesort").copy()
    gaps = roll_gaps(adjusted, field_map=fields, match_end=match_end)
    for column in price_columns:
        adjusted[column] = adjusted[column] - gaps
    adjusted["roll_gap"] = gaps
    return adjusted


def _validate_roll_input(series: pd.DataFrame, fields: dict[str, str]) -> None:
    if not isinstance(series, pd.DataFrame):
        raise TypeError("series must be a pandas DataFrame")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must be indexed by a pandas DatetimeIndex")
    if series.empty:
        raise ValueError("series must not be empty")

    required_keys = {"instrument", "open", "close"}
    missing_keys = required_keys.difference(fields)
    if missing_keys:
        raise ValueError(f"field_map is missing keys: {sorted(missing_keys)}")

    required_columns = [fields[key] for key in sorted(required_keys)]
    missing_columns = [column for column in required_columns if column not in series.columns]
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    if series.index.hasnans:
        raise ValueError("series index contains missing timestamps")
    if series[required_columns].isna().any().any():
        raise ValueError("series contains missing roll fields")
