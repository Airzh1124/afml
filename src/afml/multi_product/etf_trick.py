"""ETF trick for valuing one dollar invested in a spread."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import pandas as pd


AlignedInput: TypeAlias = float | int | pd.Series | pd.DataFrame

ETF_COLUMNS = (
    "value",
    "gross_pnl",
    "rebalance_cost",
    "net_pnl",
    "bid_ask_cost",
    "volume",
)


def etf_trick(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    point_value: AlignedInput = 1.0,
    volume: pd.DataFrame | None = None,
    carry: AlignedInput = 0.0,
    transaction_cost: AlignedInput = 0.0,
    rebalance: pd.Series | None = None,
    initial_value: float = 1.0,
    return_holdings: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the value series of one dollar invested in a futures spread.

    ``weights`` are spread weights ``omega_{i,t}``. Holdings are scaled so the
    virtual ETF has gross exposure equal to its current value:

    ``h_{i,t} = K_t * omega_{i,t} / (sum_j |omega_{j,t}| * p_{i,t} * phi_{i,t})``

    Rebalance costs follow AFML's ETF trick convention and are charged as a
    negative dividend, not embedded in the position scale.
    """
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    open_, close, weights = _prepare_core_frames(open_, close, weights)
    phi = _align_like(point_value, close, name="point_value")
    carry_frame = _align_like(carry, close, name="carry")
    tau = _align_like(transaction_cost, close, name="transaction_cost")
    rebalance_flags = _prepare_rebalance(rebalance, close.index)

    if volume is not None:
        volume = _align_frame(volume, close, name="volume")

    if (close <= 0).any().any():
        raise ValueError("close prices must be positive")
    if (open_ <= 0).any().any():
        raise ValueError("open prices must be positive")
    if (phi <= 0).any().any():
        raise ValueError("point_value must be positive")
    if (tau < 0).any().any():
        raise ValueError("transaction_cost must be non-negative")
    if volume is not None and (volume < 0).any().any():
        raise ValueError("volume must be non-negative")

    index = close.index
    columns = close.columns
    open_np = open_.to_numpy(dtype=float, copy=False)
    close_np = close.to_numpy(dtype=float, copy=False)
    weights_np = weights.to_numpy(dtype=float, copy=False)
    phi_np = phi.to_numpy(dtype=float, copy=False)
    carry_np = carry_frame.to_numpy(dtype=float, copy=False)
    tau_np = tau.to_numpy(dtype=float, copy=False)
    volume_np = (
        volume.to_numpy(dtype=float, copy=False)
        if volume is not None
        else None
    )

    values = np.empty(len(index), dtype=float)
    gross_pnl = np.zeros(len(index), dtype=float)
    rebalance_cost = np.zeros(len(index), dtype=float)
    net_pnl = np.zeros(len(index), dtype=float)
    bid_ask_cost = np.full(len(index), np.nan, dtype=float)
    etf_volume = np.full(len(index), np.nan, dtype=float)
    holdings = np.zeros_like(close_np, dtype=float)

    values[0] = float(initial_value)
    holdings[0] = _holdings_from_weights(
        value=values[0],
        weights=weights_np[0],
        close=close_np[0],
        point_value=phi_np[0],
    )

    for row in range(1, len(index)):
        prev_holdings = holdings[row - 1]
        price_move = (close_np[row] - close_np[row - 1]) * phi_np[row]
        gross_pnl[row] = np.sum(prev_holdings * (price_move + carry_np[row]))
        pre_cost_value = values[row - 1] + gross_pnl[row]

        bid_ask_cost[row] = np.sum(
            np.abs(prev_holdings) * close_np[row] * phi_np[row] * tau_np[row]
        )
        if volume_np is not None:
            etf_volume[row] = _tradeable_volume(volume_np[row], prev_holdings)

        can_rebalance = bool(rebalance_flags.iloc[row]) and row + 1 < len(index)
        if can_rebalance:
            target_holdings_pre_cost = _holdings_from_weights(
                value=pre_cost_value,
                weights=weights_np[row],
                close=close_np[row],
                point_value=phi_np[row],
            )
            rebalance_cost[row] = np.sum(
                (
                    np.abs(prev_holdings) * close_np[row]
                    + np.abs(target_holdings_pre_cost) * open_np[row + 1]
                )
                * phi_np[row]
                * tau_np[row]
            )
            values[row] = pre_cost_value - rebalance_cost[row]
            holdings[row] = _holdings_from_weights(
                value=values[row],
                weights=weights_np[row],
                close=close_np[row],
                point_value=phi_np[row],
            )
        else:
            values[row] = pre_cost_value
            holdings[row] = prev_holdings

        net_pnl[row] = values[row] - values[row - 1]

    result = pd.DataFrame(
        {
            "value": values,
            "gross_pnl": gross_pnl,
            "rebalance_cost": rebalance_cost,
            "net_pnl": net_pnl,
            "bid_ask_cost": bid_ask_cost,
            "volume": etf_volume,
        },
        index=index,
        columns=list(ETF_COLUMNS),
    )
    holdings_frame = pd.DataFrame(holdings, index=index, columns=columns)

    if return_holdings:
        return result, holdings_frame
    return result


def _prepare_core_frames(
    open_: pd.DataFrame,
    close: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, frame in {
        "open_": open_,
        "close": close,
        "weights": weights,
    }.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{name} must be indexed by a pandas DatetimeIndex")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
        if frame.isna().any().any():
            raise ValueError(f"{name} contains missing values")

    if not open_.index.equals(close.index) or not weights.index.equals(close.index):
        raise ValueError("open_, close, and weights must have matching indexes")
    if not open_.columns.equals(close.columns) or not weights.columns.equals(close.columns):
        raise ValueError("open_, close, and weights must have matching columns")

    return (
        open_.sort_index(kind="mergesort"),
        close.sort_index(kind="mergesort"),
        weights.sort_index(kind="mergesort"),
    )


def _align_like(value: AlignedInput, template: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if isinstance(value, (float, int)):
        return pd.DataFrame(float(value), index=template.index, columns=template.columns)

    if isinstance(value, pd.DataFrame):
        return _align_frame(value, template, name=name)

    if isinstance(value, pd.Series):
        if value.index.equals(template.index):
            return pd.DataFrame(
                np.repeat(value.to_numpy(dtype=float)[:, None], len(template.columns), axis=1),
                index=template.index,
                columns=template.columns,
            )
        if value.index.equals(template.columns):
            return pd.DataFrame(
                np.repeat(value.to_numpy(dtype=float)[None, :], len(template.index), axis=0),
                index=template.index,
                columns=template.columns,
            )
        raise ValueError(f"{name} series must align to index or columns")

    raise TypeError(f"{name} must be a scalar, pandas Series, or pandas DataFrame")


def _align_frame(frame: pd.DataFrame, template: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not frame.index.equals(template.index):
        raise ValueError(f"{name} must have matching index")
    if not frame.columns.equals(template.columns):
        raise ValueError(f"{name} must have matching columns")
    if frame.isna().any().any():
        raise ValueError(f"{name} contains missing values")
    return frame.astype(float)


def _prepare_rebalance(rebalance: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if rebalance is None:
        return pd.Series(True, index=index)
    if not isinstance(rebalance, pd.Series):
        raise TypeError("rebalance must be a pandas Series")
    if not rebalance.index.equals(index):
        raise ValueError("rebalance must have matching index")
    if rebalance.isna().any():
        raise ValueError("rebalance contains missing values")
    return rebalance.astype(bool)


def _holdings_from_weights(
    *,
    value: float,
    weights: np.ndarray,
    close: np.ndarray,
    point_value: np.ndarray,
) -> np.ndarray:
    gross_weight = np.abs(weights).sum()
    if gross_weight <= 0:
        raise ValueError("weights must have positive gross exposure")
    return value * weights / (gross_weight * close * point_value)


def _tradeable_volume(instrument_volume: np.ndarray, holdings: np.ndarray) -> float:
    active = np.abs(holdings) > 0
    if not active.any():
        return np.nan
    return float(np.min(instrument_volume[active] / np.abs(holdings[active])))
