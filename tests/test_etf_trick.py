"""Tests for the ETF trick."""

import numpy as np
import pandas as pd
import pytest

from afml.multi_product import etf_trick


def _market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.DatetimeIndex(
        [
            "2024-01-01 09:30:00",
            "2024-01-01 09:31:00",
            "2024-01-01 09:32:00",
        ],
        name="timestamp",
    )
    columns = ["leg_a", "leg_b"]
    open_ = pd.DataFrame(
        [[100.0, 50.0], [101.0, 49.5], [103.0, 48.0]],
        index=index,
        columns=columns,
    )
    close = pd.DataFrame(
        [[100.0, 50.0], [102.0, 49.0], [101.0, 51.0]],
        index=index,
        columns=columns,
    )
    weights = pd.DataFrame(
        [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]],
        index=index,
        columns=columns,
    )
    volume = pd.DataFrame(
        [[100.0, 100.0], [50.0, 30.0], [60.0, 40.0]],
        index=index,
        columns=columns,
    )
    return open_, close, weights, volume


def test_etf_trick_tracks_one_dollar_spread_without_costs() -> None:
    open_, close, weights, _ = _market()

    result, holdings = etf_trick(
        open_,
        close,
        weights,
        rebalance=pd.Series(False, index=close.index),
        return_holdings=True,
    )

    assert list(result["value"]) == pytest.approx([1.0, 1.02, 0.995])
    assert list(result["gross_pnl"]) == pytest.approx([0.0, 0.02, -0.025])
    assert list(result["rebalance_cost"]) == pytest.approx([0.0, 0.0, 0.0])
    assert list(result["net_pnl"]) == pytest.approx([0.0, 0.02, -0.025])
    assert list(holdings["leg_a"]) == pytest.approx([0.005, 0.005, 0.005])
    assert list(holdings["leg_b"]) == pytest.approx([-0.01, -0.01, -0.01])


def test_etf_trick_charges_rebalance_and_reports_trading_constraints() -> None:
    open_, close, weights, volume = _market()
    rebalance = pd.Series([False, True, False], index=close.index)

    result, holdings = etf_trick(
        open_,
        close,
        weights,
        volume=volume,
        transaction_cost=0.001,
        rebalance=rebalance,
        return_holdings=True,
    )

    expected_cost = (
        abs(0.005) * 102.0
        + abs(1.02 / (2 * 102.0)) * 103.0
        + abs(-0.01) * 49.0
        + abs(-1.02 / (2 * 49.0)) * 48.0
    ) * 0.001
    expected_value_1 = 1.02 - expected_cost
    expected_h1_a = expected_value_1 / (2 * 102.0)
    expected_h1_b = -expected_value_1 / (2 * 49.0)
    expected_gross_pnl_2 = expected_h1_a * -1.0 + expected_h1_b * 2.0

    assert result["rebalance_cost"].iloc[1] == pytest.approx(expected_cost)
    assert result["value"].iloc[1] == pytest.approx(expected_value_1)
    assert holdings["leg_a"].iloc[1] == pytest.approx(expected_h1_a)
    assert holdings["leg_b"].iloc[1] == pytest.approx(expected_h1_b)
    assert result["gross_pnl"].iloc[2] == pytest.approx(expected_gross_pnl_2)
    assert result["value"].iloc[2] == pytest.approx(expected_value_1 + expected_gross_pnl_2)
    assert result["bid_ask_cost"].iloc[1] == pytest.approx(
        (abs(0.005) * 102.0 + abs(-0.01) * 49.0) * 0.001
    )
    assert result["volume"].iloc[1] == pytest.approx(
        min(50.0 / abs(0.005), 30.0 / abs(-0.01))
    )


def test_etf_trick_accepts_column_series_point_values() -> None:
    open_, close, weights, _ = _market()
    point_value = pd.Series({"leg_a": 50.0, "leg_b": 25.0})

    result, holdings = etf_trick(
        open_,
        close,
        weights,
        point_value=point_value,
        rebalance=pd.Series(False, index=close.index),
        return_holdings=True,
    )

    assert result["value"].iloc[1] == pytest.approx(1.02)
    assert holdings["leg_a"].iloc[0] == pytest.approx(1.0 / (2 * 100.0 * 50.0))
    assert holdings["leg_b"].iloc[0] == pytest.approx(-1.0 / (2 * 50.0 * 25.0))


def test_etf_trick_validates_inputs() -> None:
    open_, close, weights, _ = _market()

    with pytest.raises(ValueError, match="initial_value"):
        etf_trick(open_, close, weights, initial_value=0)

    bad_weights = weights.copy()
    bad_weights.iloc[0] = 0.0
    with pytest.raises(ValueError, match="positive gross exposure"):
        etf_trick(open_, close, bad_weights)

    bad_open = open_.iloc[:-1]
    with pytest.raises(ValueError, match="matching indexes"):
        etf_trick(bad_open, close, weights)


def test_tradeable_volume_ignores_zero_weight_legs() -> None:
    open_, close, weights, volume = _market()
    weights["leg_b"] = 0.0

    result = etf_trick(
        open_,
        close,
        weights,
        volume=volume,
        rebalance=pd.Series(False, index=close.index),
    )

    assert np.isfinite(result["volume"].iloc[1])
    assert result["volume"].iloc[1] == pytest.approx(50.0 / 0.01)
