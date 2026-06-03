"""Tests for information-driven imbalance bars."""

import pandas as pd
import pytest

from afml.bars import tick_imbalance_bars, tick_rule


def _imbalance_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 100.0, 100.0, 101.0],
            "volume": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:30:01",
                "2024-01-01 09:30:02",
                "2024-01-01 09:30:03",
                "2024-01-01 09:30:04",
                "2024-01-01 09:30:05",
                "2024-01-01 09:30:06",
                "2024-01-01 09:30:07",
                "2024-01-01 09:30:08",
            ],
            name="timestamp",
        ),
    )


def test_tick_rule_inherits_previous_direction_on_flat_prices() -> None:
    prices = pd.Series(
        [100.0, 101.0, 101.0, 100.5, 100.5, 101.0],
        index=pd.date_range("2024-01-01", periods=6, freq="s"),
    )

    signs = tick_rule(prices, initial_direction=1)

    assert list(signs) == [1, 1, 1, -1, -1, 1]
    assert signs.name == "tick_rule"


def test_tick_imbalance_bars_use_dynamic_thresholds() -> None:
    bars = tick_imbalance_bars(
        _imbalance_ticks(),
        expected_ticks_init=4,
        expected_imbalance_init=0.5,
        expected_ticks_window=1,
        expected_imbalance_window=1,
    )

    assert list(bars.index) == pd.to_datetime(
        [
            "2024-01-01 09:30:01",
            "2024-01-01 09:30:05",
            "2024-01-01 09:30:07",
        ]
    ).tolist()
    assert bars.index.name == "timestamp"
    assert list(bars["open"]) == [100.0, 102.0, 100.0]
    assert list(bars["high"]) == [101.0, 102.0, 100.0]
    assert list(bars["low"]) == [100.0, 99.0, 100.0]
    assert list(bars["close"]) == [101.0, 99.0, 100.0]
    assert list(bars["volume"]) == [3, 18, 15]
    assert list(bars["dollar_value"]) == [302.0, 1804.0, 1500.0]
    assert list(bars["tick_count"]) == [2, 4, 2]
    assert list(bars["theta"]) == [2, -2, 2]
    assert list(bars["threshold"]) == [2.0, 2.0, 2.0]
    assert list(bars["expected_ticks"]) == [4.0, 2.0, 4.0]
    assert list(bars["expected_imbalance"]) == [0.5, 1.0, -0.5]


def test_tick_imbalance_bars_can_include_partial_final_bar() -> None:
    bars = tick_imbalance_bars(
        _imbalance_ticks(),
        expected_ticks_init=4,
        expected_imbalance_init=0.5,
        expected_ticks_window=1,
        expected_imbalance_window=1,
        include_partial=True,
    )

    assert len(bars) == 4
    assert bars.index[-1] == pd.Timestamp("2024-01-01 09:30:08")
    assert bars["tick_count"].iloc[-1] == 1
    assert bars["theta"].iloc[-1] == 1


def test_tick_imbalance_bars_use_min_expected_imbalance_floor() -> None:
    bars = tick_imbalance_bars(
        _imbalance_ticks(),
        expected_ticks_init=10,
        expected_imbalance_init=0.0,
        min_expected_imbalance=0.2,
    )

    assert bars.index[0] == pd.Timestamp("2024-01-01 09:30:01")
    assert bars["threshold"].iloc[0] == 2.0
    assert bars["theta"].iloc[0] == 2


def test_tick_imbalance_bars_validate_parameters() -> None:
    with pytest.raises(ValueError, match="expected_ticks_init must be positive"):
        tick_imbalance_bars(
            _imbalance_ticks(),
            expected_ticks_init=0,
            expected_imbalance_init=0.5,
        )

    with pytest.raises(ValueError, match="expected_imbalance_init"):
        tick_imbalance_bars(
            _imbalance_ticks(),
            expected_ticks_init=10,
            expected_imbalance_init=2.0,
        )


def test_tick_rule_validates_initial_direction() -> None:
    with pytest.raises(ValueError, match="initial_direction"):
        tick_rule(pd.Series([1.0, 2.0]), initial_direction=0)
