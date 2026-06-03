"""Tests for bar construction."""

import pandas as pd
import pytest

from afml.bars import dollar_bars, tick_bars, time_bars, volume_bars


def _ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 11.0, 9.0, 12.0, 13.0],
            "volume": [100, 200, 100, 50, 50],
        },
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:30:10",
                "2024-01-01 09:30:20",
                "2024-01-01 09:31:00",
                "2024-01-01 09:31:10",
            ],
            name="timestamp",
        ),
    )


def test_tick_bars_use_fixed_tick_count() -> None:
    bars = tick_bars(_ticks(), threshold=2)

    assert list(bars.index) == pd.to_datetime(
        [
            "2024-01-01 09:30:10",
            "2024-01-01 09:31:00",
            "2024-01-01 09:31:10",
        ]
    ).tolist()
    assert bars.index.name == "timestamp"
    assert list(bars["open"]) == [10.0, 9.0, 13.0]
    assert list(bars["high"]) == [11.0, 12.0, 13.0]
    assert list(bars["low"]) == [10.0, 9.0, 13.0]
    assert list(bars["close"]) == [11.0, 12.0, 13.0]
    assert list(bars["volume"]) == [300, 150, 50]
    assert list(bars["dollar_value"]) == [3200.0, 1500.0, 650.0]
    assert list(bars["tick_count"]) == [2, 2, 1]


def test_tick_bars_can_drop_partial_final_bar() -> None:
    bars = tick_bars(_ticks(), threshold=2, include_partial=False)

    assert len(bars) == 2
    assert list(bars["tick_count"]) == [2, 2]


def test_volume_bars_use_cumulative_volume() -> None:
    bars = volume_bars(_ticks(), threshold=250)

    assert list(bars.index) == pd.to_datetime(
        ["2024-01-01 09:30:10", "2024-01-01 09:31:10"]
    ).tolist()
    assert list(bars["volume"]) == [300, 200]
    assert list(bars["tick_count"]) == [2, 3]
    assert list(bars["close"]) == [11.0, 13.0]


def test_dollar_bars_use_cumulative_dollar_value() -> None:
    bars = dollar_bars(_ticks(), threshold=3000)

    assert list(bars.index) == pd.to_datetime(
        ["2024-01-01 09:30:10", "2024-01-01 09:31:10"]
    ).tolist()
    assert list(bars["dollar_value"]) == [3200.0, 2150.0]
    assert list(bars["tick_count"]) == [2, 3]
    assert list(bars["close"]) == [11.0, 13.0]


def test_time_bars_resample_by_frequency() -> None:
    bars = time_bars(_ticks(), freq="1min")

    assert list(bars.index) == pd.to_datetime(
        ["2024-01-01 09:30:00", "2024-01-01 09:31:00"]
    ).tolist()
    assert bars.index.name == "timestamp"
    assert list(bars["open"]) == [10.0, 12.0]
    assert list(bars["high"]) == [11.0, 13.0]
    assert list(bars["low"]) == [9.0, 12.0]
    assert list(bars["close"]) == [9.0, 13.0]
    assert list(bars["volume"]) == [400, 100]
    assert list(bars["dollar_value"]) == [4100.0, 1250.0]
    assert list(bars["tick_count"]) == [3, 2]


def test_empty_input_returns_empty_canonical_bars() -> None:
    ticks = pd.DataFrame(
        {"price": pd.Series(dtype=float), "volume": pd.Series(dtype=float)},
        index=pd.DatetimeIndex([], name="timestamp"),
    )

    bars = dollar_bars(ticks, threshold=1000)

    assert bars.empty
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert bars.index.name == "timestamp"
    assert list(bars.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_value",
        "tick_count",
    ]


def test_threshold_must_be_positive() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        tick_bars(_ticks(), threshold=0)


def test_missing_required_columns_raise_error() -> None:
    ticks = pd.DataFrame(
        {"price": [10.0]},
        index=pd.DatetimeIndex(["2024-01-01"], name="timestamp"),
    )

    with pytest.raises(ValueError, match="missing required columns"):
        dollar_bars(ticks, threshold=100)


def test_ticks_must_have_datetime_index() -> None:
    ticks = pd.DataFrame({"price": [10.0], "volume": [100]})

    with pytest.raises(TypeError, match="DatetimeIndex"):
        dollar_bars(ticks, threshold=100)
