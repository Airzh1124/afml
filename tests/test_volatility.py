"""Tests for dynamic volatility target estimates."""

import pandas as pd
import pytest

from afml.sampling import daily_volatility, target_from_events


def _close() -> pd.Series:
    return pd.Series(
        [100.0, 101.0, 102.0, 104.0, 103.0, 106.0],
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 10:30:00",
                "2024-01-02 09:30:00",
                "2024-01-02 10:30:00",
                "2024-01-03 09:30:00",
                "2024-01-03 10:30:00",
            ],
            name="timestamp",
        ),
    )


def test_daily_volatility_matches_afml_one_day_return_logic() -> None:
    close = _close()

    vol = daily_volatility(close, span=2)

    daily_returns = pd.Series(
        [
            104.0 / 100.0 - 1.0,
            103.0 / 101.0 - 1.0,
            106.0 / 102.0 - 1.0,
        ],
        index=close.index[3:],
    )
    expected = daily_returns.ewm(span=2).std()
    expected.name = "daily_volatility"

    pd.testing.assert_series_equal(vol, expected)


def test_daily_volatility_returns_empty_when_series_is_shorter_than_one_day() -> None:
    close = pd.Series(
        [100.0, 101.0],
        index=pd.date_range("2024-01-01", periods=2, freq="h"),
    )

    vol = daily_volatility(close)

    assert vol.empty
    assert list(vol.index) == []
    assert vol.name == "daily_volatility"


def test_daily_volatility_handles_timezone_aware_index() -> None:
    close = pd.Series(
        [100.0, 101.0, 102.0, 104.0],
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00+00:00",
                "2024-01-01 10:30:00+00:00",
                "2024-01-02 09:30:00+00:00",
                "2024-01-02 10:30:00+00:00",
            ],
            name="timestamp",
        ),
    )

    vol = daily_volatility(close, span=2)

    assert list(vol.index) == [pd.Timestamp("2024-01-02 10:30:00+00:00")]
    assert vol.name == "daily_volatility"


def test_target_from_events_aligns_and_filters_dynamic_target() -> None:
    target = pd.Series(
        [0.01, 0.03, 0.02],
        index=pd.DatetimeIndex(
            ["2024-01-01 09:30:00", "2024-01-01 09:32:00", "2024-01-01 09:34:00"]
        ),
    )
    events = pd.DatetimeIndex(
        [
            "2024-01-01 09:31:00",
            "2024-01-01 09:33:00",
            "2024-01-01 09:35:00",
        ]
    )

    aligned = target_from_events(target, events, min_ret=0.015)

    assert list(aligned.index) == pd.to_datetime(
        ["2024-01-01 09:33:00", "2024-01-01 09:35:00"]
    ).tolist()
    assert list(aligned) == [0.03, 0.02]
    assert aligned.name == "trgt"


def test_target_from_events_drops_events_before_target_starts() -> None:
    target = pd.Series(
        [0.02],
        index=pd.DatetimeIndex(["2024-01-01 09:32:00"]),
    )
    events = pd.DatetimeIndex(["2024-01-01 09:31:00", "2024-01-01 09:33:00"])

    aligned = target_from_events(target, events)

    assert list(aligned.index) == [pd.Timestamp("2024-01-01 09:33:00")]
    assert list(aligned) == [0.02]


def test_volatility_validates_inputs() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        daily_volatility(pd.DataFrame({"close": [1.0, 2.0]}))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="DatetimeIndex"):
        daily_volatility(pd.Series([1.0, 2.0]))

    with pytest.raises(ValueError, match="span must be positive"):
        daily_volatility(_close(), span=0)

    with pytest.raises(ValueError, match="min_ret"):
        target_from_events(daily_volatility(_close()), _close().index, min_ret=-1)
