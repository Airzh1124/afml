"""Tests for event-based sampling."""

import pandas as pd
import pytest

from afml.sampling import cusum_filter


def _prices() -> pd.Series:
    return pd.Series(
        [100.0, 100.5, 101.0, 100.2, 99.5, 100.4, 101.1],
        index=pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:31:00",
                "2024-01-01 09:32:00",
                "2024-01-01 09:33:00",
                "2024-01-01 09:34:00",
                "2024-01-01 09:35:00",
                "2024-01-01 09:36:00",
            ],
            name="timestamp",
        ),
    )


def test_cusum_filter_matches_symmetric_cusum_logic() -> None:
    events = cusum_filter(_prices(), threshold=1.0)

    assert list(events) == pd.to_datetime(
        [
            "2024-01-01 09:34:00",
            "2024-01-01 09:36:00",
        ]
    ).tolist()
    assert events.name == "timestamp"


def test_cusum_filter_supports_dynamic_threshold_series() -> None:
    threshold = pd.Series(
        [0.9, 2.0],
        index=pd.DatetimeIndex(["2024-01-01 09:30:00", "2024-01-01 09:33:00"]),
    )

    events = cusum_filter(_prices(), threshold=threshold)

    assert list(events) == [pd.Timestamp("2024-01-01 09:32:00")]


def test_cusum_filter_handles_irregular_dates_by_observation_order() -> None:
    prices = pd.Series(
        [100.0, 101.1, 101.2],
        index=pd.DatetimeIndex(
            ["2024-01-01 09:30:00", "2024-01-03 09:30:00", "2024-01-03 09:31:00"]
        ),
    )

    events = cusum_filter(prices, threshold=1.0)

    assert list(events) == [pd.Timestamp("2024-01-03 09:30:00")]


def test_cusum_filter_can_reset_across_large_time_gaps() -> None:
    prices = pd.Series(
        [100.0, 101.1, 102.2],
        index=pd.DatetimeIndex(
            ["2024-01-01 09:30:00", "2024-01-03 09:30:00", "2024-01-03 09:31:00"]
        ),
    )

    events = cusum_filter(prices, threshold=1.0, max_gap="1h")

    assert list(events) == [pd.Timestamp("2024-01-03 09:31:00")]


def test_cusum_filter_supports_simple_returns() -> None:
    prices = pd.Series(
        [100.0, 101.0, 102.01],
        index=pd.date_range("2024-01-01", periods=3, freq="min"),
    )

    events = cusum_filter(prices, threshold=0.015, use_returns=True)

    assert list(events) == [pd.Timestamp("2024-01-01 00:02:00")]


def test_cusum_filter_validates_inputs() -> None:
    with pytest.raises(TypeError, match="pandas Series"):
        cusum_filter(pd.DataFrame({"price": [1, 2]}), threshold=1.0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="DatetimeIndex"):
        cusum_filter(pd.Series([1.0, 2.0]), threshold=1.0)

    with pytest.raises(ValueError, match="threshold must be positive"):
        cusum_filter(_prices(), threshold=0.0)

    with pytest.raises(ValueError, match="log_returns=True requires"):
        cusum_filter(_prices(), threshold=0.01, log_returns=True)
