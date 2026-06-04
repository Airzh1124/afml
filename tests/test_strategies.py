"""Tests for primary strategy signals."""

import pandas as pd
import pytest

from afml.strategies import moving_average_cross_side


def test_moving_average_cross_side_tracks_fast_slow_cross() -> None:
    close = pd.Series(
        [10.0, 9.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        index=pd.date_range("2024-01-01", periods=7, freq="D"),
    )

    side = moving_average_cross_side(close, fast_window=2, slow_window=3)

    assert side.iloc[0] == -1.0
    assert side.iloc[-1] == 1.0
    assert side.name == "side"


def test_moving_average_cross_side_forward_fills_ties() -> None:
    close = pd.Series(
        [10.0, 9.0, 8.0, 9.0, 10.0, 10.0],
        index=pd.date_range("2024-01-01", periods=6, freq="D"),
    )

    side = moving_average_cross_side(close, fast_window=1, slow_window=2)

    assert side.loc[pd.Timestamp("2024-01-04")] == 1.0
    assert side.loc[pd.Timestamp("2024-01-06")] == 1.0


def test_moving_average_cross_side_validates_inputs() -> None:
    close = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))

    with pytest.raises(ValueError, match="smaller"):
        moving_average_cross_side(close, fast_window=3, slow_window=2)

    with pytest.raises(ValueError, match="positive"):
        moving_average_cross_side(close, fast_window=0, slow_window=2)
