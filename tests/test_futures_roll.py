"""Tests for single future roll utilities."""

import pandas as pd
import pytest

from afml.multi_product import roll_futures_series, roll_gaps


def _futures_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "FUT_CUR_GEN_TICKER": ["ESM4", "ESM4", "ESU4", "ESU4", "ESZ4"],
            "PX_OPEN": [100.0, 101.0, 110.0, 111.0, 120.0],
            "PX_LAST": [101.0, 102.0, 111.0, 112.0, 121.0],
            "VWAP": [100.5, 101.5, 110.5, 111.5, 120.5],
        },
        index=pd.DatetimeIndex(
            [
                "2024-06-17",
                "2024-06-18",
                "2024-06-19",
                "2024-06-20",
                "2024-06-21",
            ],
            name="timestamp",
        ),
    )


def test_roll_gaps_forward_matches_start_of_raw_series() -> None:
    gaps = roll_gaps(_futures_bars(), match_end=False)

    assert list(gaps) == [0.0, 0.0, 8.0, 8.0, 16.0]
    assert gaps.name == "roll_gap"


def test_roll_gaps_backward_matches_end_of_raw_series() -> None:
    gaps = roll_gaps(_futures_bars(), match_end=True)

    assert list(gaps) == [-16.0, -16.0, -8.0, -8.0, 0.0]


def test_roll_futures_series_subtracts_gaps_from_price_columns() -> None:
    adjusted = roll_futures_series(_futures_bars(), match_end=False)

    assert list(adjusted["PX_LAST"]) == [101.0, 102.0, 103.0, 104.0, 105.0]
    assert list(adjusted["VWAP"]) == [100.5, 101.5, 102.5, 103.5, 104.5]
    assert list(adjusted["roll_gap"]) == [0.0, 0.0, 8.0, 8.0, 16.0]


def test_roll_futures_series_supports_custom_fields() -> None:
    bars = _futures_bars().rename(
        columns={
            "FUT_CUR_GEN_TICKER": "instrument",
            "PX_OPEN": "open",
            "PX_LAST": "close",
        }
    )

    adjusted = roll_futures_series(
        bars,
        price_columns=("close",),
        field_map={
            "instrument": "instrument",
            "open": "open",
            "close": "close",
        },
        match_end=False,
    )

    assert list(adjusted["close"]) == [101.0, 102.0, 103.0, 104.0, 105.0]


def test_roll_gaps_return_zero_when_there_is_no_roll() -> None:
    bars = _futures_bars()
    bars["FUT_CUR_GEN_TICKER"] = "ESM4"

    gaps = roll_gaps(bars)

    assert list(gaps) == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_roll_futures_series_validates_inputs() -> None:
    bars = _futures_bars()

    with pytest.raises(TypeError, match="DataFrame"):
        roll_gaps(pd.Series([1, 2]))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="DatetimeIndex"):
        roll_gaps(bars.reset_index(drop=True))

    with pytest.raises(ValueError, match="missing price columns"):
        roll_futures_series(bars, price_columns=("missing",))
