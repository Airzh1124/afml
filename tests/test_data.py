"""Tests for market data normalization and validation."""

from pathlib import Path

import pandas as pd
import pytest

from afml.data import normalize_tick_data, read_tick_csv, validate_tick_data


SP_CSV = Path(__file__).resolve().parents[1] / "data" / "SP.csv"


def _raw_sp_like_ticks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["01/03/2000", "01/03/2000", "01/03/2000"],
            "time": ["08:30:37.000", "08:30:34.000", "08:30:36.000"],
            "price": [1495.50, 1496.40, 1496.00],
            "volume": [0, 0, 0],
        }
    )


def test_normalize_tick_data_combines_date_and_time_columns() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks())

    assert isinstance(ticks.index, pd.DatetimeIndex)
    assert ticks.index.name == "timestamp"
    assert list(ticks.index) == pd.to_datetime(
        [
            "2000-01-03 08:30:34.000",
            "2000-01-03 08:30:36.000",
            "2000-01-03 08:30:37.000",
        ]
    ).tolist()
    assert "date" not in ticks.columns
    assert "time" not in ticks.columns
    assert list(ticks["price"]) == [1496.40, 1496.00, 1495.50]
    assert list(ticks["volume"]) == [0, 0, 0]
    assert list(ticks["dollar_value"]) == [0.0, 0.0, 0.0]


def test_normalize_tick_data_accepts_existing_datetime_index() -> None:
    raw = _raw_sp_like_ticks()
    raw.index = pd.to_datetime(raw.pop("date") + " " + raw.pop("time"))

    ticks = normalize_tick_data(raw)

    assert ticks.index.name == "timestamp"
    assert list(ticks.columns) == ["price", "volume", "dollar_value"]


def test_read_tick_csv_can_load_sp_sample_when_available() -> None:
    if not SP_CSV.exists():
        pytest.skip("local SP.csv is not available")

    ticks = read_tick_csv(SP_CSV, nrows=5)

    assert len(ticks) == 5
    assert isinstance(ticks.index, pd.DatetimeIndex)
    assert ticks.index.name == "timestamp"
    assert list(ticks.columns) == ["price", "volume", "dollar_value"]
    assert ticks.index[0] == pd.Timestamp("2000-01-03 08:30:34")
    assert ticks["price"].iloc[0] == 1496.40
    assert ticks["volume"].iloc[0] == 0


def test_validate_tick_data_allows_zero_volume() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks())

    validate_tick_data(ticks)


def test_validate_tick_data_allows_duplicate_timestamps_by_default() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks())
    duplicated = pd.concat([ticks, ticks.iloc[[0]]]).sort_index(kind="mergesort")

    validate_tick_data(duplicated)


def test_validate_tick_data_can_reject_duplicate_timestamps() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks())
    duplicated = pd.concat([ticks, ticks.iloc[[0]]]).sort_index(kind="mergesort")

    with pytest.raises(ValueError, match="duplicate timestamps"):
        validate_tick_data(duplicated, allow_duplicate_timestamps=False)


def test_validate_tick_data_rejects_negative_volume() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks(), validate=False)
    ticks.loc[ticks.index[0], "volume"] = -1

    with pytest.raises(ValueError, match="volume must be non-negative"):
        validate_tick_data(ticks)


def test_validate_tick_data_rejects_non_positive_price() -> None:
    ticks = normalize_tick_data(_raw_sp_like_ticks(), validate=False)
    ticks.loc[ticks.index[0], "price"] = 0

    with pytest.raises(ValueError, match="price must be positive"):
        validate_tick_data(ticks)


def test_normalize_tick_data_requires_time_axis() -> None:
    raw = pd.DataFrame({"price": [1496.40], "volume": [0]})

    with pytest.raises(TypeError, match="timestamp columns or a pandas DatetimeIndex"):
        normalize_tick_data(raw)
