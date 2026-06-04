"""Tests for Binance trade normalization script."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_binance_trades import normalize_binance_trades  # noqa: E402


def test_normalize_binance_trades_outputs_framework_schema(tmp_path: Path) -> None:
    raw = tmp_path / "BTCUSDT-trades-2026-06-03.csv"
    raw.write_text(
        "\n".join(
            [
                "1,66760.84000000,0.00007000,4.67325880,1780444800127434,False,True",
                "2,66760.83000000,0.00042000,28.03954860,1780444800134779,True,True",
            ]
        )
    )
    output = tmp_path / "normalized.csv"

    normalize_binance_trades(raw, output, chunksize=1)

    normalized = pd.read_csv(output)
    assert list(normalized.columns) == [
        "timestamp",
        "symbol",
        "trade_id",
        "price",
        "volume",
        "dollar_value",
        "side",
        "is_buyer_maker",
        "is_best_match",
    ]
    assert list(normalized["symbol"]) == ["BTCUSDT", "BTCUSDT"]
    assert list(normalized["price"]) == [66760.84, 66760.83]
    assert list(normalized["volume"]) == [0.00007, 0.00042]
    assert list(normalized["dollar_value"]) == [4.6732588, 28.0395486]
    assert list(normalized["side"]) == [1, -1]
    assert pd.to_datetime(normalized["timestamp"]).iloc[0] == pd.Timestamp(
        "2026-06-03 00:00:00.127434+0000"
    )


def test_normalize_binance_trades_can_limit_rows(tmp_path: Path) -> None:
    raw = tmp_path / "BTCUSDT-trades-2026-06.csv"
    raw.write_text(
        "\n".join(
            [
                "1,100.0,1.0,100.0,1780444800000000,False,True",
                "2,101.0,1.0,101.0,1780444801000000,True,True",
                "3,102.0,1.0,102.0,1780444802000000,False,True",
            ]
        )
    )
    output = tmp_path / "normalized.csv"

    normalize_binance_trades(raw, output, chunksize=2, max_rows=2)

    normalized = pd.read_csv(output)
    assert list(normalized["trade_id"]) == [1, 2]
