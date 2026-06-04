"""Normalize Binance trade CSV files for the AFML research pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BINANCE_TRADE_COLUMNS = [
    "id",
    "price",
    "qty",
    "quoteQty",
    "time",
    "isBuyerMaker",
    "isBestMatch",
]

OUTPUT_COLUMNS = [
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


def normalize_binance_trades(
    input_path: str | Path,
    output_path: str | Path,
    *,
    symbol: str | None = None,
    chunksize: int = 1_000_000,
    time_unit: str = "us",
) -> Path:
    """Convert Binance raw trades to the project's tick-data schema.

    The output is a CSV with a ``timestamp`` column that can be loaded with
    ``afml.data.read_tick_csv``. ``qty`` is renamed to ``volume`` and
    ``quoteQty`` is renamed to ``dollar_value``.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_symbol = symbol or _infer_symbol(input_path)

    wrote_header = False
    for chunk in pd.read_csv(
        input_path,
        header=None,
        names=BINANCE_TRADE_COLUMNS,
        chunksize=chunksize,
    ):
        normalized = _normalize_chunk(
            chunk,
            symbol=inferred_symbol,
            time_unit=time_unit,
        )
        normalized.to_csv(
            output_path,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

    return output_path


def _normalize_chunk(chunk: pd.DataFrame, *, symbol: str, time_unit: str) -> pd.DataFrame:
    is_buyer_maker = _as_bool(chunk["isBuyerMaker"])
    is_best_match = _as_bool(chunk["isBestMatch"])
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(chunk["time"], unit=time_unit, utc=True),
            "symbol": symbol,
            "trade_id": pd.to_numeric(chunk["id"], downcast="integer"),
            "price": pd.to_numeric(chunk["price"]),
            "volume": pd.to_numeric(chunk["qty"]),
            "dollar_value": pd.to_numeric(chunk["quoteQty"]),
            "side": is_buyer_maker.map({True: -1, False: 1}),
            "is_buyer_maker": is_buyer_maker,
            "is_best_match": is_best_match,
        }
    )
    return normalized.loc[:, OUTPUT_COLUMNS]


def _as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def _infer_symbol(path: Path) -> str:
    stem = path.stem
    if "-trades-" in stem:
        return stem.split("-trades-", maxsplit=1)[0]
    return path.parent.name


def _default_output_path(input_path: Path) -> Path:
    symbol = _infer_symbol(input_path)
    return (
        Path("data")
        / "processed"
        / "binance"
        / "trades"
        / symbol
        / f"{input_path.stem}-normalized.csv"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw Binance trades CSV path.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Normalized output CSV path. Defaults under data/processed/.",
    )
    parser.add_argument("--symbol", default=None, help="Symbol override.")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--time-unit",
        default="us",
        help="Pandas timestamp unit for Binance time field. Default: us.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or _default_output_path(args.input)
    written = normalize_binance_trades(
        args.input,
        output,
        symbol=args.symbol,
        chunksize=args.chunksize,
        time_unit=args.time_unit,
    )
    print(written)


if __name__ == "__main__":
    main()
