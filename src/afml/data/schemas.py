"""Canonical market data schemas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TickSchema:
    """Column names for normalized tick-like market data.

    Normalized tick data should use a UTC timezone-aware pandas
    ``DatetimeIndex`` as the market time axis. The required columns are
    ``price`` and ``volume``; optional columns are preserved when present.
    """

    price: str = "price"
    volume: str = "volume"
    timestamp: str = "timestamp"
    date: str = "date"
    time: str = "time"
    symbol: str = "symbol"
    side: str = "side"
    bid: str = "bid"
    ask: str = "ask"
    dollar_value: str = "dollar_value"

    @property
    def required_columns(self) -> tuple[str, str]:
        return (self.price, self.volume)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        return (
            self.symbol,
            self.side,
            self.bid,
            self.ask,
            self.dollar_value,
        )


DEFAULT_TICK_SCHEMA = TickSchema()
