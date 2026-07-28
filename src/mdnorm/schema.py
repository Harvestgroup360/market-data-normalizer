"""Unified market-data schema.

All venue-specific feeds are normalized into a single, exchange-agnostic
representation so downstream research and execution code never has to care
where a tick came from.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    TRADE = "trade"
    QUOTE = "quote"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A single normalized market-data event.

    Prices and sizes use ``Decimal`` to avoid binary float rounding on money.
    Timestamps are integer nanoseconds since the Unix epoch, in UTC.
    """

    symbol: str            # canonical symbol, e.g. "BTC-USD"
    venue: str             # source venue, e.g. "binance"
    event_type: EventType
    ts_ns: int             # nanoseconds since epoch (UTC)

    # Trade fields
    price: Optional[Decimal] = None
    size: Optional[Decimal] = None
    side: Optional[Side] = None

    # Quote fields
    bid_price: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if self.event_type is EventType.TRADE and self.price is None:
            raise ValueError("trade events require a price")
        if self.event_type is EventType.QUOTE and (
            self.bid_price is None and self.ask_price is None
        ):
            raise ValueError("quote events require at least one side")

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Mid price of a quote, or ``None`` if either side is missing."""
        if self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2
        return None

    @property
    def spread(self) -> Optional[Decimal]:
        """Absolute bid/ask spread, or ``None`` if either side is missing."""
        if self.bid_price is not None and self.ask_price is not None:
            return self.ask_price - self.bid_price
        return None
