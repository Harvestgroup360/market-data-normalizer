"""Time-bar (OHLCV) aggregation.

Turn a stream of normalized trade :class:`MarketEvent` records into fixed
interval OHLCV bars — the bread-and-butter reduction from raw ticks to
candles used everywhere downstream (charts, features, backtests).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence

from .schema import EventType, MarketEvent


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar covering ``[start_ns, start_ns + interval_ns)``."""

    start_ns: int
    interval_ns: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: int
    vwap: Optional[Decimal] = None

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.interval_ns


def time_bars(
    events: Iterable[MarketEvent], interval_ns: int
) -> List[Bar]:
    """Aggregate trade events into fixed-interval OHLCV bars.

    Non-trade events (and trades without a price) are ignored. Events are
    sorted by timestamp, so out-of-order input is handled correctly. Empty
    intervals are not emitted.
    """
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")

    trades = sorted(
        (e for e in events if e.event_type is EventType.TRADE and e.price is not None),
        key=lambda e: e.ts_ns,
    )

    buckets: dict[int, dict] = {}
    order: list[int] = []

    for e in trades:
        start = (e.ts_ns // interval_ns) * interval_ns
        size = e.size if e.size is not None else Decimal(0)
        b = buckets.get(start)
        if b is None:
            buckets[start] = {
                "open": e.price, "high": e.price, "low": e.price,
                "close": e.price, "volume": size,
                "notional": e.price * size, "trades": 1,
            }
            order.append(start)
        else:
            if e.price > b["high"]:
                b["high"] = e.price
            if e.price < b["low"]:
                b["low"] = e.price
            b["close"] = e.price
            b["volume"] += size
            b["notional"] += e.price * size
            b["trades"] += 1

    out: List[Bar] = []
    for start in sorted(order):
        b = buckets[start]
        vwap = (b["notional"] / b["volume"]) if b["volume"] > 0 else None
        out.append(Bar(
            start_ns=start, interval_ns=interval_ns,
            open=b["open"], high=b["high"], low=b["low"], close=b["close"],
            volume=b["volume"], trades=b["trades"], vwap=vwap,
        ))
    return out


def resample_bars(bars: Sequence[Bar], interval_ns: int) -> List[Bar]:
    """Downsample bars to a coarser ``interval_ns`` (e.g. 1-minute -> 5-minute).

    OHLC is aggregated as open=first, high=max, low=min, close=last; volume and
    trade counts are summed; VWAP is recombined volume-weighted (exact, since a
    bar's ``vwap * volume`` is its traded notional). ``interval_ns`` should be a
    multiple of the input bars' interval.
    """
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")

    groups: dict[int, list[Bar]] = {}
    order: list[int] = []
    for b in sorted(bars, key=lambda x: x.start_ns):
        start = (b.start_ns // interval_ns) * interval_ns
        if start not in groups:
            groups[start] = []
            order.append(start)
        groups[start].append(b)

    out: List[Bar] = []
    for start in sorted(order):
        g = groups[start]
        volume = sum((x.volume for x in g), Decimal(0))
        notional = sum(
            (x.vwap * x.volume for x in g if x.vwap is not None), Decimal(0)
        )
        vwap = (notional / volume) if volume > 0 else None
        out.append(Bar(
            start_ns=start, interval_ns=interval_ns,
            open=g[0].open,
            high=max(x.high for x in g),
            low=min(x.low for x in g),
            close=g[-1].close,
            volume=volume,
            trades=sum(x.trades for x in g),
            vwap=vwap,
        ))
    return out
