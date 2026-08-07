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


def fill_gaps(bars: Sequence[Bar]) -> List[Bar]:
    """Return a gapless bar series, inserting flat bars for missing intervals.

    Backtests and feature pipelines usually want a continuous grid. Any
    interval with no trades is filled with a synthetic bar whose OHLC all equal
    the previous close, with zero volume/trades and no VWAP. Input is sorted by
    time; the grid step is taken from the first bar's ``interval_ns``.
    """
    if not bars:
        return []

    ordered = sorted(bars, key=lambda b: b.start_ns)
    interval = ordered[0].interval_ns
    out: List[Bar] = []
    expected = ordered[0].start_ns
    prev_close: Optional[Decimal] = None

    for b in ordered:
        while prev_close is not None and b.start_ns > expected:
            out.append(Bar(
                start_ns=expected, interval_ns=interval,
                open=prev_close, high=prev_close, low=prev_close,
                close=prev_close, volume=Decimal(0), trades=0, vwap=None,
            ))
            expected += interval
        out.append(b)
        prev_close = b.close
        expected = b.start_ns + interval

    return out


# -- event-driven bars ------------------------------------------------------

def _sorted_trades(events: Iterable[MarketEvent]) -> List[MarketEvent]:
    return sorted(
        (e for e in events
         if e.event_type is EventType.TRADE and e.price is not None),
        key=lambda e: e.ts_ns,
    )


def _close_bar(acc: dict) -> Bar:
    vwap = (acc["notional"] / acc["volume"]) if acc["volume"] > 0 else None
    span = acc["last_ts"] - acc["first_ts"]
    return Bar(
        start_ns=acc["first_ts"], interval_ns=span,
        open=acc["open"], high=acc["high"], low=acc["low"],
        close=acc["close"], volume=acc["volume"],
        trades=acc["trades"], vwap=vwap,
    )


def _event_bars(events, threshold_reached) -> List[Bar]:
    """Shared engine for count/volume/dollar bars.

    Accumulates sorted trades into a bar until ``threshold_reached(acc)``
    is true after adding a trade, then closes it and starts the next one.
    For event-driven bars ``start_ns`` is the first trade's timestamp and
    ``interval_ns`` is the realized span (``end_ns`` = last trade's time).
    A final partial bar is emitted if any trades remain.
    """
    out: List[Bar] = []
    acc: Optional[dict] = None
    for e in _sorted_trades(events):
        size = e.size if e.size is not None else Decimal(0)
        if acc is None:
            acc = {"first_ts": e.ts_ns, "last_ts": e.ts_ns,
                   "open": e.price, "high": e.price, "low": e.price,
                   "close": e.price, "volume": size,
                   "notional": e.price * size, "trades": 1}
        else:
            if e.price > acc["high"]:
                acc["high"] = e.price
            if e.price < acc["low"]:
                acc["low"] = e.price
            acc["close"] = e.price
            acc["last_ts"] = e.ts_ns
            acc["volume"] += size
            acc["notional"] += e.price * size
            acc["trades"] += 1
        if threshold_reached(acc):
            out.append(_close_bar(acc))
            acc = None
    if acc is not None:
        out.append(_close_bar(acc))
    return out


def count_bars(events: Iterable[MarketEvent], every: int) -> List[Bar]:
    """Aggregate trades into bars of exactly ``every`` trades (tick bars).

    The trailing partial bar (fewer than ``every`` trades) is included.
    """
    if every <= 0:
        raise ValueError("every must be positive")
    return _event_bars(events, lambda acc: acc["trades"] >= every)


def volume_bars(events: Iterable[MarketEvent], min_volume: Decimal) -> List[Bar]:
    """Aggregate trades into bars that each hold >= ``min_volume`` base units.

    A bar closes on the trade that pushes cumulative volume to the
    threshold, so bars can slightly overshoot it. The trailing partial bar
    is included.
    """
    if min_volume <= 0:
        raise ValueError("min_volume must be positive")
    return _event_bars(events, lambda acc: acc["volume"] >= min_volume)


def dollar_bars(events: Iterable[MarketEvent], min_notional: Decimal) -> List[Bar]:
    """Aggregate trades into bars of >= ``min_notional`` traded value.

    Notional is ``price * size`` summed per bar; the closing trade may
    overshoot the threshold. The trailing partial bar is included.
    """
    if min_notional <= 0:
        raise ValueError("min_notional must be positive")
    return _event_bars(events, lambda acc: acc["notional"] >= min_notional)
