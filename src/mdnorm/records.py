"""Flatten events and bars into plain dicts.

The last mile of any pipeline is getting normalized objects into a DataFrame,
a CSV writer, or a JSON payload. These helpers turn ``MarketEvent`` and ``Bar``
into flat, JSON-serialisable dicts. ``Decimal`` values are emitted as strings by
default (lossless); pass ``as_float=True`` for numeric convenience.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, List, Optional, Union

from .bars import Bar
from .schema import MarketEvent


def _num(v: Optional[Decimal], as_float: bool) -> Any:
    if v is None:
        return None
    return float(v) if as_float else str(v)


def event_to_dict(e: MarketEvent, *, as_float: bool = False) -> dict:
    """Flatten a :class:`MarketEvent` into a plain dict."""
    return {
        "symbol": e.symbol,
        "venue": e.venue,
        "event_type": e.event_type.value,
        "ts_ns": e.ts_ns,
        "price": _num(e.price, as_float),
        "size": _num(e.size, as_float),
        "side": e.side.value if e.side is not None else None,
        "bid_price": _num(e.bid_price, as_float),
        "bid_size": _num(e.bid_size, as_float),
        "ask_price": _num(e.ask_price, as_float),
        "ask_size": _num(e.ask_size, as_float),
    }


def bar_to_dict(b: Bar, *, as_float: bool = False) -> dict:
    """Flatten a :class:`Bar` into a plain dict."""
    return {
        "start_ns": b.start_ns,
        "end_ns": b.end_ns,
        "interval_ns": b.interval_ns,
        "open": _num(b.open, as_float),
        "high": _num(b.high, as_float),
        "low": _num(b.low, as_float),
        "close": _num(b.close, as_float),
        "volume": _num(b.volume, as_float),
        "trades": b.trades,
        "vwap": _num(b.vwap, as_float),
    }


def to_records(
    items: Iterable[Union[MarketEvent, Bar]], *, as_float: bool = False
) -> List[dict]:
    """Convert a sequence of events and/or bars to a list of flat dicts.

    Handy for ``pandas.DataFrame(to_records(...))``, ``csv.DictWriter`` or
    ``json.dumps``.
    """
    out: List[dict] = []
    for it in items:
        if isinstance(it, Bar):
            out.append(bar_to_dict(it, as_float=as_float))
        elif isinstance(it, MarketEvent):
            out.append(event_to_dict(it, as_float=as_float))
        else:
            raise TypeError(f"unsupported item type: {type(it).__name__}")
    return out
