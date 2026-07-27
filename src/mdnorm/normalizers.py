"""Venue-specific normalizers.

Each function takes one raw record and returns a :class:`MarketEvent`.
They are intentionally small and pure so they are trivial to test and reuse.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from .schema import EventType, MarketEvent, Side
from .symbols import canonical_symbol
from .timeutil import epoch_to_ns, fix_utc_to_ns, iso_to_ns

__all__ = ["from_csv_row", "from_ws_json", "from_fix"]


def _side(value: str | None) -> Side | None:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("buy", "b", "bid", "1"):
        return Side.BUY
    if v in ("sell", "s", "ask", "2"):
        return Side.SELL
    return None


def from_csv_row(
    row: Mapping[str, str],
    *,
    venue: str,
    mapping: Mapping[str, str] | None = None,
    ts_unit: str | None = None,
) -> MarketEvent:
    """Normalize one CSV row (a dict of column -> value).

    ``mapping`` renames columns to the canonical fields
    ``symbol, ts, price, size, side``. If ``ts_unit`` is given the timestamp
    is read as an epoch number in that unit, otherwise it is parsed as ISO-8601.
    """
    m = {"symbol": "symbol", "ts": "ts", "price": "price",
         "size": "size", "side": "side"}
    if mapping:
        m.update(mapping)

    ts_raw = row[m["ts"]]
    ts_ns = epoch_to_ns(float(ts_raw), ts_unit) if ts_unit else iso_to_ns(ts_raw)

    return MarketEvent(
        symbol=canonical_symbol(row[m["symbol"]]),
        venue=venue,
        event_type=EventType.TRADE,
        ts_ns=ts_ns,
        price=Decimal(str(row[m["price"]])),
        size=Decimal(str(row[m["size"]])) if row.get(m["size"]) else None,
        side=_side(row.get(m["side"])),
    )


def from_ws_json(
    msg: Mapping[str, Any],
    *,
    venue: str,
    mapping: Mapping[str, str] | None = None,
    ts_unit: str = "ms",
) -> MarketEvent:
    """Normalize an exchange WebSocket trade message.

    Defaults match the common ``{s, p, q, T, m}`` trade shape used by several
    large venues (symbol, price, qty, event-time-ms, is-buyer-maker).
    ``mapping`` overrides the source keys.
    """
    m = {"symbol": "s", "price": "p", "size": "q", "ts": "T", "maker": "m"}
    if mapping:
        m.update(mapping)

    # ``is_buyer_maker == True`` means the aggressor sold into the bid.
    side = None
    if m["maker"] in msg:
        side = Side.SELL if msg[m["maker"]] else Side.BUY
    elif m.get("side") in msg:
        side = _side(msg[m["side"]])

    return MarketEvent(
        symbol=canonical_symbol(str(msg[m["symbol"]])),
        venue=venue,
        event_type=EventType.TRADE,
        ts_ns=epoch_to_ns(msg[m["ts"]], ts_unit),
        price=Decimal(str(msg[m["price"]])),
        size=Decimal(str(msg[m["size"]])),
        side=side,
    )


def _parse_fix(message: str, sep: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in message.split(sep):
        if "=" in part:
            tag, _, val = part.partition("=")
            fields[tag] = val
    return fields


def from_fix(message: str, *, venue: str, sep: str = "\x01") -> MarketEvent:
    """Normalize a FIX execution/trade message (tag=value, SOH-delimited).

    Uses tags 55 (Symbol), 31 (LastPx), 32 (LastQty), 54 (Side),
    60 (TransactTime). ``sep`` defaults to the FIX SOH byte; pass ``"|"``
    for pipe-delimited test strings.
    """
    f = _parse_fix(message, sep)
    if "55" not in f or "31" not in f:
        raise ValueError("FIX message missing Symbol(55) or LastPx(31)")

    return MarketEvent(
        symbol=canonical_symbol(f["55"]),
        venue=venue,
        event_type=EventType.TRADE,
        ts_ns=fix_utc_to_ns(f["60"]) if "60" in f else 0,
        price=Decimal(f["31"]),
        size=Decimal(f["32"]) if "32" in f else None,
        side=_side(f.get("54")),
    )
