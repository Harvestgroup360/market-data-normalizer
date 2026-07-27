"""Cross-venue equivalence tests.

The same economic trade, expressed in CSV, WebSocket JSON and FIX, must
normalize to the same MarketEvent.
"""
from decimal import Decimal

import pytest

from mdnorm import (
    EventType,
    Side,
    canonical_symbol,
    from_csv_row,
    from_fix,
    from_ws_json,
)


def test_symbol_canonicalization():
    assert canonical_symbol("BTCUSDT") == "BTC-USDT"
    assert canonical_symbol("xbt/usd") == "BTC-USD"
    assert canonical_symbol("ETH_EUR") == "ETH-EUR"
    with pytest.raises(ValueError):
        canonical_symbol("")


def test_csv_iso_timestamp():
    row = {"symbol": "btc/usd", "ts": "2026-01-02T00:00:00Z",
           "price": "42000.5", "size": "0.25", "side": "buy"}
    ev = from_csv_row(row, venue="coinbase")
    assert ev.symbol == "BTC-USD"
    assert ev.event_type is EventType.TRADE
    assert ev.price == Decimal("42000.5")
    assert ev.size == Decimal("0.25")
    assert ev.side is Side.BUY
    assert ev.ts_ns == 1767312000_000_000_000


def test_ws_json_matches_csv():
    ws = {"s": "BTCUSD", "p": "42000.5", "q": "0.25",
          "T": 1767312000000, "m": False}
    ev = from_ws_json(ws, venue="binance")
    assert ev.symbol == "BTC-USD"
    assert ev.price == Decimal("42000.5")
    assert ev.side is Side.BUY          # buyer aggressor
    assert ev.ts_ns == 1767312000_000_000_000


def test_fix_matches_csv():
    msg = "|".join(["8=FIX.4.4", "35=AE", "55=BTC/USD", "31=42000.5",
                    "32=0.25", "54=1", "60=20260102-00:00:00"])
    ev = from_fix(msg, venue="lmax", sep="|")
    assert ev.symbol == "BTC-USD"
    assert ev.price == Decimal("42000.5")
    assert ev.size == Decimal("0.25")
    assert ev.side is Side.BUY
    assert ev.ts_ns == 1767312000_000_000_000


def test_three_sources_are_equivalent():
    row = {"symbol": "btc/usd", "ts": "2026-01-02T00:00:00Z",
           "price": "42000.5", "size": "0.25", "side": "buy"}
    ws = {"s": "BTCUSD", "p": "42000.5", "q": "0.25",
          "T": 1767312000000, "m": False}
    fix = "|".join(["55=BTC/USD", "31=42000.5", "32=0.25", "54=1",
                    "60=20260102-00:00:00"])

    a = from_csv_row(row, venue="v")
    b = from_ws_json(ws, venue="v")
    c = from_fix(fix, venue="v", sep="|")

    for field in ("symbol", "price", "size", "side", "ts_ns", "event_type"):
        assert getattr(a, field) == getattr(b, field) == getattr(c, field)


def test_trade_requires_price():
    with pytest.raises(ValueError):
        from_fix("55=BTC/USD|32=1", venue="v", sep="|")
