"""Quote (bid/ask) normalization tests."""
from decimal import Decimal

from mdnorm import EventType, from_csv_quote, from_ws_quote


def test_ws_quote_bookticker():
    msg = {"s": "BTCUSDT", "b": "41999.5", "B": "1.2",
           "a": "42000.5", "A": "0.8", "T": 1767312000000}
    ev = from_ws_quote(msg, venue="binance")
    assert ev.event_type is EventType.QUOTE
    assert ev.symbol == "BTC-USDT"
    assert ev.bid_price == Decimal("41999.5")
    assert ev.ask_price == Decimal("42000.5")
    assert ev.ts_ns == 1767312000_000_000_000


def test_ws_quote_without_timestamp():
    ev = from_ws_quote({"s": "ETHUSD", "b": "2500", "a": "2501"}, venue="v")
    assert ev.ts_ns == 0
    assert ev.bid_size is None and ev.ask_size is None


def test_csv_quote_matches_ws():
    row = {"symbol": "btc/usdt", "ts": "2026-01-02T00:00:00Z",
           "bid": "41999.5", "bid_size": "1.2",
           "ask": "42000.5", "ask_size": "0.8"}
    a = from_csv_quote(row, venue="v")
    b = from_ws_quote({"s": "BTCUSDT", "b": "41999.5", "B": "1.2",
                       "a": "42000.5", "A": "0.8", "T": 1767312000000}, venue="v")
    for field in ("symbol", "bid_price", "ask_price", "bid_size",
                  "ask_size", "ts_ns", "event_type"):
        assert getattr(a, field) == getattr(b, field)


def test_mid_and_spread():
    ev = from_ws_quote({"s": "BTCUSDT", "b": "100", "a": "102"}, venue="v")
    assert ev.mid_price == Decimal("101")
    assert ev.spread == Decimal("2")


def test_mid_spread_none_when_one_sided():
    ev = from_ws_quote({"s": "BTCUSDT", "b": "100"}, venue="v")
    assert ev.mid_price is None
    assert ev.spread is None
