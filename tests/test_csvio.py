"""CSV file I/O tests."""
import csv
import os
import tempfile
from decimal import Decimal

from mdnorm import EventType, MarketEvent, Side
from mdnorm.csvio import read_csv_trades, write_records_csv


def trade(ts, price, size="1"):
    return MarketEvent(
        symbol="BTC-USD", venue="v", event_type=EventType.TRADE,
        ts_ns=ts, price=Decimal(price), size=Decimal(size), side=Side.BUY,
    )


def test_write_then_read_roundtrip():
    events = [trade(0, "100.5", "2"), trade(1_000_000_000, "101", "1")]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        assert write_records_csv(events, p) == 2
        back = read_csv_trades(p, venue="v", mapping={"ts": "ts_ns"}, ts_unit="ns")
    assert [e.ts_ns for e in back] == [0, 1_000_000_000]
    assert back[0].price == Decimal("100.5")
    assert back[0].size == Decimal("2")
    assert back[0].side is Side.BUY


def test_read_iso_timestamp_csv():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "iso.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "ts", "price", "size", "side"])
            w.writerow(["btc/usd", "2026-01-02T00:00:00Z", "42000", "0.5", "buy"])
        evs = read_csv_trades(p, venue="coinbase")
    assert evs[0].symbol == "BTC-USD"
    assert evs[0].ts_ns == 1767312000_000_000_000
    assert evs[0].price == Decimal("42000")


def test_write_has_header_and_rows():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.csv")
        write_records_csv([trade(0, "100")], p)
        with open(p, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    assert rows[0]["symbol"] == "BTC-USD"
    assert rows[0]["price"] == "100"


def test_write_empty_returns_zero():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "e.csv")
        assert write_records_csv([], p) == 0
        assert os.path.getsize(p) == 0
