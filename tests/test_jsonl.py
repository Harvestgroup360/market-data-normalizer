"""NDJSON round-trip tests."""
import json
from decimal import Decimal

import pytest

from mdnorm import (
    EventType,
    MarketEvent,
    Side,
    event_from_dict,
    read_jsonl_events,
    time_bars,
    write_jsonl,
)

MIN_NS = 60_000_000_000


def _trade(ts_ns, price, size="1", side=Side.BUY):
    return MarketEvent(
        symbol="BTC-USD", venue="binance", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal(size), side=side,
    )


def _quote(ts_ns):
    return MarketEvent(
        symbol="BTC-USD", venue="kraken", event_type=EventType.QUOTE,
        ts_ns=ts_ns,
        bid_price=Decimal("99.5"), bid_size=Decimal("2"),
        ask_price=Decimal("100.5"), ask_size=Decimal("3"),
    )


def test_events_round_trip(tmp_path):
    events = [_trade(1_000, "100.25"), _quote(2_000),
              _trade(3_000, "101.5", side=Side.SELL)]
    path = str(tmp_path / "events.jsonl")

    assert write_jsonl(events, path) == 3
    back = read_jsonl_events(path)

    assert back == events  # frozen dataclasses compare by value


def test_round_trip_preserves_decimal_precision(tmp_path):
    e = _trade(1_000, "100.123456789")
    path = str(tmp_path / "one.jsonl")
    write_jsonl([e], path)
    (back,) = read_jsonl_events(path)
    assert back.price == Decimal("100.123456789")


def test_write_bars_as_jsonl(tmp_path):
    events = [_trade(0, "100"), _trade(30 * 10**9, "102"),
              _trade(MIN_NS, "101")]
    bars = time_bars(events, MIN_NS)
    path = str(tmp_path / "bars.jsonl")

    assert write_jsonl(bars, path) == 2
    lines = [json.loads(l) for l in open(path) if l.strip()]
    assert lines[0]["open"] == "100" and lines[0]["close"] == "102"
    assert lines[1]["interval_ns"] == MIN_NS


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "sparse.jsonl"
    row = json.dumps({"symbol": "BTC-USD", "venue": "x",
                      "event_type": "trade", "ts_ns": 1, "price": "100"})
    path.write_text(row + "\n\n" + row + "\n")
    assert len(read_jsonl_events(str(path))) == 2


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"symbol": "BTC-USD"\n')
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl_events(str(path))


def test_missing_field_raises(tmp_path):
    path = tmp_path / "short.jsonl"
    path.write_text(json.dumps({"symbol": "BTC-USD", "venue": "x"}) + "\n")
    with pytest.raises(ValueError, match="missing required field"):
        read_jsonl_events(str(path))


def test_event_from_dict_accepts_numeric_values():
    e = event_from_dict({
        "symbol": "ETH-USD", "venue": "x", "event_type": "trade",
        "ts_ns": 5, "price": 100.5, "size": 2,
    })
    assert e.price == Decimal("100.5") and e.size == Decimal("2")
