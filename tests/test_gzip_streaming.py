"""Gzip transparency and streaming-reader tests."""
import gzip
import json
import types
from decimal import Decimal

from mdnorm import (
    EventType,
    MarketEvent,
    Side,
    iter_csv_trades,
    iter_jsonl_events,
    read_csv_trades,
    read_jsonl_events,
    write_jsonl,
    write_records_csv,
)
from mdnorm.cli import main

CSV_TEXT = (
    "symbol,ts,price,size,side\n"
    "BTCUSD,2026-08-05T00:00:01Z,100.0,1,buy\n"
    "BTCUSD,2026-08-05T00:00:30Z,101.0,2,sell\n"
)


def _trade(ts_ns, price):
    return MarketEvent(
        symbol="BTC-USD", venue="x", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal("1"), side=Side.BUY,
    )


def test_read_csv_gz(tmp_path):
    path = tmp_path / "trades.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(CSV_TEXT)
    events = read_csv_trades(str(path), venue="binance")
    assert len(events) == 2
    assert events[0].price == Decimal("100.0")


def test_write_csv_gz_round_trip(tmp_path):
    events = [_trade(1_000, "100.5"), _trade(2_000, "101.5")]
    path = tmp_path / "out.csv.gz"
    assert write_records_csv(events, str(path)) == 2
    with gzip.open(path, "rt", encoding="utf-8") as f:
        content = f.read()
    assert "100.5" in content and content.startswith("symbol")


def test_jsonl_gz_round_trip(tmp_path):
    events = [_trade(1_000, "100.123456789")]
    path = tmp_path / "events.jsonl.gz"
    assert write_jsonl(events, str(path)) == 1
    back = read_jsonl_events(str(path))
    assert back == events


def test_iter_csv_trades_is_lazy(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text(CSV_TEXT)
    it = iter_csv_trades(str(path), venue="x")
    assert isinstance(it, types.GeneratorType)
    first = next(it)
    assert first.symbol == "BTC-USD"
    assert len(list(it)) == 1  # ровно один оставшийся


def test_iter_jsonl_events_is_lazy(tmp_path):
    path = tmp_path / "e.jsonl"
    row = json.dumps({"symbol": "BTC-USD", "venue": "x",
                      "event_type": "trade", "ts_ns": 1, "price": "100"})
    path.write_text(row + "\n" + row + "\n")
    it = iter_jsonl_events(str(path))
    assert isinstance(it, types.GeneratorType)
    assert len(list(it)) == 2


def test_cli_gz_end_to_end(tmp_path, capsys):
    src = tmp_path / "trades.csv.gz"
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write(CSV_TEXT)
    out = tmp_path / "bars.jsonl.gz"
    rc = main(["bars", str(src), "--interval", "1m", "-o", str(out)])
    assert rc == 0
    with gzip.open(out, "rt", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1
    assert lines[0]["open"] == "100.0" and lines[0]["close"] == "101.0"
