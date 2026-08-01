"""Serialization / records tests."""
import json
from decimal import Decimal

import pytest

from mdnorm import Bar, EventType, MarketEvent, Side
from mdnorm.records import bar_to_dict, event_to_dict, to_records


def trade(ts=0, price="100.5", size="2"):
    return MarketEvent(
        symbol="BTC-USD", venue="v", event_type=EventType.TRADE,
        ts_ns=ts, price=Decimal(price), size=Decimal(size), side=Side.BUY,
    )


def a_bar():
    c = Decimal("101")
    return Bar(start_ns=0, interval_ns=1_000, open=c, high=c, low=c,
               close=c, volume=Decimal("5"), trades=3, vwap=c)


def test_event_to_dict_lossless_strings():
    d = event_to_dict(trade())
    assert d["symbol"] == "BTC-USD"
    assert d["event_type"] == "trade"
    assert d["price"] == "100.5"  # Decimal preserved as string
    assert d["side"] == "buy"
    assert d["bid_price"] is None


def test_event_to_dict_as_float():
    d = event_to_dict(trade(price="100.5"), as_float=True)
    assert d["price"] == 100.5
    assert isinstance(d["price"], float)


def test_bar_to_dict_includes_end_ns():
    d = bar_to_dict(a_bar())
    assert d["start_ns"] == 0 and d["end_ns"] == 1_000
    assert d["close"] == "101" and d["trades"] == 3


def test_to_records_dispatches_by_type_and_is_json_safe():
    recs = to_records([trade(), a_bar()])
    assert len(recs) == 2
    assert "price" in recs[0] and "open" in recs[1]
    json.dumps(recs)  # must not raise


def test_to_records_rejects_unknown():
    with pytest.raises(TypeError):
        to_records([object()])
