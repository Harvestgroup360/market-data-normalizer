"""Stream merge / dedupe tests."""
from decimal import Decimal

from mdnorm import EventType, MarketEvent, Side
from mdnorm.streams import dedupe, merge_streams


def trade(ts, price, venue="v"):
    return MarketEvent(
        symbol="BTC-USD", venue=venue, event_type=EventType.TRADE,
        ts_ns=ts, price=Decimal(str(price)), size=Decimal("1"), side=Side.BUY,
    )


def test_merge_orders_by_timestamp():
    a = [trade(0, 100), trade(30, 102)]
    b = [trade(10, 101), trade(20, 103)]
    merged = merge_streams(a, b)
    assert [e.ts_ns for e in merged] == [0, 10, 20, 30]


def test_merge_is_stable_for_equal_timestamps():
    a = [trade(5, 100, venue="A")]
    b = [trade(5, 100, venue="B")]
    merged = merge_streams(a, b)
    assert [e.venue for e in merged] == ["A", "B"]  # stream order preserved


def test_dedupe_removes_exact_duplicates():
    e = trade(0, 100)
    out = dedupe([e, e, trade(1, 101), e])
    # only the first occurrence of the duplicate survives
    assert [x.ts_ns for x in out] == [0, 1]


def test_dedupe_first_seen_order():
    out = dedupe([trade(0, 100), trade(1, 101), trade(0, 100)])
    assert [x.ts_ns for x in out] == [0, 1]


def test_merge_then_dedupe_pipeline():
    a = [trade(0, 100), trade(10, 101)]
    b = [trade(0, 100), trade(20, 102)]  # trade(0,100) duplicated across venues... same fields
    merged = dedupe(merge_streams(a, b))
    assert [e.ts_ns for e in merged] == [0, 10, 20]
