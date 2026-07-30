"""Data-quality check tests."""
from decimal import Decimal

from mdnorm import EventType, MarketEvent, Side
from mdnorm.quality import clean, find_issues


def trade(ts_ns, price, size=1):
    return MarketEvent(
        symbol="BTC-USD", venue="v", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(str(price)), size=Decimal(str(size)),
        side=Side.BUY,
    )


def test_detects_price_outlier():
    events = [trade(0, 100), trade(1, 100.5), trade(2, 200)]  # +100% jump
    issues = find_issues(events, max_return=Decimal("0.1"))
    kinds = [(i.kind, i.index) for i in issues]
    assert ("outlier", 2) in kinds
    assert all(i.index != 1 for i in issues)  # 0.5% move is fine


def test_detects_out_of_order_and_gap():
    events = [trade(0, 100), trade(100, 100), trade(50, 100)]
    issues = find_issues(events)
    assert any(i.kind == "out_of_order" and i.index == 2 for i in issues)

    gap_issues = find_issues([trade(0, 100), trade(1_000, 100)], max_gap_ns=500)
    assert any(i.kind == "gap" for i in gap_issues)


def test_detects_non_positive():
    events = [trade(0, 100), trade(1, -5)]
    issues = find_issues(events)
    assert any(i.kind == "non_positive" and i.index == 1 for i in issues)


def test_clean_drops_bad_events_keeps_good():
    events = [trade(0, 100), trade(1, 100.5), trade(2, 200), trade(3, 100.7)]
    cleaned, issues = clean(events, max_return=Decimal("0.1"))
    prices = [e.price for e in cleaned]
    assert Decimal("200") not in prices
    assert len(cleaned) == 3
    assert any(i.kind == "outlier" for i in issues)


def test_clean_empty():
    cleaned, issues = clean([])
    assert cleaned == [] and issues == []
