"""OHLCV time-bar aggregation tests."""
from decimal import Decimal

import pytest

from mdnorm import EventType, MarketEvent, Side, time_bars


def trade(ts_ns, price, size):
    return MarketEvent(
        symbol="BTC-USD", venue="v", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(str(price)), size=Decimal(str(size)),
        side=Side.BUY,
    )


def test_single_interval_ohlcv():
    interval = 1_000_000_000  # 1s in ns
    events = [
        trade(0, 100, 1),
        trade(100, 105, 2),
        trade(200, 98, 1),
        trade(300, 102, 1),
    ]
    bars = time_bars(events, interval)
    assert len(bars) == 1
    b = bars[0]
    assert b.open == Decimal("100")
    assert b.high == Decimal("105")
    assert b.low == Decimal("98")
    assert b.close == Decimal("102")
    assert b.volume == Decimal("5")
    assert b.trades == 4
    # vwap = (100*1 + 105*2 + 98*1 + 102*1) / 5 = 510/5 = 102
    assert b.vwap == Decimal("102")


def test_splits_into_intervals():
    interval = 1_000  # ns
    bars = time_bars([trade(0, 100, 1), trade(1_500, 110, 2)], interval)
    assert len(bars) == 2
    assert bars[0].start_ns == 0
    assert bars[1].start_ns == 1_000
    assert bars[1].end_ns == 2_000


def test_out_of_order_input_sorted():
    interval = 1_000_000_000
    bars = time_bars([trade(300, 102, 1), trade(0, 100, 1)], interval)
    assert bars[0].open == Decimal("100")
    assert bars[0].close == Decimal("102")


def test_non_trade_events_ignored():
    quote = MarketEvent(
        symbol="BTC-USD", venue="v", event_type=EventType.QUOTE,
        ts_ns=0, bid_price=Decimal("99"), ask_price=Decimal("101"),
    )
    bars = time_bars([quote, trade(0, 100, 1)], 1_000_000_000)
    assert len(bars) == 1
    assert bars[0].trades == 1


def test_invalid_interval():
    with pytest.raises(ValueError):
        time_bars([], 0)
