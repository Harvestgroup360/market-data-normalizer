"""Pipeline composition tests."""
from decimal import Decimal

from mdnorm import Bar, EventType, MarketEvent, Pipeline, Side

MIN_NS = 60_000_000_000


def _trade(ts_ns, price, size="1"):
    return MarketEvent(
        symbol="BTC-USD", venue="binance", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal(size), side=Side.BUY,
    )


def test_full_chain_dedupe_clean_bars_fill():
    events = [
        _trade(0, "100"),
        _trade(0, "100"),            # exact duplicate -> dedupe
        _trade(10 * 10**9, "1000"),  # 10x outlier -> clean
        _trade(30 * 10**9, "101"),
        _trade(2 * MIN_NS, "102"),   # minute 2 (minute 1 empty -> fill_gaps)
    ]
    pipe = (
        Pipeline()
        .dedupe()
        .clean(max_return=Decimal("0.5"))
        .time_bars(MIN_NS)
        .fill_gaps()
    )
    bars = pipe.run(events)

    assert [type(b) for b in bars] == [Bar, Bar, Bar]
    assert bars[0].open == Decimal("100") and bars[0].close == Decimal("101")
    assert bars[0].trades == 2                     # dup and outlier are gone
    assert bars[1].volume == 0                     # flat fill for empty minute
    assert bars[1].close == bars[0].close
    assert any(i.kind == "outlier" for i in pipe.last_issues)


def test_resample_after_time_bars():
    events = [_trade(i * MIN_NS, "100") for i in range(10)]
    bars = Pipeline().time_bars(MIN_NS).resample(5 * MIN_NS).run(events)
    assert len(bars) == 2
    assert bars[0].interval_ns == 5 * MIN_NS


def test_steps_report_order():
    pipe = Pipeline().dedupe().clean().time_bars(MIN_NS).fill_gaps()
    assert pipe.steps == ["dedupe", "clean", "time_bars", "fill_gaps"]


def test_custom_apply_step():
    events = [_trade(0, "100"), _trade(1, "101"), _trade(2, "102")]
    pipe = Pipeline().apply(
        "drop_first", lambda data: data[1:]
    ).time_bars(MIN_NS)
    bars = pipe.run(events)
    assert bars[0].open == Decimal("101") and bars[0].trades == 2
    assert pipe.steps == ["drop_first", "time_bars"]


def test_pipeline_is_reusable():
    pipe = Pipeline().time_bars(MIN_NS)
    a = pipe.run([_trade(0, "100")])
    b = pipe.run([_trade(0, "200"), _trade(1, "201")])
    assert len(a) == 1 and len(b) == 1
    assert a[0].open == Decimal("100") and b[0].close == Decimal("201")


def test_empty_pipeline_passes_through():
    events = [_trade(0, "100")]
    assert Pipeline().run(events) == events
