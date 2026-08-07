"""Count/volume/dollar (event-driven) bar tests."""
from decimal import Decimal

import pytest

from mdnorm import (
    EventType,
    MarketEvent,
    Pipeline,
    Side,
    count_bars,
    dollar_bars,
    volume_bars,
)
from mdnorm.cli import main


def _trade(ts_ns, price, size="1"):
    return MarketEvent(
        symbol="BTC-USD", venue="x", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal(size), side=Side.BUY,
    )


EVENTS = [
    _trade(1_000, "100", "1"),
    _trade(2_000, "102", "2"),
    _trade(3_000, "101", "1"),
    _trade(4_000, "103", "3"),
    _trade(5_000, "104", "1"),
]


def test_count_bars_exact_and_partial():
    bars = count_bars(EVENTS, every=2)
    assert [b.trades for b in bars] == [2, 2, 1]
    b0 = bars[0]
    assert (b0.open, b0.high, b0.low, b0.close) == (
        Decimal("100"), Decimal("102"), Decimal("100"), Decimal("102"))
    assert b0.start_ns == 1_000 and b0.end_ns == 2_000  # realized span
    # vwap = (100*1 + 102*2) / 3
    assert b0.vwap == Decimal("304") / Decimal("3")


def test_count_bars_ignores_quotes_and_sorts():
    quote = MarketEvent(
        symbol="BTC-USD", venue="x", event_type=EventType.QUOTE, ts_ns=1_500,
        bid_price=Decimal("99"), ask_price=Decimal("101"),
    )
    shuffled = [EVENTS[2], quote, EVENTS[0], EVENTS[1]]
    bars = count_bars(shuffled, every=3)
    assert len(bars) == 1
    assert bars[0].open == Decimal("100") and bars[0].close == Decimal("101")


def test_volume_bars_threshold_and_overshoot():
    bars = volume_bars(EVENTS, min_volume=Decimal("3"))
    # cum sizes: 1,3 -> close | 1,4 -> close (overshoot) | 1 -> partial
    assert [b.volume for b in bars] == [Decimal("3"), Decimal("4"), Decimal("1")]
    assert bars[1].high == Decimal("103")


def test_dollar_bars_threshold():
    bars = dollar_bars(EVENTS, min_notional=Decimal("300"))
    # notional: 100, 304 -> close | 101, 410 -> close | 104 -> partial
    assert len(bars) == 3
    assert bars[0].trades == 2 and bars[1].trades == 2 and bars[2].trades == 1


@pytest.mark.parametrize("fn,bad", [
    (count_bars, 0), (volume_bars, Decimal("0")), (dollar_bars, Decimal("-1")),
])
def test_rejects_non_positive_thresholds(fn, bad):
    with pytest.raises(ValueError):
        fn(EVENTS, bad)


def test_empty_input():
    assert count_bars([], 5) == []


def test_pipeline_event_bar_steps():
    bars = Pipeline().dedupe().count_bars(2).run(EVENTS)
    assert [b.trades for b in bars] == [2, 2, 1]
    pipe = Pipeline().volume_bars(Decimal("3"))
    assert pipe.steps == ["volume_bars"]


def test_cli_every_trades(tmp_path, capsys):
    src = tmp_path / "t.csv"
    src.write_text(
        "symbol,ts,price,size,side\n"
        + "".join(f"BTCUSD,2026-08-07T00:00:0{i}Z,10{i}.0,1,buy\n"
                  for i in range(5))
    )
    out = tmp_path / "bars.csv"
    rc = main(["bars", str(src), "--every-trades", "2", "-o", str(out)])
    assert rc == 0
    assert "wrote 3 bar(s)" in capsys.readouterr().out


def test_cli_interval_and_event_flags_are_exclusive(tmp_path, capsys):
    src = tmp_path / "t.csv"
    src.write_text("symbol,ts,price,size,side\nBTCUSD,2026-08-07T00:00:01Z,100.0,1,buy\n")
    with pytest.raises(SystemExit):
        main(["bars", str(src), "--interval", "1m",
              "--every-trades", "10", "-o", str(tmp_path / "o.csv")])
