"""Gap-filling tests for bar series."""
from decimal import Decimal

from mdnorm import Bar
from mdnorm.bars import fill_gaps


def bar(start, close, vol=5, trades=2):
    c = Decimal(str(close))
    return Bar(
        start_ns=start, interval_ns=1_000,
        open=c, high=c, low=c, close=c,
        volume=Decimal(str(vol)), trades=trades, vwap=c,
    )


def test_fills_single_gap_with_flat_bar():
    out = fill_gaps([bar(0, 100), bar(2_000, 110)])
    assert [b.start_ns for b in out] == [0, 1_000, 2_000]
    gap = out[1]
    assert gap.open == gap.high == gap.low == gap.close == Decimal("100")
    assert gap.volume == Decimal("0")
    assert gap.trades == 0
    assert gap.vwap is None


def test_no_gaps_unchanged():
    inp = [bar(0, 100), bar(1_000, 101), bar(2_000, 102)]
    out = fill_gaps(inp)
    assert [b.start_ns for b in out] == [0, 1_000, 2_000]
    assert out[1].volume == Decimal("5")  # real bar, not synthetic


def test_multiple_missing_intervals():
    out = fill_gaps([bar(0, 100), bar(4_000, 120)])
    assert [b.start_ns for b in out] == [0, 1_000, 2_000, 3_000, 4_000]
    assert all(out[i].close == Decimal("100") for i in (1, 2, 3))


def test_sorts_input():
    out = fill_gaps([bar(2_000, 110), bar(0, 100)])
    assert out[0].start_ns == 0 and out[-1].start_ns == 2_000


def test_empty_and_single():
    assert fill_gaps([]) == []
    assert len(fill_gaps([bar(0, 100)])) == 1
