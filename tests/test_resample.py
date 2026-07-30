"""Bar resampling (downsampling) tests."""
from decimal import Decimal

import pytest

from mdnorm import Bar
from mdnorm.bars import resample_bars


def bar(start, o, h, l, c, vol, trades, vwap):
    return Bar(
        start_ns=start, interval_ns=1_000,
        open=Decimal(str(o)), high=Decimal(str(h)), low=Decimal(str(l)),
        close=Decimal(str(c)), volume=Decimal(str(vol)), trades=trades,
        vwap=Decimal(str(vwap)),
    )


def test_resample_merges_two_bars():
    b1 = bar(0, 100, 105, 99, 102, 10, 3, 101)
    b2 = bar(1_000, 102, 108, 101, 107, 20, 5, 104)
    out = resample_bars([b1, b2], 2_000)
    assert len(out) == 1
    r = out[0]
    assert r.open == Decimal("100")
    assert r.high == Decimal("108")
    assert r.low == Decimal("99")
    assert r.close == Decimal("107")
    assert r.volume == Decimal("30")
    assert r.trades == 8
    # vwap = (101*10 + 104*20) / 30 = 3090/30 = 103
    assert r.vwap == Decimal("103")
    assert r.interval_ns == 2_000


def test_resample_keeps_separate_groups():
    b1 = bar(0, 100, 101, 99, 100, 5, 1, 100)
    b2 = bar(5_000, 200, 201, 199, 200, 5, 1, 200)
    out = resample_bars([b1, b2], 2_000)
    assert len(out) == 2
    assert [b.start_ns for b in out] == [0, 4_000]


def test_resample_sorts_input():
    b1 = bar(0, 100, 100, 100, 100, 1, 1, 100)
    b2 = bar(1_000, 110, 110, 110, 110, 1, 1, 110)
    out = resample_bars([b2, b1], 2_000)  # reversed
    assert out[0].open == Decimal("100")
    assert out[0].close == Decimal("110")


def test_resample_invalid_interval():
    with pytest.raises(ValueError):
        resample_bars([], 0)
