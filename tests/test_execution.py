"""Execution benchmark tests: VWAP, TWAP, slippage, shortfall, participation."""
from decimal import Decimal

import pytest

from mdnorm import (
    EventType,
    Fill,
    MarketEvent,
    Side,
    average_fill_price,
    evaluate,
    exclude_fills,
    implementation_shortfall_bps,
    participation_rate,
    slippage_bps,
    twap,
    vwap,
)

D = Decimal
SEC = 1_000_000_000


def t(ts, price, size="10"):
    return MarketEvent(symbol="BTC-USD", venue="x", event_type=EventType.TRADE,
                       ts_ns=ts, price=D(price), size=D(size))


def f(ts, price, size="10", side=Side.BUY):
    return Fill(ts_ns=ts, price=D(price), size=D(size), side=side)


# -- VWAP -------------------------------------------------------------------

def test_vwap_weights_by_size():
    # 100*1 + 200*3 = 700 over 4 units
    assert vwap([t(1, "100", "1"), t(2, "200", "3")]) == D("175")


def test_vwap_window_is_half_open():
    trades = [t(1, "100"), t(5, "200"), t(9, "300")]
    assert vwap(trades, start_ns=5, end_ns=9) == D("200")


def test_vwap_is_none_without_volume():
    assert vwap([]) is None
    assert vwap([t(1, "100", "0")]) is None


def test_vwap_ignores_sizeless_trades():
    sizeless = MarketEvent(symbol="BTC-USD", venue="x", event_type=EventType.TRADE,
                           ts_ns=1, price=D("999"))
    assert vwap([sizeless, t(2, "100", "1")]) == D("100")


def test_vwap_ignores_quotes():
    q = MarketEvent(symbol="BTC-USD", venue="x", event_type=EventType.QUOTE,
                    ts_ns=1, bid_price=D("1"), ask_price=D("2"))
    assert vwap([q, t(2, "100", "1")]) == D("100")


# -- TWAP -------------------------------------------------------------------

def test_twap_averages_one_price_per_interval():
    trades = [t(0, "100"), t(1, "110"), t(SEC, "200")]
    # bucket 0 closes at 110, bucket 1 at 200
    assert twap(trades, interval_ns=SEC) == D("155")


def test_twap_skips_silent_buckets_instead_of_inventing_prices():
    trades = [t(0, "100"), t(2 * SEC, "200")]
    # the middle second never traded; it is not carried forward
    assert twap(trades, interval_ns=SEC) == D("150")


def test_twap_is_none_on_an_empty_window():
    assert twap([t(1, "100")], interval_ns=SEC, start_ns=10 * SEC) is None


def test_twap_rejects_a_non_positive_interval():
    with pytest.raises(ValueError):
        twap([t(1, "100")], interval_ns=0)


# -- removing your own prints ------------------------------------------------

def test_your_own_print_is_removed_from_the_tape():
    tape = [t(1, "100", "10"), t(2, "101", "5"), t(3, "102", "10")]
    out = exclude_fills(tape, [f(2, "101", "5")])
    assert [e.ts_ns for e in out] == [1, 3]


def test_each_fill_removes_at_most_one_print():
    tape = [t(1, "100", "10"), t(2, "100", "10"), t(3, "100", "10")]
    out = exclude_fills(tape, [f(1, "100", "10")])
    assert len(out) == 2


def test_a_near_miss_needs_a_tolerance():
    tape = [t(1_000, "100", "10")]
    assert len(exclude_fills(tape, [f(1_050, "100", "10")])) == 1
    assert len(exclude_fills(tape, [f(1_050, "100", "10")], tolerance_ns=100)) == 0


def test_a_different_size_is_not_your_print():
    tape = [t(1, "100", "10")]
    assert len(exclude_fills(tape, [f(1, "100", "7")])) == 1


def test_exclude_fills_does_not_mutate_the_input():
    tape = [t(1, "100", "10")]
    exclude_fills(tape, [f(1, "100", "10")])
    assert len(tape) == 1


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError):
        exclude_fills([], [], tolerance_ns=-1)


def test_excluding_your_prints_changes_the_benchmark():
    """The whole reason the function exists."""
    tape = [t(1, "100", "10"), t(2, "110", "90")]      # our 90 at 110
    mine = [f(2, "110", "90", Side.BUY)]
    assert vwap(tape) == D("109")                       # flattered
    assert vwap(exclude_fills(tape, mine)) == D("100")  # the real market


# -- participation -----------------------------------------------------------

def test_participation_is_our_share_of_total_volume():
    tape = [t(1, "100", "25"), t(2, "100", "75")]
    assert participation_rate([f(1, "100", "25")], tape) == D("0.25")


def test_participation_is_none_without_market_volume():
    assert participation_rate([f(1, "100", "1")], []) is None


def test_participation_respects_the_window():
    tape = [t(1, "100", "50"), t(9, "100", "50")]
    assert participation_rate([f(1, "100", "50")], tape, start_ns=0, end_ns=5) == 1


# -- scoring -----------------------------------------------------------------

def test_average_fill_price_is_size_weighted():
    assert average_fill_price([f(1, "100", "1"), f(2, "200", "3")]) == D("175")


def test_average_fill_price_of_nothing():
    assert average_fill_price([]) is None


def test_buying_below_the_benchmark_scores_positive():
    assert slippage_bps([f(1, "99", "1", Side.BUY)], D("100")) == D("100")


def test_buying_above_the_benchmark_scores_negative():
    assert slippage_bps([f(1, "101", "1", Side.BUY)], D("100")) == D("-100")


def test_selling_above_the_benchmark_scores_positive():
    assert slippage_bps([f(1, "101", "1", Side.SELL)], D("100")) == D("100")


def test_selling_below_the_benchmark_scores_negative():
    assert slippage_bps([f(1, "99", "1", Side.SELL)], D("100")) == D("-100")


def test_mixed_sides_are_refused_rather_than_netted():
    fills = [f(1, "100", "1", Side.BUY), f(2, "100", "1", Side.SELL)]
    with pytest.raises(ValueError, match="same side"):
        slippage_bps(fills, D("100"))


def test_slippage_of_nothing_is_none():
    assert slippage_bps([], D("100")) is None


def test_a_non_positive_benchmark_is_rejected():
    with pytest.raises(ValueError):
        slippage_bps([f(1, "100", "1")], D("0"))


def test_shortfall_uses_the_decision_price():
    assert implementation_shortfall_bps([f(1, "102", "1", Side.BUY)], D("100")) == D("-200")


# -- validation --------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"price": D("0")}, {"size": D("0")}, {"size": D("-1")}, {"ts_ns": -1},
])
def test_fills_are_validated(kwargs):
    args = {"ts_ns": 1, "price": D("100"), "size": D("1"), "side": Side.BUY}
    args.update(kwargs)
    with pytest.raises(ValueError):
        Fill(**args)


# -- the summary -------------------------------------------------------------

def test_evaluate_reports_the_numbers_together():
    tape = [t(1, "100", "10"), t(2, "101", "5"), t(3, "102", "10"), t(4, "103", "5")]
    mine = [f(2, "101", "5", Side.BUY), f(4, "103", "5", Side.BUY)]
    r = evaluate(mine, tape, decision_price=D("100"))
    assert r.side is Side.BUY
    assert r.filled_size == D("10") and r.average_price == D("102")
    assert r.own_prints_removed == 2
    # window is [2, 5): market volume there is 5 + 10 + 5 = 20, ours is 10
    assert r.participation_rate == D("0.5")
    assert r.vwap == D("102")             # only the 102 print is left after exclusion
    assert r.shortfall_bps < 0            # bought above the decision price


def test_a_single_fill_leaves_nothing_to_benchmark_against():
    """A degenerate but real case, pinned so it cannot surprise anyone.

    The window spans one fill, the only print in it is that fill, and once
    your own print is removed there is no market left. The summary says so
    with a null VWAP and 100% participation rather than inventing a score.
    """
    r = evaluate([f(2, "101", "5", Side.BUY)],
                 [t(1, "100", "10"), t(2, "101", "5"), t(3, "102", "10")])
    assert r.vwap is None and r.slippage_vs_vwap_bps is None
    assert r.participation_rate == 1


def test_evaluate_excludes_own_prints_by_default():
    tape = [t(1, "100", "10"), t(1, "110", "90")]
    mine = [f(1, "110", "90", Side.BUY)]
    with_exclusion = evaluate(mine, tape)
    without = evaluate(mine, tape, exclude_own=False)
    assert with_exclusion.vwap == D("100")
    assert without.vwap == D("109")
    # and the flattery is visible: excluding your prints makes the score worse
    assert with_exclusion.slippage_vs_vwap_bps < without.slippage_vs_vwap_bps


def test_evaluate_measures_participation_against_the_full_tape():
    """Participation includes your own volume; the benchmark does not."""
    tape = [t(1, "100", "50"), t(1, "100", "50")]
    r = evaluate([f(1, "100", "50", Side.BUY)], tape)
    assert r.participation_rate == D("0.5")


def test_evaluate_window_spans_first_to_last_fill():
    tape = [t(1, "100", "1"), t(5, "500", "1"), t(50, "999", "1")]
    r = evaluate([f(1, "100", "1"), f(5, "500", "1")], tape)
    assert r.vwap is None or r.vwap != D("999")


def test_evaluate_can_add_twap():
    tape = [t(0, "100", "1"), t(SEC, "200", "1")]
    r = evaluate([f(0, "100", "1"), f(SEC, "200", "1")], tape,
                 twap_interval_ns=SEC, exclude_own=False)
    assert r.twap == D("150")


def test_an_explicit_window_rescues_the_single_fill_case():
    """The escape hatch for the degenerate case above."""
    tape = [t(1, "100", "10"), t(2, "101", "5"), t(3, "102", "10")]
    r = evaluate([f(2, "101", "5", Side.BUY)], tape, start_ns=1, end_ns=4)
    assert r.vwap == D("101")            # 100x10 and 102x10, ours removed
    assert r.slippage_vs_vwap_bps == 0
    assert r.participation_rate == D("0.2")


def test_an_inverted_window_is_rejected():
    with pytest.raises(ValueError):
        evaluate([f(2, "101", "5")], [], start_ns=10, end_ns=5)


def test_evaluate_of_nothing_is_none():
    assert evaluate([], [t(1, "100")]) is None


def test_evaluate_refuses_mixed_sides():
    with pytest.raises(ValueError, match="same side"):
        evaluate([f(1, "100", "1", Side.BUY), f(2, "100", "1", Side.SELL)], [])


def test_evaluate_survives_an_empty_tape():
    r = evaluate([f(1, "100", "1", Side.BUY)], [])
    assert r.vwap is None and r.slippage_vs_vwap_bps is None
    assert r.participation_rate is None and r.average_price == D("100")
