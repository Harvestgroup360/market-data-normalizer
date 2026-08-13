"""Trade classification and microstructure metric tests."""
import random
from decimal import Decimal

import pytest

from mdnorm import (
    EventType,
    MarketEvent,
    Pipeline,
    Side,
    SideRule,
    effective_spreads,
    imbalance_bars,
    infer_sides,
    mean_effective_spread,
    quote_rule,
    roll_spread,
    signed_volume,
    tick_rule,
    trade_imbalance,
)

D = Decimal


def trade(ts, price, size="1", side=None):
    return MarketEvent(
        symbol="BTC-USD", venue="x", event_type=EventType.TRADE,
        ts_ns=ts, price=D(price), size=D(size), side=side,
    )


def quote(ts, bid, ask):
    return MarketEvent(
        symbol="BTC-USD", venue="x", event_type=EventType.QUOTE, ts_ns=ts,
        bid_price=D(bid), bid_size=D("1"), ask_price=D(ask), ask_size=D("1"),
    )


def sides(events):
    return [e.side for e in events if e.event_type is EventType.TRADE]


# -- the primitive rules ----------------------------------------------------

def test_quote_rule_above_below_and_at_the_mid():
    assert quote_rule(D("101"), D("99"), D("101")) is Side.BUY
    assert quote_rule(D("99"), D("99"), D("101")) is Side.SELL
    assert quote_rule(D("100"), D("99"), D("101")) is None   # at the mid
    assert quote_rule(D("100"), None, D("101")) is None      # unquoted


def test_tick_rule_directions_and_zero_ticks():
    prices = [D(p) for p in ("100", "101", "101", "100", "100", "102")]
    assert tick_rule(prices) == [
        None,        # nothing to compare against yet
        Side.BUY,    # uptick
        Side.BUY,    # zero-uptick inherits
        Side.SELL,   # downtick
        Side.SELL,   # zero-downtick inherits
        Side.BUY,    # uptick
    ]


def test_tick_rule_flat_series_is_all_unknown():
    assert tick_rule([D("100")] * 4) == [None] * 4


# -- infer_sides ------------------------------------------------------------

def test_quote_rule_over_a_stream():
    events = [quote(1, "99", "101"), trade(2, "101"), trade(3, "99"), trade(4, "100")]
    out = infer_sides(events, rule=SideRule.QUOTE)
    assert sides(out) == [Side.BUY, Side.SELL, None]   # mid trade stays unknown


def test_lee_ready_falls_back_to_the_tick_rule_at_the_mid():
    events = [quote(1, "99", "101"), trade(2, "101"), trade(3, "100")]
    # The 101 print is an uptick from nothing, then 100 sits exactly on the
    # mid: the quote rule abstains and the tick rule calls it a downtick.
    out = infer_sides(events, rule=SideRule.LEE_READY)
    assert sides(out) == [Side.BUY, Side.SELL]


def test_lee_ready_falls_back_before_the_first_quote():
    events = [trade(1, "100"), trade(2, "101"), quote(3, "99", "101"), trade(4, "101")]
    out = infer_sides(events, rule=SideRule.LEE_READY)
    assert sides(out) == [None, Side.BUY, Side.BUY]


def test_tick_rule_ignores_quotes_entirely():
    events = [quote(1, "500", "600"), trade(2, "100"), trade(3, "101")]
    out = infer_sides(events, rule=SideRule.TICK)
    assert sides(out) == [None, Side.BUY]


def test_reported_side_is_not_overwritten():
    events = [quote(1, "99", "101"), trade(2, "101", side=Side.SELL)]
    assert sides(infer_sides(events))[0] is Side.SELL


def test_overwrite_replaces_the_reported_side():
    events = [quote(1, "99", "101"), trade(2, "101", side=Side.SELL)]
    assert sides(infer_sides(events, overwrite=True))[0] is Side.BUY


def test_lag_selects_an_earlier_quote():
    events = [quote(10, "99", "101"), quote(100, "200", "202"), trade(105, "201")]
    # Contemporaneous: 201 is inside 200/202, below the 201 mid -> unclassified,
    # so Lee-Ready falls back to the tick rule, which has nothing yet.
    assert sides(infer_sides(events))[0] is None
    # With a 50ns lag the old 99/101 quote applies and 201 is far above its mid.
    assert sides(infer_sides(events, lag_ns=50))[0] is Side.BUY


def test_quotes_pass_through_untouched_and_order_is_preserved():
    events = [trade(5, "101"), quote(1, "99", "101"), trade(3, "99")]
    out = infer_sides(events)
    assert [e.ts_ns for e in out] == [5, 1, 3]
    assert out[1].event_type is EventType.QUOTE and out[1].side is None
    assert out[2].side is Side.SELL


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError):
        infer_sides([], lag_ns=-1)


# -- metrics ----------------------------------------------------------------

def test_signed_volume_and_imbalance():
    events = [
        trade(1, "100", "3", Side.BUY),
        trade(2, "100", "1", Side.SELL),
        trade(3, "100", "5", None),      # unclassified: ignored by both
    ]
    assert signed_volume(events) == D("2")
    assert trade_imbalance(events) == D("2") / D("4")


def test_imbalance_is_none_when_nothing_is_classified():
    assert trade_imbalance([trade(1, "100", "5")]) is None


def test_imbalance_bounds():
    assert trade_imbalance([trade(1, "100", "2", Side.BUY)]) == 1
    assert trade_imbalance([trade(1, "100", "2", Side.SELL)]) == -1


def test_effective_spread_uses_the_prevailing_quote():
    events = [quote(1, "99", "101"), trade(2, "101"), trade(3, "100.5")]
    # mid is 100: 2*|101-100| = 2, 2*|100.5-100| = 1
    assert effective_spreads(events) == [D("2"), D("1")]
    assert mean_effective_spread(events) == D("1.5")


def test_trades_before_any_quote_are_skipped_not_zeroed():
    events = [trade(1, "100"), quote(2, "99", "101"), trade(3, "101")]
    assert effective_spreads(events) == [D("2")]


def test_mean_effective_spread_is_none_without_quotes():
    assert mean_effective_spread([trade(1, "100")]) is None


def test_roll_spread_recovers_bid_ask_bounce():
    """A flat mid of 100 quoted 2 wide: the estimator should find the 2."""
    rnd = random.Random(7)
    events = [
        trade(i, "101" if rnd.random() < 0.5 else "99") for i in range(2000)
    ]
    est = roll_spread(events)
    assert est is not None
    assert abs(est - D("2")) < D("0.15")


def test_roll_spread_overstates_when_signs_alternate_perfectly():
    """Roll assumes serially uncorrelated trade signs.

    Strict alternation is the maximally autocorrelated case and violates
    that assumption, which doubles the estimate. Encoded here so the
    limitation is visible rather than discovered in production.
    """
    events = [trade(i, "101" if i % 2 else "99") for i in range(50)]
    assert roll_spread(events) == D("4")   # true spread is 2


def test_roll_spread_is_none_on_a_trend():
    events = [trade(i, str(100 + i)) for i in range(20)]
    assert roll_spread(events) is None


def test_roll_spread_needs_enough_prices():
    assert roll_spread([trade(1, "100"), trade(2, "101")]) is None


# -- imbalance bars ---------------------------------------------------------

def test_imbalance_bars_close_on_directional_flow():
    events = [
        trade(1, "100", "2", Side.BUY),
        trade(2, "101", "1", Side.SELL),
        trade(3, "102", "2", Side.BUY),   # signed = 3 -> closes
        trade(4, "103", "1", Side.BUY),
    ]
    bars = imbalance_bars(events, D("3"))
    assert len(bars) == 2
    assert bars[0].trades == 3 and bars[0].open == D("100") and bars[0].close == D("102")
    assert bars[1].trades == 1          # trailing partial bar


def test_imbalance_bars_trigger_on_either_direction():
    events = [trade(i, "100", "1", Side.SELL) for i in range(1, 4)]
    assert len(imbalance_bars(events, D("2"))) == 2


def test_balanced_flow_never_closes_a_bar():
    events = [
        trade(1, "100", "1", Side.BUY),
        trade(2, "100", "1", Side.SELL),
        trade(3, "100", "1", Side.BUY),
        trade(4, "100", "1", Side.SELL),
    ]
    bars = imbalance_bars(events, D("2"))
    assert len(bars) == 1 and bars[0].trades == 4


def test_unclassified_trades_do_not_fabricate_imbalance():
    events = [trade(i, "100", "10") for i in range(1, 6)]
    bars = imbalance_bars(events, D("1"))
    assert len(bars) == 1 and bars[0].volume == D("50")


def test_imbalance_bars_by_tick_count():
    events = [
        trade(1, "100", "99", Side.BUY),
        trade(2, "100", "99", Side.BUY),
        trade(3, "100", "99", Side.SELL),
    ]
    bars = imbalance_bars(events, D("2"), by="tick")
    assert len(bars) == 2 and bars[0].trades == 2


def test_imbalance_bars_reject_a_non_positive_threshold():
    with pytest.raises(ValueError):
        imbalance_bars([], D("0"))


def test_imbalance_bars_reject_an_unknown_measure():
    with pytest.raises(ValueError):
        imbalance_bars([], D("1"), by="notional")


# -- pipeline ---------------------------------------------------------------

def test_pipeline_infers_then_builds_imbalance_bars():
    events = [
        quote(1, "99", "101"),
        trade(2, "101", "2"),    # buy
        trade(3, "101", "2"),    # zero-tick / above mid -> buy, closes at 4
        trade(4, "99", "1"),     # sell
    ]
    pipe = Pipeline().infer_sides().imbalance_bars(Decimal("4"))
    bars = pipe.run(events)
    assert pipe.steps == ["infer_sides", "imbalance_bars"]
    assert len(bars) == 2
    assert bars[0].trades == 2 and bars[1].trades == 1
