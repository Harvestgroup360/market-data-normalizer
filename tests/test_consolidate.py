"""Multi-venue quote consolidation tests."""
from decimal import Decimal

import pytest

from mdnorm import (
    CONSOLIDATED,
    Consolidator,
    EventType,
    MarketEvent,
    Side,
    consolidate,
)

D = Decimal
SEC = 1_000_000_000


def q(ts, venue, bid=None, ask=None, bid_size="1", ask_size="1", symbol="BTC-USD"):
    return MarketEvent(
        symbol=symbol, venue=venue, event_type=EventType.QUOTE, ts_ns=ts,
        bid_price=D(bid) if bid else None,
        bid_size=D(bid_size) if bid else None,
        ask_price=D(ask) if ask else None,
        ask_size=D(ask_size) if ask else None,
    )


# -- the basic job ---------------------------------------------------------

def test_best_bid_and_offer_come_from_different_venues():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    c.update(q(2, "beta", bid="99", ask="101"))
    assert c.best_bid.venue == "alpha" and c.best_bid.price == D("100")
    assert c.best_ask.venue == "beta" and c.best_ask.price == D("101")
    assert c.spread == D("1")


def test_consolidated_event_is_labelled_as_such():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    top = c.to_quote()
    assert top.venue == CONSOLIDATED and top.symbol == "BTC-USD"
    assert top.event_type is EventType.QUOTE


def test_a_venue_updating_its_own_quote_replaces_it():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    c.update(q(2, "alpha", bid="98", ask="103"))
    assert c.best_bid.price == D("98") and c.best_ask.price == D("103")


def test_one_sided_venues_contribute_to_one_side_only():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100"))
    c.update(q(2, "beta", ask="101"))
    assert c.best_bid.venue == "alpha" and c.best_ask.venue == "beta"


def test_the_schema_already_forbids_a_two_sided_empty_quote():
    """No guard needed in the consolidator: MarketEvent rejects it upstream."""
    with pytest.raises(ValueError):
        q(1, "alpha")


def test_other_symbols_are_not_merged_in():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    c.update(q(2, "beta", bid="9999", ask="9998", symbol="ETH-USD"))
    assert c.best_bid.price == D("100") and c.venues == ["alpha"]


def test_trades_are_ignored():
    c = Consolidator("BTC-USD")
    trade = MarketEvent(symbol="BTC-USD", venue="alpha", event_type=EventType.TRADE,
                        ts_ns=1, price=D("100"), size=D("1"), side=Side.BUY)
    assert c.update(trade) is None


# -- staleness -------------------------------------------------------------

def test_a_quiet_venue_stops_counting_after_the_cutoff():
    """The failure that makes a consolidated feed look great and be fiction."""
    c = Consolidator("BTC-USD", max_age_ns=2 * SEC)
    c.update(q(1 * SEC, "dead", bid="105", ask="106"))   # generous, then silent
    c.update(q(2 * SEC, "live", bid="100", ask="101"))
    assert c.best_bid.venue == "dead"                    # still fresh
    c.update(q(4 * SEC, "live", bid="100", ask="101"))   # 3s since dead spoke
    assert c.best_bid.venue == "live" and c.best_bid.price == D("100")


def test_stale_venues_are_reportable():
    c = Consolidator("BTC-USD", max_age_ns=2 * SEC)
    c.update(q(1 * SEC, "dead", bid="105", ask="106"))
    c.update(q(5 * SEC, "live", bid="100", ask="101"))
    assert c.stale_venues() == ["dead"]
    assert c.fresh_venues() == ["live"]


def test_a_venue_exactly_at_the_cutoff_is_still_fresh():
    c = Consolidator("BTC-USD", max_age_ns=2 * SEC)
    c.update(q(1 * SEC, "alpha", bid="105", ask="106"))
    c.update(q(3 * SEC, "beta", bid="100", ask="101"))
    assert c.best_bid.venue == "alpha"


def test_without_a_cutoff_nothing_is_ever_stale():
    c = Consolidator("BTC-USD")
    c.update(q(1, "dead", bid="105", ask="106"))
    c.update(q(10**12, "live", bid="100", ask="101"))
    assert c.stale_venues() == []
    assert c.best_bid.venue == "dead"


def test_max_age_must_be_positive():
    with pytest.raises(ValueError):
        Consolidator("X", max_age_ns=0)


# -- crossed markets -------------------------------------------------------

def test_a_healthy_consolidation_is_not_crossed():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    c.update(q(2, "beta", bid="99", ask="101"))
    assert c.is_crossed is False and c.crossed_updates == 0


def test_a_cross_between_venues_is_reported_and_counted():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="103", ask="104"))
    c.update(q(2, "beta", bid="99", ask="101"))     # alpha bids above beta's offer
    assert c.is_crossed is True
    assert c.spread == D("-2")
    assert c.crossed_updates == 1


def test_a_locked_market_counts_as_crossed():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="101", ask="104"))
    c.update(q(2, "beta", bid="99", ask="101"))
    assert c.is_crossed is True and c.spread == 0


# -- tie-breaking ----------------------------------------------------------

def test_equal_prices_are_broken_by_size():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", bid_size="1"))
    c.update(q(2, "beta", bid="100", bid_size="5"))
    assert c.best_bid.venue == "beta"


def test_equal_prices_and_sizes_are_broken_by_venue_name():
    c = Consolidator("BTC-USD")
    c.update(q(1, "zeta", bid="100", bid_size="1"))
    c.update(q(2, "alpha", bid="100", bid_size="1"))
    assert c.best_bid.venue == "alpha"


def test_the_tie_break_is_stable_across_insertion_orders():
    a = Consolidator("BTC-USD")
    for e in (q(1, "zeta", bid="100"), q(2, "alpha", bid="100")):
        a.update(e)
    b = Consolidator("BTC-USD")
    for e in (q(1, "alpha", bid="100"), q(2, "zeta", bid="100")):
        b.update(e)
    assert a.best_bid.venue == b.best_bid.venue


# -- emission ---------------------------------------------------------------

def test_update_emits_only_when_the_top_moves():
    c = Consolidator("BTC-USD")
    assert c.update(q(1, "alpha", bid="100", ask="102")) is not None
    assert c.update(q(2, "beta", bid="98", ask="103")) is None      # worse: no change
    assert c.update(q(3, "beta", bid="101", ask="103")) is not None  # better bid


def test_consolidate_returns_one_event_per_change():
    quotes = [
        q(1, "alpha", bid="100", ask="102"),
        q(2, "beta", bid="99", ask="101"),
        q(3, "beta", bid="99", ask="101"),      # unchanged
        q(4, "alpha", bid="100.5", ask="102"),
    ]
    out = consolidate(quotes)
    assert [e.ts_ns for e in out] == [1, 2, 4]
    assert out[-1].bid_price == D("100.5") and out[-1].ask_price == D("101")


def test_consolidate_sorts_by_timestamp_first():
    out = consolidate([
        q(5, "beta", bid="99", ask="101"),
        q(1, "alpha", bid="100", ask="102"),
    ])
    assert [e.ts_ns for e in out] == [1, 5]


def test_consolidate_on_an_empty_stream():
    assert consolidate([]) == []


def test_consolidate_picks_the_first_symbol_by_default():
    out = consolidate([q(1, "a", bid="100", ask="101"),
                       q(2, "b", bid="500", ask="501", symbol="ETH-USD")])
    assert all(e.symbol == "BTC-USD" for e in out)


def test_consolidate_can_target_a_symbol_explicitly():
    out = consolidate([q(1, "a", bid="100", ask="101"),
                       q(2, "b", bid="500", ask="501", symbol="ETH-USD")],
                      symbol="ETH-USD")
    assert [e.symbol for e in out] == ["ETH-USD"]


def test_consolidate_applies_the_staleness_cutoff():
    quotes = [
        q(1 * SEC, "dead", bid="105", ask="106"),
        q(2 * SEC, "live", bid="100", ask="101"),
        q(9 * SEC, "live", bid="100", ask="101"),
    ]
    out = consolidate(quotes, max_age_ns=2 * SEC)
    assert out[-1].bid_price == D("100")     # the dead venue stopped voting


# -- leadership ------------------------------------------------------------

def test_leadership_counts_which_venue_sets_the_price():
    c = Consolidator("BTC-USD")
    c.update(q(1, "alpha", bid="100", ask="102"))
    c.update(q(2, "beta", bid="99", ask="101"))
    c.update(q(3, "beta", bid="101", ask="101"))
    # Counted per update, not per change: alpha held the best bid through the
    # first two updates, beta took it on the third.
    assert c.leadership["alpha"] == {"bid": 2, "ask": 1}
    assert c.leadership["beta"] == {"bid": 1, "ask": 2}


def test_leadership_is_empty_before_any_quote():
    assert Consolidator("X").leadership == {}


# -- integration -----------------------------------------------------------

def test_consolidated_quotes_feed_the_microstructure_metrics():
    from mdnorm import mean_effective_spread
    top = consolidate([q(1, "alpha", bid="100", ask="102"),
                       q(2, "beta", bid="99", ask="101")])
    trade = MarketEvent(symbol="BTC-USD", venue="beta", event_type=EventType.TRADE,
                        ts_ns=3, price=D("101"), size=D("1"))
    # consolidated mid is 100.5, so the effective spread is 2 * 0.5
    assert mean_effective_spread(top + [trade]) == D("1.0")


def test_a_reconstructed_book_can_be_consolidated():
    """Books become quotes, quotes consolidate — the pieces compose."""
    from mdnorm import OrderBook
    tops = []
    for venue, bid, ask in (("alpha", "100", "102"), ("beta", "99", "101")):
        b = OrderBook("BTC-USD", venue)
        b.apply_snapshot(1, [(D(bid), D("1"))], [(D(ask), D("1"))])
        tops.append(b.to_quote())
    out = consolidate(tops)
    assert out[-1].bid_price == D("100") and out[-1].ask_price == D("101")
