"""Order book reconstruction tests."""
from decimal import Decimal

import pytest

from mdnorm import (
    BookDelta,
    EventType,
    OrderBook,
    SequenceGapError,
    Side,
    mean_effective_spread,
    replay_book,
)

D = Decimal


def book(**kw):
    b = OrderBook("BTC-USD", "binance", **kw)
    b.apply_snapshot(
        1_000,
        bids=[(D("100"), D("2")), (D("99"), D("5"))],
        asks=[(D("101"), D("3")), (D("102"), D("4"))],
        seq=10,
    )
    return b


def delta(ts, side, price, size, seq=None):
    return BookDelta(ts, side, D(price), D(size), seq=seq)


# -- snapshot and top of book ----------------------------------------------

def test_snapshot_establishes_both_sides():
    b = book()
    assert b.best_bid == (D("100"), D("2"))
    assert b.best_ask == (D("101"), D("3"))
    assert b.mid == D("100.5")
    assert b.spread == D("1")
    assert b.ts_ns == 1_000 and b.seq == 10


def test_snapshot_replaces_rather_than_merges():
    b = book()
    b.apply_snapshot(2_000, bids=[(D("50"), D("1"))], asks=[(D("60"), D("1"))], seq=20)
    assert b.best_bid == (D("50"), D("1"))
    assert b.depth(Side.BUY, 5) == [(D("50"), D("1"))]


def test_empty_book_has_no_top():
    b = OrderBook("X", "v")
    assert b.best_bid is None and b.best_ask is None
    assert b.mid is None and b.spread is None
    assert b.to_quote() is None


# -- deltas ----------------------------------------------------------------

def test_delta_improves_the_bid():
    b = book()
    b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=11))
    assert b.best_bid == (D("100.5"), D("1"))
    assert b.spread == D("0.5")


def test_zero_size_removes_the_level():
    b = book()
    b.apply(delta(1_100, Side.BUY, "100", "0", seq=11))
    assert b.best_bid == (D("99"), D("5"))


def test_size_is_absolute_not_a_delta():
    """Feeds send the new resting total, not a change."""
    b = book()
    b.apply(delta(1_100, Side.BUY, "100", "7", seq=11))
    assert b.best_bid == (D("100"), D("7"))


def test_removing_a_missing_level_is_harmless():
    b = book()
    b.apply(delta(1_100, Side.SELL, "500", "0", seq=11))
    assert b.best_ask == (D("101"), D("3"))


def test_deltas_deep_in_the_book_do_not_move_the_top():
    b = book()
    b.apply(delta(1_100, Side.SELL, "102", "9", seq=11))
    assert b.best_ask == (D("101"), D("3"))
    assert b.depth(Side.SELL, 2)[1] == (D("102"), D("9"))


def test_timestamp_advances_with_each_delta():
    b = book()
    b.apply(delta(1_500, Side.BUY, "98", "1", seq=11))
    assert b.ts_ns == 1_500


# -- ordering --------------------------------------------------------------

def test_depth_is_ordered_best_first():
    b = book()
    b.apply_many([
        delta(1_100, Side.BUY, "97", "1", seq=11),
        delta(1_200, Side.BUY, "101.5", "1", seq=12),
        delta(1_300, Side.SELL, "100.5", "1", seq=13),
    ])
    assert [p for p, _ in b.depth(Side.BUY, 4)] == [D("101.5"), D("100"), D("99"), D("97")]
    assert [p for p, _ in b.depth(Side.SELL, 3)] == [D("100.5"), D("101"), D("102")]


def test_depth_rejects_non_positive_levels():
    with pytest.raises(ValueError):
        book().depth(Side.BUY, 0)


# -- sequence integrity ----------------------------------------------------

def test_a_gap_raises_rather_than_corrupting_the_book():
    b = book()
    with pytest.raises(SequenceGapError, match="expected 11 but received 13"):
        b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=13))


def test_the_gap_message_counts_the_missing_updates():
    b = book()
    with pytest.raises(SequenceGapError, match=r"2 update\(s\) missing"):
        b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=13))


def test_a_replayed_message_is_rejected():
    b = book()
    with pytest.raises(SequenceGapError, match="not newer"):
        b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=10))


def test_the_book_is_unchanged_after_a_rejected_delta():
    b = book()
    with pytest.raises(SequenceGapError):
        b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=99))
    assert b.best_bid == (D("100"), D("2")) and b.seq == 10


def test_a_snapshot_resynchronises_after_a_gap():
    b = book()
    with pytest.raises(SequenceGapError):
        b.apply(delta(1_100, Side.BUY, "100.5", "1", seq=99))
    b.apply_snapshot(1_200, bids=[(D("100.5"), D("1"))], asks=[(D("101"), D("3"))], seq=99)
    b.apply(delta(1_300, Side.BUY, "100.6", "1", seq=100))
    assert b.best_bid == (D("100.6"), D("1"))


def test_unnumbered_feeds_are_accepted():
    b = book()
    b.apply(delta(1_100, Side.BUY, "100.5", "1"))
    assert b.best_bid == (D("100.5"), D("1")) and b.seq == 10


def test_strict_sequence_can_be_switched_off():
    b = OrderBook("X", "v", strict_sequence=False)
    b.apply_snapshot(1, bids=[(D("10"), D("1"))], asks=[(D("11"), D("1"))], seq=1)
    b.apply(delta(2, Side.BUY, "10.5", "1", seq=50))
    assert b.best_bid == (D("10.5"), D("1"))


# -- crossed books ---------------------------------------------------------

def test_a_healthy_book_is_not_crossed():
    assert book().is_crossed is False


def test_a_crossed_book_is_reported_not_hidden():
    b = book()
    b.apply(delta(1_100, Side.BUY, "101.5", "1", seq=11))
    assert b.is_crossed is True
    assert b.spread == D("-0.5")     # negative, and visibly so


def test_a_locked_book_counts_as_crossed():
    b = book()
    b.apply(delta(1_100, Side.BUY, "101", "1", seq=11))
    assert b.is_crossed is True and b.spread == 0


# -- book imbalance --------------------------------------------------------

def test_imbalance_at_the_top_of_book():
    b = book()                       # 2 bid vs 3 ask
    assert b.imbalance() == (D("2") - D("3")) / D("5")


def test_imbalance_over_several_levels():
    b = book()                       # bids 2+5, asks 3+4
    assert b.imbalance(2) == (D("7") - D("7")) / D("14")


def test_imbalance_is_none_when_a_side_is_empty():
    b = OrderBook("X", "v")
    b.apply_snapshot(1, bids=[(D("10"), D("1"))], asks=[])
    assert b.imbalance() is None


# -- depth cap -------------------------------------------------------------

def test_max_depth_drops_the_worst_levels():
    b = OrderBook("X", "v", max_depth=2)
    b.apply_snapshot(1, bids=[(D("10"), D("1")), (D("9"), D("1")), (D("8"), D("1"))],
                     asks=[(D("11"), D("1"))])
    assert [p for p, _ in b.depth(Side.BUY, 5)] == [D("10"), D("9")]
    b.apply(delta(2, Side.BUY, "10.5", "1"))
    assert [p for p, _ in b.depth(Side.BUY, 5)] == [D("10.5"), D("10")]


def test_max_depth_must_be_positive():
    with pytest.raises(ValueError):
        OrderBook("X", "v", max_depth=0)


# -- validation ------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"price": D("0")}, {"price": D("-1")}, {"size": D("-1")}, {"ts_ns": -1},
])
def test_deltas_are_validated(kwargs):
    args = {"ts_ns": 1, "side": Side.BUY, "price": D("100"), "size": D("1")}
    args.update(kwargs)
    with pytest.raises(ValueError):
        BookDelta(**args)


# -- bridge into the rest of the library -----------------------------------

def test_to_quote_produces_a_usable_market_event():
    q = book().to_quote()
    assert q.event_type is EventType.QUOTE
    assert q.symbol == "BTC-USD" and q.venue == "binance"
    assert q.bid_price == D("100") and q.ask_price == D("101")
    assert q.mid_price == D("100.5") and q.spread == D("1")


def test_to_quote_handles_a_one_sided_book():
    b = OrderBook("X", "v")
    b.apply_snapshot(1, bids=[(D("10"), D("1"))], asks=[])
    q = b.to_quote()
    assert q.bid_price == D("10") and q.ask_price is None


def test_replay_emits_only_when_the_top_changes():
    b = book()
    deltas = [
        delta(1_100, Side.SELL, "102", "9", seq=11),    # deep: no emit
        delta(1_200, Side.BUY, "100.5", "1", seq=12),   # new best bid: emit
        delta(1_300, Side.BUY, "99", "6", seq=13),      # deep: no emit
        delta(1_400, Side.SELL, "101", "0", seq=14),    # best ask gone: emit
    ]
    quotes = list(replay_book(b, deltas))
    assert [q.ts_ns for q in quotes] == [1_200, 1_400]
    assert quotes[-1].ask_price == D("102")


def test_replay_can_emit_on_every_delta():
    b = book()
    deltas = [delta(1_100 + i, Side.SELL, "102", str(i + 1), seq=11 + i) for i in range(3)]
    assert len(list(replay_book(b, deltas, top_of_book_only=False))) == 3


def test_replay_leaves_the_book_inspectable():
    b = book()
    list(replay_book(b, [delta(1_100, Side.BUY, "100.5", "1", seq=11)]))
    assert b.best_bid == (D("100.5"), D("1"))


def test_reconstructed_quotes_feed_the_microstructure_metrics():
    """A book becomes quotes, and quotes are what the rest of the library eats."""
    from mdnorm import MarketEvent
    b = book()
    quotes = list(replay_book(b, [delta(1_100, Side.BUY, "100.5", "1", seq=11)]))
    trade = MarketEvent(symbol="BTC-USD", venue="binance", event_type=EventType.TRADE,
                        ts_ns=1_200, price=D("101"), size=D("1"))
    # mid after the delta is 100.75, so the effective spread is 2 * 0.25
    assert mean_effective_spread(quotes + [trade]) == D("0.50")
