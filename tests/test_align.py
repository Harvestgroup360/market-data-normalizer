"""As-of alignment tests: the backward-looking join, staleness, bar ends."""
from decimal import Decimal

import pytest

from mdnorm import (
    AlignedRow,
    AsOfSeries,
    BarField,
    EventType,
    Field,
    MarketEvent,
    align,
    align_bars,
    align_on,
    grid,
    time_bars,
)

D = Decimal
SEC = 1_000_000_000
MS = 1_000_000


def t(ts, price, size="1", symbol="X"):
    return MarketEvent(symbol=symbol, venue="v", event_type=EventType.TRADE,
                       ts_ns=ts, price=D(price), size=D(size))


def q(ts, bid, ask, symbol="X"):
    return MarketEvent(symbol=symbol, venue="v", event_type=EventType.QUOTE,
                       ts_ns=ts, bid_price=D(bid), bid_size=D("1"),
                       ask_price=D(ask), ask_size=D("1"))


# -- the join itself ---------------------------------------------------------

def test_as_of_takes_the_last_value_at_or_before():
    s = AsOfSeries.from_events([t(10, "100"), t(20, "200")])
    assert s.at(19) == (D("100"), 9)
    assert s.at(20) == (D("200"), 0)
    assert s.at(21) == (D("200"), 1)


def test_nothing_is_visible_before_the_first_observation():
    """The one case that must never forward-fill backwards."""
    s = AsOfSeries.from_events([t(10, "100")])
    assert s.at(9) == (None, None)


def test_the_join_never_reaches_forward_for_a_nearer_value():
    """A value 1ns in the future is still the future."""
    s = AsOfSeries.from_events([t(0, "100"), t(1_000, "999")])
    value, _ = s.at(999)
    assert value == D("100")


def test_a_repeated_timestamp_is_superseded_by_the_later_input():
    s = AsOfSeries.from_events([t(10, "100"), t(10, "101")])
    assert s.at(10) == (D("101"), 0)
    assert len(s) == 1


def test_out_of_order_input_is_sorted():
    s = AsOfSeries.from_events([t(20, "200"), t(10, "100")])
    assert s.at(15)[0] == D("100")
    assert (s.first_ts_ns, s.last_ts_ns) == (10, 20)


def test_an_empty_series_has_no_span():
    s = AsOfSeries([])
    assert len(s) == 0
    assert s.first_ts_ns is None and s.last_ts_ns is None
    assert s.at(1) == (None, None)


def test_negative_observation_timestamps_are_rejected():
    with pytest.raises(ValueError):
        AsOfSeries([(-1, D("1"))])


# -- fields ------------------------------------------------------------------

def test_price_comes_from_trades_only():
    s = AsOfSeries.from_events([q(1, "99", "101"), t(2, "100")], field=Field.PRICE)
    assert len(s) == 1 and s.at(2)[0] == D("100")


def test_mid_comes_from_quotes_only():
    s = AsOfSeries.from_events([t(1, "500"), q(2, "99", "101")], field=Field.MID)
    assert len(s) == 1 and s.at(2)[0] == D("100")


def test_a_one_sided_quote_has_no_mid_and_is_skipped():
    one_sided = MarketEvent(symbol="X", venue="v", event_type=EventType.QUOTE,
                            ts_ns=1, bid_price=D("99"), bid_size=D("1"))
    s = AsOfSeries.from_events([one_sided], field=Field.MID)
    assert len(s) == 0


@pytest.mark.parametrize("field,expected", [
    (Field.BID, D("99")), (Field.ASK, D("101")), (Field.MID, D("100")),
])
def test_quote_fields(field, expected):
    s = AsOfSeries.from_events([q(1, "99", "101")], field=field)
    assert s.at(1)[0] == expected


# -- staleness ---------------------------------------------------------------

def test_a_value_older_than_the_window_is_dropped_but_its_age_is_kept():
    """A frozen price is worse than a missing one; both must be tellable apart."""
    s = AsOfSeries.from_events([t(0, "100")])
    assert s.at(10 * SEC, max_age_ns=5 * SEC) == (None, 10 * SEC)
    assert s.at(4 * SEC, max_age_ns=5 * SEC) == (D("100"), 4 * SEC)


def test_a_value_exactly_at_the_staleness_limit_survives():
    s = AsOfSeries.from_events([t(0, "100")])
    assert s.at(5 * SEC, max_age_ns=5 * SEC)[0] == D("100")


def test_a_negative_staleness_window_is_rejected():
    with pytest.raises(ValueError):
        AsOfSeries([]).at(1, max_age_ns=-1)


def test_stale_and_missing_columns_are_reported_separately():
    rows = align({"live": [t(0, "1"), t(9 * SEC, "2")],
                  "dead": [t(0, "50")],
                  "unborn": [t(20 * SEC, "7")]},
                 interval_ns=9 * SEC, start_ns=9 * SEC, end_ns=9 * SEC + 1,
                 max_age_ns=2 * SEC)
    row = rows[0]
    assert row.values["live"] == D("2")
    assert row.stale == ("dead",)       # had data, too old to use
    assert row.missing == ("unborn",)   # never had data yet
    assert row.complete is False


# -- delay -------------------------------------------------------------------

def test_a_delayed_series_cannot_be_read_before_it_arrives():
    s = AsOfSeries.from_events([t(0, "100")]).delayed(250 * MS)
    assert s.at(249 * MS) == (None, None)
    assert s.at(250 * MS) == (D("100"), 0)


def test_delay_does_not_mutate_the_original():
    s = AsOfSeries.from_events([t(0, "100")])
    s.delayed(SEC)
    assert s.at(0)[0] == D("100")


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError):
        AsOfSeries([]).delayed(-1)


def test_delay_changes_which_value_a_grid_point_sees():
    fast = {"a": AsOfSeries.from_events([t(0, "100"), t(9, "200")])}
    slow = {"a": AsOfSeries.from_events([t(0, "100"), t(9, "200")]).delayed(5)}
    assert align_on([10], fast)[0].values["a"] == D("200")
    assert align_on([10], slow)[0].values["a"] == D("100")


# -- the grid ----------------------------------------------------------------

def test_grid_is_half_open():
    assert grid(0, 10, 5) == [0, 5]


def test_grid_covers_a_partial_final_interval():
    assert grid(0, 11, 5) == [0, 5, 10]


def test_grid_rejects_a_non_positive_interval():
    with pytest.raises(ValueError):
        grid(0, 10, 0)


def test_grid_rejects_an_inverted_window():
    with pytest.raises(ValueError):
        grid(10, 5, 1)


def test_an_absurd_grid_is_refused_rather_than_allocated():
    """A nanosecond grid over a day is a typo, not a request."""
    with pytest.raises(ValueError, match="rows"):
        grid(0, 86_400 * SEC, 1)


# -- align -------------------------------------------------------------------

def test_align_puts_one_row_per_grid_point():
    rows = align({"A": [t(1 * SEC, "100"), t(5 * SEC, "101")],
                  "B": [t(2 * SEC, "200")]},
                 interval_ns=3 * SEC)
    assert [r.ts_ns for r in rows] == [3 * SEC, 6 * SEC]
    assert rows[0].values == {"A": D("100"), "B": D("200")}
    assert rows[1].values == {"A": D("101"), "B": D("200")}


def test_align_starts_at_the_first_grid_point_that_can_have_data():
    rows = align({"A": [t(7, "100")]}, interval_ns=5)
    assert rows[0].ts_ns == 10


def test_align_window_can_be_pinned_so_runs_line_up():
    a = align({"A": [t(10, "1")]}, interval_ns=5, start_ns=0, end_ns=20)
    b = align({"B": [t(17, "2")]}, interval_ns=5, start_ns=0, end_ns=20)
    assert [r.ts_ns for r in a] == [r.ts_ns for r in b] == [0, 5, 10, 15]
    assert a[0].values["A"] is None      # honest hole, not a back-fill


def test_align_of_nothing_is_empty():
    assert align({}, interval_ns=SEC) == []
    assert align({"A": []}, interval_ns=SEC) == []


def test_align_is_empty_when_no_stream_has_the_requested_field():
    assert align({"A": [t(1, "100")]}, interval_ns=SEC, field=Field.MID) == []


def test_a_derived_bound_that_closes_the_window_early_yields_no_rows():
    """The first grid point lands past the pinned end; that is not an error."""
    assert align({"A": [t(7, "100")]}, interval_ns=5, end_ns=8) == []


def test_pinning_both_bounds_inverted_is_an_error():
    with pytest.raises(ValueError, match="greater"):
        align({"A": [t(7, "100")]}, interval_ns=5, start_ns=100, end_ns=50)


def test_the_default_window_reaches_the_last_observation():
    rows = align({"A": [t(0, "1"), t(5 * SEC, "2")]}, interval_ns=3 * SEC)
    assert [r.ts_ns for r in rows] == [0, 3 * SEC, 6 * SEC]
    assert rows[-1].values["A"] == D("2")


def test_require_all_drops_incomplete_rows():
    streams = {"A": [t(0, "1")], "B": [t(10, "2")]}
    loose = align(streams, interval_ns=5, start_ns=0, end_ns=15)
    strict = align(streams, interval_ns=5, start_ns=0, end_ns=15, require_all=True)
    assert [r.ts_ns for r in loose] == [0, 5, 10]
    assert [r.ts_ns for r in strict] == [10]


def test_align_accepts_ready_made_series_alongside_raw_events():
    rows = align({"raw": [t(0, "1")],
                  "prepared": AsOfSeries.from_events([t(0, "2")])},
                 interval_ns=5, start_ns=0, end_ns=6)
    assert rows[0].values == {"raw": D("1"), "prepared": D("2")}


def test_align_on_uses_the_timestamps_you_give_it():
    rows = align_on([100, 300], {"A": [t(50, "1"), t(200, "2")]})
    assert [(r.ts_ns, r.values["A"]) for r in rows] == [(100, D("1")), (300, D("2"))]


def test_align_on_sorts_the_timestamps():
    rows = align_on([300, 100], {"A": [t(50, "1")]})
    assert [r.ts_ns for r in rows] == [100, 300]


def test_align_on_an_event_stream_of_a_reference_instrument():
    """The common research shape: one row per print of the thing you trade."""
    reference = [t(100, "10"), t(400, "11")]
    rows = align_on([e.ts_ns for e in reference],
                    {"other": [t(50, "1"), t(300, "2")]})
    assert [r.values["other"] for r in rows] == [D("1"), D("2")]


# -- rows --------------------------------------------------------------------

def test_a_row_reports_completeness():
    full = AlignedRow(1, {"a": D("1")}, {"a": 0})
    holed = AlignedRow(1, {"a": None}, {"a": None})
    assert full.complete is True
    assert holed.complete is False
    assert AlignedRow(1, {}, {}).complete is False


# -- bars: the look-ahead that survives everything else ----------------------

def test_a_bar_is_observable_at_its_end_not_its_label():
    bars = time_bars([t(0, "100"), t(SEC + 1, "200")], SEC)
    s = AsOfSeries.from_bars(bars)
    assert s.at(SEC - 1) == (None, None)      # the 09:30 bar is not out yet
    assert s.at(SEC)[0] == D("100")           # now it has closed


def test_joining_bars_on_their_label_would_import_the_future():
    """Pinned because this is the bug the module exists to prevent.

    The bar starting at 0 closes at 100 and covers everything up to 1s. A join
    on the label would let a grid point at 0 see that close; the bar-end rule
    makes the same grid point see nothing, which is the truth at time zero.
    """
    bars = time_bars([t(0, "1"), t(SEC // 2, "100")], SEC)
    label_join = AsOfSeries([(b.start_ns, b.close) for b in bars])
    end_join = AsOfSeries.from_bars(bars)
    assert label_join.at(0)[0] == D("100")    # half a second of hindsight
    assert end_join.at(0) == (None, None)


def test_align_bars_gives_the_last_closed_bar_per_column():
    a = time_bars([t(0, "100"), t(SEC, "110"), t(2 * SEC, "120")], SEC)
    b = time_bars([t(0, "200"), t(SEC, "210"), t(2 * SEC, "220")], SEC)
    rows = align_bars({"A": a, "B": b}, interval_ns=SEC)
    assert [r.ts_ns for r in rows] == [SEC, 2 * SEC, 3 * SEC]
    assert rows[0].values == {"A": D("100"), "B": D("200")}
    assert rows[-1].values == {"A": D("120"), "B": D("220")}


@pytest.mark.parametrize("field,expected", [
    (BarField.OPEN, D("100")), (BarField.HIGH, D("120")),
    (BarField.LOW, D("90")), (BarField.CLOSE, D("110")),
])
def test_bar_fields(field, expected):
    bars = time_bars([t(0, "100"), t(1, "120"), t(2, "90"), t(3, "110")], SEC)
    assert AsOfSeries.from_bars(bars, field=field).at(SEC)[0] == expected


def test_a_bar_without_a_vwap_is_skipped_rather_than_zeroed():
    sizeless = MarketEvent(symbol="X", venue="v", event_type=EventType.TRADE,
                           ts_ns=0, price=D("100"))
    bars = time_bars([sizeless], SEC)
    assert bars[0].vwap is None
    assert len(AsOfSeries.from_bars(bars, field=BarField.VWAP)) == 0


# -- integration -------------------------------------------------------------

def test_a_consolidated_top_of_book_can_be_a_column():
    from mdnorm import consolidate
    tops = consolidate([
        MarketEvent(symbol="X", venue="a", event_type=EventType.QUOTE,
                    ts_ns=SEC, bid_price=D("99"), bid_size=D("1"),
                    ask_price=D("101"), ask_size=D("1")),
    ])
    rows = align({"nbbo": tops, "trades": [t(SEC, "100")]},
                 interval_ns=SEC, field=Field.MID, start_ns=SEC, end_ns=SEC + 1)
    assert rows[0].values["nbbo"] == D("100")
    assert rows[0].values["trades"] is None    # a trade carries no mid
