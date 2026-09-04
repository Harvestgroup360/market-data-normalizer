"""Auction tests: the crosses, and what including them does to a benchmark."""
from datetime import date, time
from decimal import Decimal

import pytest

from mdnorm import EventType, MarketEvent, Session, session_bounds
from mdnorm.auctions import (
    AuctionKind,
    AuctionReport,
    AuctionWindow,
    VwapGap,
    auction_report,
    auction_windows,
    exclude_auctions,
    in_auction,
    split_auctions,
    vwap_gap,
)
from mdnorm.calendars import EarlyClose, Holiday, TradingCalendar

D = Decimal
S = 1_000_000_000
MIN = 60 * S

RTH = Session(start=time(9, 30), end=time(16, 0), tz="America/New_York")
DAYS = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]


def calendar(days=DAYS, early=(), holidays=()):
    return TradingCalendar(
        RTH, first_day=days[0], last_day=days[-1],
        early_closes=tuple(EarlyClose(d, time(13, 0)) for d in early),
        holidays=tuple(Holiday(d) for d in holidays))


def trade(ts, price, size):
    return MarketEvent(symbol="X", venue="v", ts_ns=ts,
                       event_type=EventType.TRADE,
                       price=D(price), size=D(size))


def quote(ts):
    return MarketEvent(symbol="X", venue="v", ts_ns=ts,
                       event_type=EventType.QUOTE,
                       bid_price=D("100.00"), ask_price=D("100.02"))


def day_events(day, *, open_size=200_000, close_size=900_000, n=60,
               session=RTH):
    o, c = session_bounds(day, session)
    out = [trade(o, "100.00", open_size)]
    for i in range(1, n):
        out.append(trade(o + i * MIN, "100.00", 1_000))
    out.append(trade(c, "101.00", close_size))
    return out


# -- windows ---------------------------------------------------------------

def test_a_window_is_built_for_each_cross_of_each_day():
    w = auction_windows(DAYS, calendar())
    assert len(w) == 6
    assert [x.kind for x in w[:2]] == [AuctionKind.OPEN, AuctionKind.CLOSE]
    assert {x.day for x in w} == set(DAYS)


def test_the_smallest_window_still_holds_a_print_stamped_at_the_bell():
    day = DAYS[0]
    o, c = session_bounds(day, RTH)
    w = auction_windows([day], calendar())
    assert in_auction(o, w) is AuctionKind.OPEN
    assert in_auction(c, w) is AuctionKind.CLOSE
    assert in_auction(o + 1, w) is None
    assert in_auction(c - 1, w) is None


def test_the_windows_widen_on_request():
    day = DAYS[0]
    o, c = session_bounds(day, RTH)
    w = auction_windows([day], calendar(), open_ns=30 * S, close_ns=30 * S)
    assert in_auction(o + 30 * S, w) is AuctionKind.OPEN
    assert in_auction(o + 31 * S, w) is None
    assert in_auction(c - 30 * S, w) is AuctionKind.CLOSE


def test_a_cross_published_after_the_bell_needs_lead():
    day = DAYS[0]
    _, c = session_bounds(day, RTH)
    plain = auction_windows([day], calendar())
    assert in_auction(c + 5 * S, plain) is None
    with_lead = auction_windows([day], calendar(), lead_ns=10 * S)
    assert in_auction(c + 5 * S, with_lead) is AuctionKind.CLOSE


def test_an_early_close_moves_the_closing_window_to_where_the_venue_shut():
    """The reason the windows come from a calendar rather than a constant."""
    short = DAYS[1]
    cal = calendar(early=[short])
    w = [x for x in auction_windows([short], cal)
         if x.kind is AuctionKind.CLOSE][0]
    regular_close = session_bounds(short, RTH)[1]
    early_close = regular_close - 3 * 3600 * S      # 13:00 instead of 16:00
    assert early_close in w
    assert regular_close not in w


def test_a_non_trading_day_produces_no_windows():
    cal = calendar(holidays=[DAYS[1]])
    assert [x.day for x in auction_windows(DAYS, cal)] == [DAYS[0]] * 2 + \
        [DAYS[2]] * 2


def test_a_day_outside_the_calendars_range_produces_no_windows():
    cal = calendar()
    assert auction_windows([date(2027, 6, 1)], cal) == []


def test_negative_extents_are_refused():
    for kwargs in ({"open_ns": -1}, {"close_ns": -1}, {"lead_ns": -1}):
        with pytest.raises(ValueError, match="non-negative"):
            auction_windows(DAYS, calendar(), **kwargs)


def test_a_window_must_cover_a_positive_span():
    with pytest.raises(ValueError, match="positive span"):
        AuctionWindow(AuctionKind.OPEN, 100, 100)


def test_in_auction_with_no_windows_is_always_none():
    assert in_auction(0, []) is None


# -- splitting -------------------------------------------------------------

def test_split_puts_the_crosses_on_one_side_and_keeps_order():
    day = DAYS[0]
    events = day_events(day)
    w = auction_windows([day], calendar())
    cont, auc = split_auctions(events, w)
    assert len(auc) == 2
    assert [e.size for e in auc] == [D(200_000), D(900_000)]
    assert len(cont) == len(events) - 2
    assert [e.ts_ns for e in cont] == sorted(e.ts_ns for e in cont)


def test_a_quote_inside_a_window_stays_with_the_continuous_stream():
    """It is still a quote, and the continuous side needs its state."""
    day = DAYS[0]
    o, _ = session_bounds(day, RTH)
    events = [quote(o), trade(o, "100.00", 500)]
    cont, auc = split_auctions(events, auction_windows([day], calendar()))
    assert len(auc) == 1 and auc[0].event_type is EventType.TRADE
    assert len(cont) == 1 and cont[0].event_type is EventType.QUOTE


def test_exclude_auctions_is_the_continuous_half():
    day = DAYS[0]
    events = day_events(day)
    w = auction_windows([day], calendar())
    assert exclude_auctions(events, w) == split_auctions(events, w)[0]


def test_no_windows_means_nothing_is_split_off():
    events = day_events(DAYS[0])
    cont, auc = split_auctions(events, [])
    assert auc == [] and len(cont) == len(events)


# -- report ----------------------------------------------------------------

def test_the_report_counts_volume_and_notional_separately():
    day = DAYS[0]
    events = day_events(day, n=60)
    r = auction_report(events, auction_windows([day], calendar()))
    assert r.trades == 61
    assert r.auction_trades == 2
    assert r.auction_volume == D(1_100_000)
    assert r.volume == D(1_100_000) + D(59_000)


def test_notional_share_exceeds_volume_share_when_a_cross_prints_high():
    """The crosses print at the ends of the range, not at its average."""
    day = DAYS[0]
    events = day_events(day)
    r = auction_report(events, auction_windows([day], calendar()))
    assert r.notional_share > r.volume_share


def test_largest_print_is_measured_even_with_no_windows_at_all():
    events = day_events(DAYS[0])
    r = auction_report(events)
    assert r.windows == 0
    assert r.auction_trades == 0
    assert r.volume_share == 0
    assert r.largest_print == D(900_000)
    assert r.largest_print_share > D("0.9") / 2


def test_an_empty_input_reports_no_shares_rather_than_zero():
    r = auction_report([])
    assert r.trades == 0
    assert r.volume_share is None
    assert r.notional_share is None
    assert r.largest_print_share is None


def test_trades_without_a_size_are_counted_but_not_weighted():
    day = DAYS[0]
    o, _ = session_bounds(day, RTH)
    sizeless = MarketEvent(symbol="X", venue="v", ts_ns=o + MIN,
                           event_type=EventType.TRADE, price=D("100"))
    events = [trade(o, "100.00", 1_000), sizeless]
    r = auction_report(events, auction_windows([day], calendar()))
    assert r.trades == 2
    assert r.volume == D(1_000)


def test_quotes_never_reach_the_report():
    day = DAYS[0]
    o, _ = session_bounds(day, RTH)
    r = auction_report([quote(o), trade(o, "100.00", 5)],
                       auction_windows([day], calendar()))
    assert r.trades == 1


# -- the benchmark gap -----------------------------------------------------

def test_the_two_benchmarks_differ_and_both_are_reported():
    day = DAYS[0]
    events = day_events(day)
    g = vwap_gap(events, auction_windows([day], calendar()))
    assert g.with_auctions is not None and g.without_auctions is not None
    assert g.with_auctions != g.without_auctions
    assert g.auction_only is not None
    assert g.difference == g.with_auctions - g.without_auctions


def test_the_difference_is_reported_in_basis_points_of_the_continuous_price():
    day = DAYS[0]
    events = day_events(day)
    g = vwap_gap(events, auction_windows([day], calendar()))
    expected = g.difference / g.without_auctions * 10_000
    assert g.difference_bps == expected
    assert g.difference_bps > 0        # the close printed above the day


def test_a_day_with_no_cross_in_the_windows_has_no_gap():
    day = DAYS[0]
    o, _ = session_bounds(day, RTH)
    events = [trade(o + i * MIN, "100.00", 1_000) for i in range(1, 30)]
    g = vwap_gap(events, auction_windows([day], calendar()))
    assert g.auction_volume_share == 0
    assert g.difference == 0
    assert g.difference_bps == 0
    assert g.auction_only is None


def test_a_file_that_is_only_crosses_has_no_continuous_benchmark():
    day = DAYS[0]
    o, c = session_bounds(day, RTH)
    events = [trade(o, "100.00", 10), trade(c, "101.00", 10)]
    g = vwap_gap(events, auction_windows([day], calendar()))
    assert g.without_auctions is None
    assert g.difference is None
    assert g.difference_bps is None
    assert g.auction_volume_share == 1


def test_the_volume_share_matches_the_report():
    day = DAYS[0]
    events = day_events(day)
    w = auction_windows([day], calendar())
    assert vwap_gap(events, w).auction_volume_share == \
        auction_report(events, w).volume_share


def test_an_empty_input_gives_no_benchmark_at_all():
    g = vwap_gap([], [])
    assert g.with_auctions is None and g.without_auctions is None
    assert g.auction_volume_share is None


def test_a_bigger_cross_moves_the_benchmark_further():
    day = DAYS[0]
    w = auction_windows([day], calendar())
    small = vwap_gap(day_events(day, close_size=10_000), w)
    large = vwap_gap(day_events(day, close_size=2_000_000), w)
    assert large.difference_bps > small.difference_bps


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (AuctionWindow(AuctionKind.OPEN, 0, 1),
                AuctionReport(0, 0, D(0), D(0), D(0), D(0), D(0), 0),
                VwapGap(None, None, None, None)):
        with pytest.raises(Exception):
            obj.kind = "x"  # type: ignore[misc]
