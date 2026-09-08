"""Coverage tests: which silences are ordinary and which are missing data."""
from datetime import date, time
from decimal import Decimal

import pytest

from mdnorm.calendars import EarlyClose, Holiday, TradingCalendar
from mdnorm.coverage import (
    CoverageReport,
    Gap,
    PanelCoverage,
    coverage_report,
    explain_gaps,
    find_gaps,
    panel_coverage,
)
from mdnorm.halts import Halt, HaltKind
from mdnorm.schema import EventType, MarketEvent
from mdnorm.sessions import parse_session

D = Decimal
S = 1_000_000_000
M = 60 * S
H = 60 * M
DAY = 24 * H

SESSION = parse_session("09:30-16:00", "America/New_York")
CAL = TradingCalendar(SESSION, first_day=date(2026, 3, 1),
                      last_day=date(2026, 3, 31),
                      holidays=[Holiday(date(2026, 3, 10), "test holiday")],
                      early_closes=[EarlyClose(date(2026, 3, 11), time(13, 0))],
                      name="demo")


def ev(ts, symbol="AAA", price="100"):
    return MarketEvent(symbol=symbol, venue="XNYS",
                       event_type=EventType.TRADE, ts_ns=ts, price=D(price))


def at(day, hh, mm):
    """A UTC nanosecond stamp for a New York wall-clock time in March 2026."""
    from datetime import datetime, timedelta, timezone
    dt = datetime(2026, 3, day, hh, mm,
                  tzinfo=timezone(timedelta(hours=-5 if day < 8 else -4)))
    return int(dt.timestamp()) * S


# -- finding ---------------------------------------------------------------

def test_gaps_are_found_between_consecutive_events():
    got = find_gaps([ev(0), ev(M), ev(20 * M), ev(21 * M)], min_gap_ns=5 * M)
    assert [(g.start_ns, g.end_ns) for g in got] == [(M, 20 * M)]


def test_the_threshold_selects_which_silences_count():
    events = [ev(0), ev(3 * M), ev(20 * M)]
    assert len(find_gaps(events, min_gap_ns=2 * M)) == 2
    assert len(find_gaps(events, min_gap_ns=5 * M)) == 1
    assert len(find_gaps(events, min_gap_ns=30 * M)) == 0


def test_a_gap_exactly_at_the_threshold_counts():
    assert len(find_gaps([ev(0), ev(5 * M)], min_gap_ns=5 * M)) == 1


def test_the_threshold_is_required_to_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        find_gaps([ev(0), ev(M)], min_gap_ns=0)


def test_gaps_are_found_per_symbol_not_across_the_tape():
    """Two names alternating are not a name with no gaps."""
    events = [ev(0, "AAA"), ev(M, "BBB"), ev(20 * M, "AAA"), ev(21 * M, "BBB")]
    got = find_gaps(events, min_gap_ns=5 * M)
    assert [g.symbol for g in got] == ["AAA", "BBB"]


def test_events_out_of_order_are_sorted_before_measuring():
    got = find_gaps([ev(20 * M), ev(0), ev(M)], min_gap_ns=5 * M)
    assert [(g.start_ns, g.end_ns) for g in got] == [(M, 20 * M)]


def test_a_single_event_has_no_gap_to_measure():
    assert find_gaps([ev(0)], min_gap_ns=M) == []
    assert find_gaps([], min_gap_ns=M) == []


def test_a_gap_ends_after_it_starts():
    with pytest.raises(ValueError, match="ends after it starts"):
        Gap("AAA", 5 * M, 5 * M)


# -- explaining ------------------------------------------------------------

def test_without_a_calendar_every_instant_is_trading_time():
    [g] = explain_gaps(find_gaps([ev(at(4, 16, 0)), ev(at(5, 9, 30))],
                                 min_gap_ns=H))
    assert g.outside_session_ns == 0
    assert g.unexplained_ns == g.duration_ns
    assert not g.is_explained


def test_an_overnight_gap_is_fully_explained_by_the_calendar():
    [g] = explain_gaps(find_gaps([ev(at(4, 16, 0)), ev(at(5, 9, 30))],
                                 min_gap_ns=H), calendar=CAL)
    assert g.unexplained_ns == 0
    assert g.outside_session_ns == g.duration_ns
    assert g.is_explained


def test_a_weekend_is_explained_too():
    # Friday 6 March close to Monday 9 March open
    [g] = explain_gaps(find_gaps([ev(at(6, 16, 0)), ev(at(9, 9, 30))],
                                 min_gap_ns=H), calendar=CAL)
    assert g.duration_ns > 2 * DAY
    assert g.unexplained_ns == 0


def test_a_holiday_inside_the_gap_costs_nothing_unexplained():
    [g] = explain_gaps(find_gaps([ev(at(9, 16, 0)), ev(at(11, 9, 30))],
                                 min_gap_ns=H), calendar=CAL)
    assert g.unexplained_ns == 0          # 10 March did not trade at all


def test_an_early_close_leaves_the_afternoon_explained():
    """11 March closed at 13:00; the missing afternoon is not missing data."""
    [g] = explain_gaps(find_gaps([ev(at(11, 12, 55)), ev(at(12, 9, 30))],
                                 min_gap_ns=H), calendar=CAL)
    assert g.unexplained_ns == 5 * M      # only 12:55 to 13:00 was open


def test_an_intraday_gap_is_unexplained_by_default():
    [g] = explain_gaps(find_gaps([ev(at(12, 10, 0)), ev(at(12, 12, 0))],
                                 min_gap_ns=H), calendar=CAL)
    assert g.unexplained_ns == 2 * H
    assert g.outside_session_ns == 0


def test_a_halt_accounts_for_the_part_it_covers():
    halts = [Halt("AAA", at(12, 10, 30), at(12, 11, 30), HaltKind.VOLATILITY)]
    [g] = explain_gaps(find_gaps([ev(at(12, 10, 0)), ev(at(12, 12, 0))],
                                 min_gap_ns=H), calendar=CAL, halts=halts)
    assert g.halt_ns == H
    assert g.unexplained_ns == H
    assert g.explained_ns == H


def test_a_halt_on_another_symbol_explains_nothing_here():
    halts = [Halt("BBB", at(12, 10, 30), at(12, 11, 30))]
    [g] = explain_gaps(find_gaps([ev(at(12, 10, 0)), ev(at(12, 12, 0))],
                                 min_gap_ns=H), calendar=CAL, halts=halts)
    assert g.halt_ns == 0
    assert g.unexplained_ns == 2 * H


def test_a_halt_outside_the_session_is_not_double_counted():
    """Overnight is already explained; a halt across it adds nothing."""
    halts = [Halt("AAA", at(11, 17, 0), at(11, 18, 0))]
    [g] = explain_gaps(find_gaps([ev(at(11, 12, 59)), ev(at(12, 9, 30))],
                                 min_gap_ns=H), calendar=CAL, halts=halts)
    assert g.halt_ns == 0
    assert g.outside_session_ns + g.halt_ns + g.unexplained_ns == g.duration_ns


def test_the_three_components_always_sum_to_the_duration():
    halts = [Halt("AAA", at(12, 10, 30), at(12, 11, 30))]
    events = [ev(at(11, 12, 0)), ev(at(12, 10, 0)), ev(at(12, 12, 0)),
              ev(at(13, 15, 0))]
    for g in explain_gaps(find_gaps(events, min_gap_ns=M),
                          calendar=CAL, halts=halts):
        assert (g.outside_session_ns + g.halt_ns + g.unexplained_ns
                == g.duration_ns)


def test_a_gap_running_past_the_calendar_range_counts_as_closed():
    """A calendar that says nothing about a date cannot be read as open."""
    cal = TradingCalendar(SESSION, first_day=date(2026, 3, 1),
                          last_day=date(2026, 3, 12), name="short")
    [g] = explain_gaps(find_gaps([ev(at(12, 15, 0)), ev(at(16, 10, 0))],
                                 min_gap_ns=H), calendar=cal)
    assert g.unexplained_ns == H          # only 15:00-16:00 on the 12th


# -- the report ------------------------------------------------------------

def test_the_report_totals_the_three_buckets():
    halts = [Halt("AAA", at(12, 10, 30), at(12, 11, 30))]
    events = [ev(at(11, 12, 0)), ev(at(12, 10, 0)), ev(at(12, 12, 0))]
    r = coverage_report(events, min_gap_ns=M, calendar=CAL, halts=halts)
    assert r.symbols == 1 and r.events == 3
    assert r.gaps == 2
    assert r.halt_ns == H
    # 11 March 12:00-13:00 (it closed early) + 12 March 09:30-10:00 = 1h30m,
    # then 10:00-12:00 less the halted hour = 1h.
    assert r.unexplained_ns == 2 * H + 30 * M
    assert r.longest_unexplained_ns == H + 30 * M
    assert r.gap_ns == (r.outside_session_ns + r.halt_ns + r.unexplained_ns)


def test_the_report_records_whether_a_calendar_was_used():
    events = [ev(at(4, 16, 0)), ev(at(5, 9, 30))]
    assert coverage_report(events, min_gap_ns=H).calendar is False
    assert coverage_report(events, min_gap_ns=H, calendar=CAL).calendar is True


def test_the_same_data_reads_very_differently_without_a_calendar():
    events = [ev(at(4, 16, 0)), ev(at(5, 9, 30))]
    blind = coverage_report(events, min_gap_ns=H)
    seeing = coverage_report(events, min_gap_ns=H, calendar=CAL)
    assert blind.unexplained_ns > 0
    assert seeing.unexplained_ns == 0
    assert blind.gap_ns == seeing.gap_ns          # the silence is the same


def test_the_longest_unexplained_stretch_is_reported():
    events = [ev(at(12, 10, 0)), ev(at(12, 11, 0)),
              ev(at(12, 13, 0)), ev(at(12, 15, 30))]
    r = coverage_report(events, min_gap_ns=30 * M, calendar=CAL)
    assert r.longest_unexplained_ns == 2 * H + 30 * M


def test_span_is_summed_over_symbols():
    events = [ev(at(12, 10, 0), "AAA"), ev(at(12, 15, 0), "AAA"),
              ev(at(12, 10, 0), "BBB"), ev(at(12, 15, 0), "BBB")]
    r = coverage_report(events, min_gap_ns=H, calendar=CAL)
    assert r.symbols == 2
    assert r.span_ns == 10 * H


def test_shares_are_none_rather_than_zero_when_undefined():
    r = coverage_report([], min_gap_ns=M)
    assert r.explained_share is None
    assert r.unexplained_share is None
    r = coverage_report([ev(0), ev(M)], min_gap_ns=H)
    assert r.gaps == 0
    assert r.explained_share is None          # no gap time to divide


def test_the_explained_share_is_a_share_of_the_gaps():
    halts = [Halt("AAA", at(12, 10, 0), at(12, 11, 0))]
    r = coverage_report([ev(at(12, 9, 59)), ev(at(12, 12, 0))],
                        min_gap_ns=M, calendar=CAL, halts=halts)
    assert r.explained_share == Decimal(r.explained_ns) / r.gap_ns
    assert 0 < r.explained_share < 1


# -- the panel -------------------------------------------------------------

def test_the_panel_counts_symbols_present_in_each_bucket():
    events = [ev(0, "AAA"), ev(0, "BBB"), ev(0, "CCC"),
              ev(DAY, "AAA"), ev(DAY, "BBB"),
              ev(2 * DAY, "AAA")]
    p = panel_coverage(events, start_ns=0, end_ns=3 * DAY, step_ns=DAY)
    assert p.points == 3 and p.symbols == 3
    assert p.counts == (3, 2, 1)
    assert p.widest == 3 and p.narrowest == 1 and p.median == 2


def test_full_width_points_are_counted():
    events = [ev(0, "AAA"), ev(0, "BBB"), ev(DAY, "AAA")]
    p = panel_coverage(events, start_ns=0, end_ns=2 * DAY, step_ns=DAY)
    assert p.full == 1
    assert p.full_share == D(1) / 2


def test_a_stated_universe_sees_a_name_that_never_printed():
    """The one worth knowing about is the one absent from the whole sample."""
    events = [ev(0, "AAA"), ev(DAY, "AAA")]
    p = panel_coverage(events, start_ns=0, end_ns=2 * DAY, step_ns=DAY,
                       symbols=["AAA", "BBB"])
    assert p.symbols == 2
    assert p.counts == (1, 1)
    assert p.full == 0


def test_events_outside_the_grid_are_ignored():
    base = 100 * DAY
    events = [ev(base - DAY, "AAA"), ev(base, "AAA"), ev(base + 5 * DAY, "AAA")]
    p = panel_coverage(events, start_ns=base, end_ns=base + 2 * DAY,
                       step_ns=DAY)
    assert p.counts == (1, 0)


def test_a_symbol_outside_a_stated_universe_is_ignored():
    events = [ev(0, "AAA"), ev(0, "ZZZ")]
    p = panel_coverage(events, start_ns=0, end_ns=DAY, step_ns=DAY,
                       symbols=["AAA"])
    assert p.counts == (1,)


def test_many_prints_in_one_bucket_still_count_once():
    events = [ev(i * 100, "AAA") for i in range(50)]
    p = panel_coverage(events, start_ns=0, end_ns=DAY, step_ns=DAY)
    assert p.counts == (1,)


def test_a_partial_final_bucket_is_still_a_point():
    p = panel_coverage([ev(0, "AAA")], start_ns=0, end_ns=DAY + H,
                       step_ns=DAY)
    assert p.points == 2


def test_the_grid_is_validated():
    with pytest.raises(ValueError, match="step_ns must be positive"):
        panel_coverage([ev(0)], start_ns=0, end_ns=DAY, step_ns=0)
    with pytest.raises(ValueError, match="must follow"):
        panel_coverage([ev(0)], start_ns=DAY, end_ns=0, step_ns=H)


def test_an_empty_panel_reports_none_rather_than_zero():
    p = panel_coverage([], start_ns=0, end_ns=DAY, step_ns=DAY,
                       symbols=[])
    assert p.narrowest == 0 and p.median == 0
    assert p.full_share == 1              # vacuously, and points is 1
    p2 = PanelCoverage(points=0, symbols=0, counts=(), step_ns=DAY)
    assert p2.full_share is None and p2.median is None


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Gap("AAA", 0, 1), CoverageReport(0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                PanelCoverage(0, 0, (), 1)):
        with pytest.raises(Exception):
            obj.symbols = 5  # type: ignore[misc]


# -- the sample bounds -----------------------------------------------------

def test_without_bounds_a_feed_that_stops_produces_no_gap_at_all():
    """The last observation has nothing after it to be distant from."""
    events = [ev(at(12, 10, 0)), ev(at(12, 10, 0) + S)]
    assert find_gaps(events, min_gap_ns=M) == []


def test_with_bounds_the_silence_after_the_last_print_is_a_gap():
    events = [ev(at(12, 10, 0)), ev(at(12, 10, 0) + S)]
    got = find_gaps(events, min_gap_ns=M,
                    start_ns=at(12, 9, 30), end_ns=at(12, 16, 0))
    assert len(got) == 2                      # the head silence and the tail
    assert got[-1].start_ns == at(12, 10, 0) + S
    assert got[-1].end_ns == at(12, 16, 0)


def test_the_silence_before_the_first_print_is_a_gap_too():
    events = [ev(at(12, 15, 0))]
    got = find_gaps(events, min_gap_ns=M,
                    start_ns=at(12, 9, 30), end_ns=at(12, 16, 0))
    assert [(g.start_ns, g.end_ns) for g in got] == [
        (at(12, 9, 30), at(12, 15, 0)), (at(12, 15, 0), at(12, 16, 0))]


def test_a_symbol_that_never_printed_is_one_gap_the_whole_width():
    events = [ev(at(12, 10, 0), "AAA"), ev(at(12, 15, 0), "AAA")]
    got = find_gaps(events, min_gap_ns=M, start_ns=at(12, 9, 30),
                    end_ns=at(12, 16, 0), symbols=["AAA", "BBB"])
    [bbb] = [g for g in got if g.symbol == "BBB"]
    assert bbb.start_ns == at(12, 9, 30) and bbb.end_ns == at(12, 16, 0)


def test_events_outside_the_stated_bounds_are_not_counted():
    events = [ev(at(11, 10, 0)), ev(at(12, 12, 0)), ev(at(13, 10, 0))]
    got = find_gaps(events, min_gap_ns=M, start_ns=at(12, 9, 30),
                    end_ns=at(12, 16, 0))
    assert all(at(12, 9, 30) <= g.start_ns <= at(12, 16, 0) for g in got)


def test_a_feed_that_stops_mid_session_is_caught_by_the_report():
    """Three hours of a session with nothing in it, and a calendar to prove it."""
    events = [ev(at(12, 9, 30) + i * M) for i in range(120)]   # last at 11:29
    r = coverage_report(events, min_gap_ns=5 * M, calendar=CAL,
                        start_ns=at(12, 9, 30), end_ns=at(12, 16, 0))
    assert r.unexplained_ns == 4 * H + 31 * M
    assert r.span_ns == 6 * H + 30 * M


def test_bounds_are_validated():
    with pytest.raises(ValueError, match="must follow"):
        find_gaps([ev(0)], min_gap_ns=M, start_ns=2 * M, end_ns=M)
