"""Halt tests: the silence, the gap across it, and the fills invented in it."""
from decimal import Decimal

import pytest

from mdnorm.halts import (
    Decision,
    Halt,
    HaltKind,
    HaltReport,
    ReopenGap,
    Unfillable,
    exclude_halted,
    halt_report,
    halted,
    read_halts_csv,
    reopen_gaps,
    split_halted,
    unfillable,
)
from mdnorm.schema import EventType, MarketEvent

D = Decimal
S = 1_000_000_000          # one second in nanoseconds
M = 60 * S


def trade(ts, price, symbol="AAA", size=100):
    return MarketEvent(symbol=symbol, venue="XNYS",
                       event_type=EventType.TRADE, ts_ns=ts,
                       price=D(str(price)), size=D(size))


def quote(ts, symbol="AAA"):
    return MarketEvent(symbol=symbol, venue="XNYS",
                       event_type=EventType.QUOTE, ts_ns=ts,
                       bid_price=D(10), ask_price=D(11))


PAUSE = Halt("AAA", 10 * M, 20 * M, HaltKind.VOLATILITY, "LULD")


# -- the window ------------------------------------------------------------

def test_a_halt_is_half_open_at_both_ends():
    assert 10 * M in PAUSE                 # the instant it begins is halted
    assert 20 * M - 1 in PAUSE
    assert 20 * M not in PAUSE             # the reopen is tradable
    assert 10 * M - 1 not in PAUSE


def test_a_halt_ends_after_it_starts():
    with pytest.raises(ValueError, match="ends after it starts"):
        Halt("AAA", 5 * M, 5 * M)
    with pytest.raises(ValueError, match="ends after it starts"):
        Halt("AAA", 5 * M, M)


def test_duration_is_the_window():
    assert PAUSE.duration_ns == 10 * M


def test_halted_matches_on_symbol_as_well_as_time():
    assert halted(15 * M, "AAA", [PAUSE]) is PAUSE
    assert halted(15 * M, "BBB", [PAUSE]) is None
    assert halted(25 * M, "AAA", [PAUSE]) is None


def test_a_symbol_with_no_halt_records_is_not_halted():
    """Which is not the same as a symbol whose halts were never loaded."""
    assert halted(15 * M, "CCC", [PAUSE]) is None


# -- splitting -------------------------------------------------------------

def test_the_split_hands_back_both_halves():
    events = [trade(5 * M, 100), trade(12 * M, 100), trade(25 * M, 90)]
    tradable, during = split_halted(events, [PAUSE])
    assert [e.ts_ns for e in tradable] == [5 * M, 25 * M]
    assert [e.ts_ns for e in during] == [12 * M]
    assert len(tradable) + len(during) == len(events)


def test_the_split_preserves_input_order():
    events = [trade(t, 100) for t in (25 * M, 5 * M, 12 * M, 30 * M)]
    tradable, _ = split_halted(events, [PAUSE])
    assert [e.ts_ns for e in tradable] == [25 * M, 5 * M, 30 * M]


def test_quotes_inside_a_halt_are_held_out_too():
    """An indicative quote during a pause is not a price you can hit."""
    _, during = split_halted([quote(12 * M)], [PAUSE])
    assert len(during) == 1


def test_another_symbol_is_untouched_by_this_halt():
    events = [trade(12 * M, 100, "AAA"), trade(12 * M, 50, "BBB")]
    tradable, during = split_halted(events, [PAUSE])
    assert [e.symbol for e in tradable] == ["BBB"]
    assert [e.symbol for e in during] == ["AAA"]


def test_exclude_returns_the_tradable_half():
    events = [trade(5 * M, 100), trade(12 * M, 100)]
    assert exclude_halted(events, [PAUSE]) == [events[0]]


def test_no_halts_leaves_everything_tradable():
    events = [trade(t, 100) for t in (5 * M, 12 * M, 25 * M)]
    tradable, during = split_halted(events, [])
    assert tradable == events and during == []


# -- the report ------------------------------------------------------------

def test_the_report_counts_time_and_prints():
    events = [trade(0, 100), trade(12 * M, 100), trade(30 * M, 90)]
    r = halt_report(events, [PAUSE])
    assert r.halts == 1 and r.symbols == 1
    assert r.halted_ns == 10 * M
    assert r.covered_ns == 30 * M
    assert r.halted_share == D(1) / 3
    assert r.events == 3 and r.events_during == 1
    assert r.during_share == D(1) / 3


def test_covered_time_is_the_extent_of_the_data_not_a_session_length():
    """A file covering ten minutes is not a file covering a day."""
    events = [trade(0, 100), trade(10 * M, 100)]
    assert halt_report(events, []).covered_ns == 10 * M


def test_halted_time_is_summed_over_symbols():
    """Two names paused for the same hour is two instrument-hours."""
    halts = [Halt("AAA", 10 * M, 20 * M), Halt("BBB", 10 * M, 20 * M)]
    events = [trade(0, 100, "AAA"), trade(30 * M, 100, "AAA"),
              trade(0, 50, "BBB"), trade(30 * M, 50, "BBB")]
    r = halt_report(events, halts)
    assert r.symbols == 2
    assert r.halted_ns == 20 * M
    assert r.covered_ns == 60 * M
    assert r.halted_share == D(1) / 3


def test_the_longest_halt_is_reported():
    halts = [Halt("AAA", 0, 5 * M), Halt("AAA", 10 * M, 30 * M)]
    assert halt_report([trade(0, 100)], halts).longest_ns == 20 * M


def test_a_report_with_no_halts_says_so_rather_than_guessing():
    r = halt_report([trade(0, 100), trade(M, 100)], [])
    assert r.halts == 0 and r.halted_ns == 0
    assert r.halted_share == 0
    assert r.events_during == 0
    assert r.longest_ns == 0


def test_an_empty_input_reports_no_shares_rather_than_zero():
    r = halt_report([], [])
    assert r.halted_share is None
    assert r.during_share is None


def test_a_single_event_covers_no_time():
    assert halt_report([trade(0, 100)], []).halted_share is None


# -- the reopening gap -----------------------------------------------------

def test_the_gap_is_the_last_price_before_and_the_first_after():
    events = [trade(0, 100), trade(9 * M, D("41.20")),
              trade(20 * M, D("33.60")), trade(25 * M, D("34.00"))]
    gap = reopen_gaps(events, [PAUSE])[0]
    assert gap.last_before == D("41.20")
    assert gap.first_after == D("33.60")
    assert gap.move == D("-7.60")
    assert round(gap.move_bps, 2) == D("-1844.66")


def test_a_print_inside_the_halt_is_not_the_reopening_price():
    """A late report of a pre-halt execution would understate the gap."""
    events = [trade(9 * M, D("41.20")), trade(12 * M, D("41.15")),
              trade(20 * M, D("33.60"))]
    gap = reopen_gaps(events, [PAUSE])[0]
    assert gap.last_before == D("41.20")
    assert gap.first_after == D("33.60")


def test_the_reopening_print_is_the_one_exactly_at_the_reopen():
    events = [trade(9 * M, D(40)), trade(20 * M, D(30)), trade(21 * M, D(31))]
    assert reopen_gaps(events, [PAUSE])[0].first_after == D(30)


def test_a_halt_that_never_reopened_in_this_file_reports_no_move():
    gap = reopen_gaps([trade(9 * M, D(40))], [PAUSE])[0]
    assert gap.last_before == D(40)
    assert gap.first_after is None
    assert gap.move is None and gap.move_bps is None


def test_a_halt_at_the_very_start_has_nothing_before_it():
    gap = reopen_gaps([trade(20 * M, D(30))], [PAUSE])[0]
    assert gap.last_before is None
    assert gap.move is None


def test_gaps_only_use_prices_from_their_own_symbol():
    events = [trade(9 * M, D(40), "AAA"), trade(9 * M, D(5), "BBB"),
              trade(20 * M, D(30), "AAA"), trade(20 * M, D(6), "BBB")]
    gap = reopen_gaps(events, [PAUSE])[0]
    assert gap.last_before == D(40) and gap.first_after == D(30)


def test_quotes_are_not_prices_for_this_purpose():
    gap = reopen_gaps([quote(9 * M), trade(20 * M, D(30))], [PAUSE])[0]
    assert gap.last_before is None


def test_one_gap_per_halt_in_a_stable_order():
    halts = [Halt("BBB", 0, M), Halt("AAA", 30 * M, 40 * M),
             Halt("AAA", 10 * M, 20 * M)]
    got = reopen_gaps([trade(0, 100)], halts)
    assert [(g.halt.symbol, g.halt.start_ns) for g in got] == [
        ("AAA", 10 * M), ("AAA", 30 * M), ("BBB", 0)]


# -- what a strategy did in the silence ------------------------------------

def test_decisions_inside_a_halt_are_counted_and_weighted():
    decisions = [Decision(5 * M, "AAA", D(100)),
                 Decision(12 * M, "AAA", D(900)),
                 Decision(25 * M, "AAA", D(100))]
    u = unfillable(decisions, [PAUSE])
    assert u.decisions == 3 and u.unfillable == 1
    assert u.count_share == D(1) / 3
    assert u.value == D(1100) and u.unfillable_value == D(900)
    assert u.value_share == D(900) / 1100


def test_the_value_share_exceeds_the_count_share_when_the_big_ones_were_stale():
    """Which is the whole point: one fill in three, nine tenths of the money."""
    u = unfillable([Decision(5 * M, "AAA", D(50)),
                    Decision(12 * M, "AAA", D(900)),
                    Decision(25 * M, "AAA", D(50))], [PAUSE])
    assert u.value_share > u.count_share


def test_a_short_and_a_long_do_not_cancel_into_a_reassuring_zero():
    u = unfillable([Decision(12 * M, "AAA", D(-500)),
                    Decision(13 * M, "AAA", D(500))], [PAUSE])
    assert u.value == D(1000)
    assert u.unfillable_value == D(1000)
    assert u.value_share == 1


def test_symbols_with_no_halt_records_are_named_rather_than_assumed_clean():
    u = unfillable([Decision(12 * M, "AAA", D(1)),
                    Decision(12 * M, "ZZZ", D(1)),
                    Decision(12 * M, "YYY", D(1))], [PAUSE])
    assert u.unfillable == 1
    assert u.unmatched_symbols == ("YYY", "ZZZ")


def test_a_clean_strategy_reports_nothing_unfillable():
    u = unfillable([Decision(5 * M, "AAA", D(10)),
                    Decision(25 * M, "AAA", D(10))], [PAUSE])
    assert u.unfillable == 0
    assert u.count_share == 0 and u.value_share == 0


def test_valueless_decisions_report_a_count_but_no_value_share():
    u = unfillable([Decision(12 * M, "AAA"), Decision(25 * M, "AAA")],
                   [PAUSE])
    assert u.count_share == D("0.5")
    assert u.value_share is None


def test_no_decisions_reports_no_shares():
    u = unfillable([], [PAUSE])
    assert u.count_share is None and u.value_share is None


# -- reading -------------------------------------------------------------

def test_halts_are_read_from_csv(tmp_path):
    p = tmp_path / "halts.csv"
    p.write_text("symbol,start,end,kind,reason\n"
                 f"AAA,{10 * M},{20 * M},volatility,LULD\n"
                 f"BBB,{5 * M},{6 * M},regulatory,news pending\n")
    got = read_halts_csv(str(p))
    assert [h.symbol for h in got] == ["AAA", "BBB"]
    assert got[0].kind is HaltKind.VOLATILITY
    assert got[1].reason == "news pending"


def test_an_unrecognised_kind_becomes_unknown_rather_than_an_error(tmp_path):
    """A vendor inventing a code should not stop a pipeline that wants the window."""
    p = tmp_path / "halts.csv"
    p.write_text(f"symbol,start,end,kind\nAAA,{M},{2 * M},T12\n")
    assert read_halts_csv(str(p))[0].kind is HaltKind.UNKNOWN


def test_kind_and_reason_are_optional(tmp_path):
    p = tmp_path / "halts.csv"
    p.write_text(f"symbol,start,end\nAAA,{M},{2 * M}\n")
    got = read_halts_csv(str(p))
    assert got[0].kind is HaltKind.UNKNOWN and got[0].reason == ""


def test_a_bad_row_names_its_line(tmp_path):
    p = tmp_path / "halts.csv"
    p.write_text(f"symbol,start,end\nAAA,{M},{2 * M}\nBBB,x,{M}\n")
    with pytest.raises(ValueError, match="line 3"):
        read_halts_csv(str(p))


def test_an_empty_file_is_not_a_day_without_halts(tmp_path):
    p = tmp_path / "halts.csv"
    p.write_text("symbol,start,end\n")
    with pytest.raises(ValueError, match="no halts"):
        read_halts_csv(str(p))


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Halt("AAA", 0, 1), Decision(0, "AAA"),
                ReopenGap(PAUSE, None, None),
                HaltReport(0, 0, 0, 0, 0, 0, 0),
                Unfillable(0, 0, D(0), D(0))):
        with pytest.raises(Exception):
            obj.symbol = "ZZZ"  # type: ignore[misc]
