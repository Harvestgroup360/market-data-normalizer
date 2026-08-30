"""Tick-grid tests: bands, ties, rounding against yourself, and detection."""
from decimal import Decimal

import pytest

from mdnorm import (
    GridReport,
    Rounding,
    Side,
    TickBand,
    TickSchedule,
    TickTable,
    grid_report,
    read_tick_table_csv,
    spread_in_ticks,
)

D = Decimal
NS = 1_000_000_000


def table():
    """Sub-dollar in hundredths of a cent, above a dollar in cents."""
    return TickTable([TickBand(D("0"), D("0.0001")),
                      TickBand(D("1"), D("0.01"))], name="two-band")


# -- bands ------------------------------------------------------------------


def test_the_band_decides_the_tick():
    t = table()
    assert t.tick_at(D("0.5000")) == D("0.0001")
    assert t.tick_at(D("1.00")) == D("0.01")       # the boundary is inclusive
    assert t.tick_at(D("4200.00")) == D("0.01")


def test_a_price_below_the_table_is_refused_not_guessed():
    t = TickTable([TickBand(D("1"), D("0.01"))])
    with pytest.raises(ValueError) as exc:
        t.tick_at(D("0.50"))
    assert "below this tick table" in str(exc.value)


def test_bands_are_sorted_however_they_arrive():
    t = TickTable([TickBand(D("1"), D("0.01")),
                   TickBand(D("0"), D("0.0001"))])
    assert [b.min_price for b in t.bands] == [D("0"), D("1")]
    assert t.floor == 0


def test_an_empty_table_is_rejected():
    with pytest.raises(ValueError):
        TickTable([])


def test_two_bands_cannot_start_together():
    with pytest.raises(ValueError):
        TickTable([TickBand(D("1"), D("0.01")), TickBand(D("1"), D("0.05"))])


def test_a_band_needs_a_positive_tick():
    with pytest.raises(ValueError):
        TickBand(D("1"), D("0"))
    with pytest.raises(ValueError):
        TickBand(D("-1"), D("0.01"))


# -- on the grid ------------------------------------------------------------


def test_on_grid_and_offset():
    t = table()
    assert t.on_grid(D("42.30")) is True
    assert t.on_grid(D("42.305")) is False
    assert t.offset(D("42.305")) == D("0.005")
    assert t.offset(D("42.30")) == 0


def test_a_price_carrying_extra_zeros_is_still_on_the_grid():
    assert table().on_grid(D("42.3000")) is True


# -- rounding ---------------------------------------------------------------


def test_rounding_needs_a_stated_mode():
    with pytest.raises(TypeError):
        table().round(D("42.305"), "nearest")


def test_down_and_up():
    t = table()
    assert t.round(D("42.307"), Rounding.DOWN) == D("42.30")
    assert t.round(D("42.301"), Rounding.UP) == D("42.31")


def test_a_price_already_on_the_grid_is_untouched():
    t = table()
    for mode in Rounding:
        assert t.round(D("42.30"), mode) == D("42.30")


def test_nearest_goes_to_the_closer_tick_whichever_tie_rule():
    t = table()
    for mode in (Rounding.NEAREST_DOWN, Rounding.NEAREST_UP):
        assert t.round(D("42.301"), mode) == D("42.30")
        assert t.round(D("42.309"), mode) == D("42.31")


def test_an_exact_half_tick_is_where_the_two_rules_part():
    """A mid between adjacent ticks is a half-tick every time, not rarely."""
    t = table()
    mid = D("42.305")
    assert t.round(mid, Rounding.NEAREST_DOWN) == D("42.30")
    assert t.round(mid, Rounding.NEAREST_UP) == D("42.31")


def test_executable_rounds_against_the_caller():
    t = table()
    assert t.executable(D("42.307"), Side.BUY) == D("42.30")    # not 42.31
    assert t.executable(D("42.301"), Side.SELL) == D("42.31")   # not 42.30


def test_executable_never_improves_on_the_wanted_price():
    t = table()
    for raw in ("42.301", "42.305", "42.309"):
        p = D(raw)
        assert t.executable(p, Side.BUY) <= p
        assert t.executable(p, Side.SELL) >= p


# -- distance ---------------------------------------------------------------


def test_ticks_between_within_one_band():
    assert table().ticks_between(D("42.30"), D("42.35")) == 5


def test_ticks_between_is_refused_across_a_band_boundary():
    with pytest.raises(ValueError) as exc:
        table().ticks_between(D("0.9999"), D("1.05"))
    assert "across a band boundary" in str(exc.value)


def test_ticks_between_rejects_an_inverted_pair():
    with pytest.raises(ValueError):
        table().ticks_between(D("42.35"), D("42.30"))


def test_spread_in_ticks():
    t = table()
    assert spread_in_ticks(D("42.30"), D("42.31"), t) == 1
    assert spread_in_ticks(D("42.30"), D("42.34"), t) == 4


def test_a_spread_narrower_than_a_tick_is_reported_not_hidden():
    """Below one tick the two sides did not come from the same place."""
    assert spread_in_ticks(D("42.300"), D("42.305"), table()) == D("0.5")


def test_a_crossed_quote_is_an_error():
    with pytest.raises(ValueError):
        spread_in_ticks(D("42.31"), D("42.30"), table())


# -- the report -------------------------------------------------------------


def test_raw_prints_sit_on_the_grid():
    r = grid_report([D("42.30"), D("42.31"), D("42.29")], table())
    assert r.total == 3 and r.on_grid == 3 and r.off_grid == 0
    assert r.looks_raw is True
    assert r.share_on_grid == 1
    assert r.worst_offset is None


def test_a_back_adjusted_history_does_not():
    """Dividing by a split factor leaves prices the venue never accepted.

    Not every one of them: 498 divided by three is 166 exactly, and lands
    back on the grid by luck. One survivor does not make the series raw,
    which is why `looks_raw` needs every price rather than most of them.
    """
    raw = [D("500.00"), D("502.00"), D("498.00")]
    adjusted = [p / 3 for p in raw]
    r = grid_report(adjusted, table())
    assert r.off_grid == 2 and r.on_grid == 1
    assert r.looks_raw is False
    assert r.example is not None


def test_a_mid_is_off_the_grid_by_half_a_tick():
    r = grid_report([D("42.305")], table())
    assert r.off_grid == 1
    assert r.worst_offset == D("0.005")


def test_prices_below_the_table_are_counted_apart_from_failures():
    t = TickTable([TickBand(D("1"), D("0.01"))])
    r = grid_report([D("42.30"), D("0.5"), D("0.25")], t)
    assert r.on_grid == 1 and r.off_grid == 0 and r.below_table == 2
    assert r.total == 3
    assert r.looks_raw is True          # of what it could judge


def test_a_report_with_nothing_to_judge_says_so():
    t = TickTable([TickBand(D("1"), D("0.01"))])
    r = grid_report([D("0.5")], t)
    assert r.share_on_grid is None and r.looks_raw is None


def test_an_empty_report():
    r = grid_report([], table())
    assert r.total == 0 and r.looks_raw is None


def test_share_on_grid_is_computed_over_what_was_judged():
    r = GridReport(total=10, on_grid=3, off_grid=1, below_table=6,
                   worst_offset=None, example=None)
    assert r.share_on_grid == D("0.75")      # not 0.3


# -- the schedule -----------------------------------------------------------


def schedule():
    old = TickTable([TickBand(D("0"), D("0.05"))], name="old")
    new = TickTable([TickBand(D("0"), D("0.01"))], name="new")
    return TickSchedule([(0, old), (1000 * NS, new)])


def test_the_grid_in_force_depends_on_when_you_ask():
    s = schedule()
    assert s.at(0).name == "old"
    assert s.at(999 * NS).name == "old"
    assert s.at(1000 * NS).name == "new"
    assert len(s) == 2


def test_before_the_first_table_the_schedule_refuses():
    s = TickSchedule([(1000 * NS, table())])
    with pytest.raises(ValueError) as exc:
        s.at(0)
    assert "precedes the first tick table" in str(exc.value)


def test_a_price_can_be_on_one_grid_and_off_the_other():
    s = schedule()
    p = D("42.31")
    assert s.at(0).on_grid(p) is False           # five-cent grid
    assert s.at(1000 * NS).on_grid(p) is True    # one-cent grid


def test_an_empty_schedule_is_rejected():
    with pytest.raises(ValueError):
        TickSchedule([])


def test_two_tables_cannot_take_effect_together():
    with pytest.raises(ValueError):
        TickSchedule([(0, table()), (0, table())])


def test_a_negative_effective_time_is_rejected():
    with pytest.raises(ValueError):
        TickSchedule([(-1, table())])


# -- CSV --------------------------------------------------------------------


def test_read_tick_table_csv(tmp_path):
    p = tmp_path / "ticks.csv"
    p.write_text("min_price,tick\n1,0.01\n0,0.0001\n")
    t = read_tick_table_csv(str(p), name="venue")
    assert t.tick_at(D("0.5")) == D("0.0001")
    assert t.tick_at(D("10")) == D("0.01")
    assert t.name == "venue"


def test_read_tick_table_csv_rejects_an_empty_file(tmp_path):
    p = tmp_path / "ticks.csv"
    p.write_text("min_price,tick\n")
    with pytest.raises(ValueError) as exc:
        read_tick_table_csv(str(p))
    assert "cannot be empty" in str(exc.value)


def test_reprs_are_informative():
    assert "TickTable" in repr(table())
    assert "TickSchedule" in repr(schedule())
