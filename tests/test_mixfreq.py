"""Mixed-frequency tests: a slow value read at the right moment, not its label."""
from datetime import date, time
from decimal import Decimal

import pytest

from mdnorm import (
    Bar,
    BarField,
    LeakReport,
    Period,
    PeriodSeries,
    Session,
    US_EQUITY_RTH,
    grid,
    leak_report,
    read_periods_csv,
    session_bounds,
)

D = Decimal
NS = 1_000_000_000
MINUTE = 60 * NS
HOUR = 3600 * NS
DAY = 86_400 * NS


# -- session_bounds ---------------------------------------------------------


def test_session_bounds_new_york_winter_and_summer():
    """The same local close is a different UTC hour on either side of DST."""
    _, jan_close = session_bounds(date(2026, 1, 15), US_EQUITY_RTH)
    _, jul_close = session_bounds(date(2026, 7, 15), US_EQUITY_RTH)
    # 16:00 New York is 21:00 UTC in winter and 20:00 UTC in summer.
    assert jan_close % DAY == 21 * HOUR
    assert jul_close % DAY == 20 * HOUR


def test_session_bounds_open_precedes_close():
    open_ns, close_ns = session_bounds(date(2026, 3, 4), US_EQUITY_RTH)
    assert open_ns < close_ns
    assert close_ns - open_ns == 6 * HOUR + 30 * MINUTE


def test_session_bounds_overnight_closes_next_day():
    overnight = Session(start=time(18, 0), end=time(17, 0), tz="UTC")
    open_ns, close_ns = session_bounds(date(2026, 3, 4), overnight)
    assert close_ns - open_ns == 23 * HOUR


def test_session_bounds_full_day_follows_the_calendar():
    """A 24-hour session on a DST day is not 24 hours long."""
    full = Session(start=time(0, 0), end=time(0, 0), tz="America/New_York",
                   days=frozenset(range(7)))
    normal = session_bounds(date(2026, 6, 1), full)
    spring = session_bounds(date(2026, 3, 8), full)  # clocks go forward
    assert normal[1] - normal[0] == 24 * HOUR
    assert spring[1] - spring[0] == 23 * HOUR


def test_session_bounds_answers_for_a_day_outside_session_days():
    """A Sunday has bounds if you ask for them; guessing would be worse."""
    open_ns, close_ns = session_bounds(date(2026, 3, 8), US_EQUITY_RTH)
    assert open_ns < close_ns


# -- Period -----------------------------------------------------------------


def test_period_rejects_empty_and_negative_spans():
    with pytest.raises(ValueError):
        Period(100, 100, D("1"))
    with pytest.raises(ValueError):
        Period(100, 50, D("1"))
    with pytest.raises(ValueError):
        Period(-1, 100, D("1"))


# -- construction -----------------------------------------------------------


def daily(n=3, *, lag_ns=0):
    """Three consecutive UTC days, values 1, 2, 3."""
    periods = [Period(i * DAY, (i + 1) * DAY, D(i + 1)) for i in range(n)]
    return PeriodSeries(periods, publication_lag_ns=lag_ns, name="slow")


def test_periods_are_sorted_and_deduplicated():
    s = PeriodSeries([
        Period(DAY, 2 * DAY, D("2")),
        Period(0, DAY, D("1")),
        Period(DAY, 2 * DAY, D("9")),  # same span, later input wins
    ])
    assert [p.value for p in s.periods] == [D("1"), D("9")]
    assert len(s) == 2


def test_negative_publication_lag_is_rejected():
    with pytest.raises(ValueError):
        PeriodSeries([], publication_lag_ns=-1)


def test_from_sessions_uses_the_session_close():
    values = {date(2026, 1, 15): D("100"), date(2026, 7, 15): D("200")}
    s = PeriodSeries.from_sessions(values, US_EQUITY_RTH)
    ends = [p.end_ns for p in s.periods]
    assert ends == sorted(ends)
    assert ends[0] == session_bounds(date(2026, 1, 15), US_EQUITY_RTH)[1]


def test_from_daily_bars_ignores_the_bars_own_label():
    """A midnight-stamped daily bar still becomes knowable at the close."""
    day = date(2026, 1, 15)
    open_ns, close_ns = session_bounds(day, US_EQUITY_RTH)
    midnight_bar = Bar(start_ns=open_ns, interval_ns=DAY, open=D("1"),
                       high=D("2"), low=D("1"), close=D("1.5"),
                       volume=D("10"), trades=3)
    s = PeriodSeries.from_daily_bars([midnight_bar], US_EQUITY_RTH)
    assert len(s) == 1
    assert s.periods[0].end_ns == close_ns
    assert s.periods[0].end_ns != midnight_bar.end_ns
    assert s.periods[0].value == D("1.5")


def test_from_daily_bars_skips_a_missing_field():
    bar = Bar(start_ns=0, interval_ns=DAY, open=D("1"), high=D("2"),
              low=D("1"), close=D("1.5"), volume=D("10"), trades=3,
              vwap=None)
    s = PeriodSeries.from_daily_bars([bar], US_EQUITY_RTH,
                                     field=BarField.VWAP)
    assert len(s) == 0


# -- the two keys -----------------------------------------------------------


def test_value_is_unavailable_until_its_period_closes():
    s = daily()
    assert s.at(0)[0] is None                 # first period has not closed
    assert s.at(DAY - 1)[0] is None           # one nanosecond before close
    assert s.at(DAY)[0] == D("1")             # the instant it closes


def test_publication_lag_delays_readability_further():
    s = daily(lag_ns=2 * HOUR)
    assert s.at(DAY)[0] is None
    assert s.at(DAY + 2 * HOUR - 1)[0] is None
    assert s.at(DAY + 2 * HOUR)[0] == D("1")


def test_labelled_series_is_the_join_that_leaks():
    s = daily()
    honest = s.knowable_series()
    naive = s.labelled_series()
    # Midway through the first day the naive join already shows that day's
    # value; the honest one shows nothing at all.
    t = DAY // 2
    assert naive.at(t)[0] == D("1")
    assert honest.at(t)[0] is None


def test_knowable_series_carries_the_lag():
    s = daily(lag_ns=HOUR)
    assert s.knowable_series().first_ts_ns == DAY + HOUR


def test_at_reports_staleness_separately_from_absence():
    s = daily()
    assert s.at(DAY // 2) == (None, None)          # nothing published yet
    value, age = s.at(2 * DAY + HOUR, max_age_ns=MINUTE)
    assert value is None and age == HOUR           # stale, not absent


def test_series_name_flows_into_both_keys():
    s = daily()
    assert s.knowable_series().name == "slow"
    assert s.labelled_series(name="other").name == "other"


def test_knowable_at_is_end_plus_lag():
    s = daily(lag_ns=HOUR)
    assert s.knowable_at(s.periods[0]) == DAY + HOUR


# -- joining alongside a fast stream ----------------------------------------


def test_joins_into_align_like_any_other_series():
    from mdnorm import AsOfSeries

    slow = daily().knowable_series(name="daily")
    fast = AsOfSeries([(i * HOUR, D(i)) for i in range(72)], name="hourly")
    points = grid(0, 3 * DAY, 12 * HOUR)
    rows = [
        {"daily": slow.at(t)[0], "hourly": fast.at(t)[0]} for t in points
    ]
    # Nothing from the slow series before the first close at t = 1 day.
    assert rows[0]["daily"] is None
    assert rows[1]["daily"] is None
    assert rows[2]["daily"] == D("1")


# -- leak_report ------------------------------------------------------------


def test_label_join_leaks_at_every_point_of_a_contiguous_series():
    """Back-to-back periods leave the naive join nowhere safe to stand.

    Each period's label is the previous period's close, so the moment one
    value becomes readable the label join has already moved on to the next
    one — which will not be readable for another whole period.
    """
    s = daily()
    points = grid(0, 3 * DAY, 6 * HOUR)  # 12 points, four per day
    r = leak_report(s, points)
    assert isinstance(r, LeakReport)
    assert r.grid_points == 12
    assert r.label_points == 12
    assert r.leaking_points == 12
    assert r.knowable_points == 12 - 4  # nothing readable during day one
    assert r.leaking_fraction == 1


def test_leak_report_lead_is_measured_to_the_moment_of_readability():
    s = daily(lag_ns=HOUR)
    r = leak_report(s, [0])
    assert r.leaking_points == 1
    assert r.max_lead_ns == DAY + HOUR  # start of period to close plus lag


def test_no_leak_in_the_gap_between_a_close_and_the_next_label():
    """Real sessions leave a safe window: after the close, before tomorrow.

    This is why the count is worth computing rather than assuming. Overnight
    points are fine; the damage is concentrated inside the session.
    """
    values = {date(2026, 1, 15): D("100"), date(2026, 1, 16): D("101")}
    s = PeriodSeries.from_sessions(values, US_EQUITY_RTH)
    close_15 = session_bounds(date(2026, 1, 15), US_EQUITY_RTH)[1]
    open_16 = session_bounds(date(2026, 1, 16), US_EQUITY_RTH)[0]
    overnight = [close_15, close_15 + HOUR, open_16 - 1]
    r = leak_report(s, overnight)
    assert r.label_points == 3
    assert r.leaking_points == 0
    assert r.max_lead_ns is None
    assert r.leaking_fraction == 0
    # One minute after the next open, the same join is wrong again.
    assert leak_report(s, [open_16 + MINUTE]).leaking_points == 1


def test_leak_report_ignores_points_before_the_series_starts():
    s = PeriodSeries([Period(10 * DAY, 11 * DAY, D("1"))])
    r = leak_report(s, [0, DAY])
    assert r.label_points == 0
    assert r.leaking_points == 0


def test_leaking_fraction_is_none_on_an_empty_grid():
    assert leak_report(daily(), []).leaking_fraction is None


def test_leak_report_measures_against_the_right_period():
    """A leak is measured against its own value's close, not the newest one."""
    s = daily(n=3)
    # Midway through day two: the label join shows day two's value, which
    # becomes readable at the end of day two, not the end of the series.
    r = leak_report(s, [DAY + DAY // 2])
    assert r.leaking_points == 1
    assert r.max_lead_ns == DAY // 2


# -- CSV --------------------------------------------------------------------


def test_read_periods_csv_accepts_iso_and_nanoseconds(tmp_path):
    p = tmp_path / "periods.csv"
    p.write_text(
        "start,end,value\n"
        "2026-01-01T00:00:00Z,2026-01-02T00:00:00Z,1.5\n"
        f"{2 * DAY},{3 * DAY},2.5\n"
    )
    s = read_periods_csv(str(p), publication_lag_ns=HOUR, name="csv")
    assert len(s) == 2
    assert s.name == "csv"
    assert [pp.value for pp in s.periods] == [D("2.5"), D("1.5")]
    assert s.publication_lag_ns == HOUR


def test_read_periods_csv_rejects_a_file_without_an_end(tmp_path):
    p = tmp_path / "labels_only.csv"
    p.write_text("start,value\n0,1.5\n")
    with pytest.raises(KeyError):
        read_periods_csv(str(p))


def test_repr_is_informative():
    assert "PeriodSeries" in repr(daily())
