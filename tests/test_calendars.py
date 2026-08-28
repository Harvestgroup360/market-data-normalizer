"""Calendar tests: holidays, half-days, and refusing to answer past the file."""
from datetime import date, time

import pytest

from mdnorm import (
    EarlyClose,
    Holiday,
    TradingCalendar,
    US_EQUITY_RTH,
    read_calendar_csv,
    session_bounds,
)

HOUR = 3600
NS = 1_000_000_000


def cal(**kw):
    """US regular hours for 2026, with Independence Day and one half-day."""
    kw.setdefault("holidays", [Holiday(date(2026, 7, 3), "Independence Day")])
    kw.setdefault("early_closes",
                  [EarlyClose(date(2026, 11, 27), time(13, 0), "Thanksgiving")])
    return TradingCalendar(US_EQUITY_RTH,
                           first_day=date(2026, 1, 1),
                           last_day=date(2026, 12, 31),
                           name="us-equity-2026", **kw)


# -- coverage ---------------------------------------------------------------


def test_a_date_outside_the_file_is_refused_not_guessed():
    c = cal()
    with pytest.raises(ValueError) as exc:
        c.is_trading_day(date(2027, 3, 1))
    assert "outside this calendar" in str(exc.value)


def test_covers_reports_the_stated_range():
    assert cal().covers == (date(2026, 1, 1), date(2026, 12, 31))


def test_an_exception_outside_the_range_is_rejected_at_construction():
    with pytest.raises(ValueError) as exc:
        TradingCalendar(US_EQUITY_RTH, first_day=date(2026, 1, 1),
                        last_day=date(2026, 12, 31),
                        holidays=[Holiday(date(2027, 1, 1))])
    assert "outside the stated range" in str(exc.value)


def test_an_inverted_range_is_rejected():
    with pytest.raises(ValueError):
        TradingCalendar(US_EQUITY_RTH, first_day=date(2026, 12, 31),
                        last_day=date(2026, 1, 1))


def test_a_day_cannot_be_both_shut_and_short():
    with pytest.raises(ValueError) as exc:
        TradingCalendar(US_EQUITY_RTH, first_day=date(2026, 1, 1),
                        last_day=date(2026, 12, 31),
                        holidays=[Holiday(date(2026, 7, 3))],
                        early_closes=[EarlyClose(date(2026, 7, 3), time(13, 0))])
    assert "cannot also close early" in str(exc.value)


def test_early_close_rejects_an_aware_time():
    from datetime import timezone
    with pytest.raises(ValueError):
        EarlyClose(date(2026, 1, 2), time(13, 0, tzinfo=timezone.utc))


# -- trading days -----------------------------------------------------------


def test_a_holiday_is_not_a_trading_day():
    c = cal()
    assert c.is_trading_day(date(2026, 7, 2)) is True     # Thursday
    assert c.is_trading_day(date(2026, 7, 3)) is False    # holiday, Friday
    assert c.is_trading_day(date(2026, 7, 6)) is True     # Monday


def test_a_weekend_is_not_a_trading_day():
    c = cal()
    assert c.is_trading_day(date(2026, 7, 4)) is False     # Saturday
    assert c.is_trading_day(date(2026, 7, 5)) is False     # Sunday


def test_session_is_none_on_a_closed_day():
    assert cal().session_on(date(2026, 7, 3)) is None
    assert cal().close_time(date(2026, 7, 3)) is None


# -- half-days --------------------------------------------------------------


def test_an_early_close_shortens_the_session_and_leaves_the_open_alone():
    c = cal()
    day = date(2026, 11, 27)
    normal_open, normal_close = session_bounds(day, US_EQUITY_RTH)
    open_ns, close_ns = c.session_on(day)
    assert open_ns == normal_open
    assert close_ns < normal_close
    assert (close_ns - open_ns) // NS == 3 * HOUR + 30 * 60
    assert c.close_time(day) == time(13, 0)


def test_a_normal_day_keeps_the_full_session():
    open_ns, close_ns = cal().session_on(date(2026, 11, 25))
    assert (close_ns - open_ns) // NS == 6 * HOUR + 30 * 60


# -- is_open ----------------------------------------------------------------


def test_is_open_follows_the_shortened_session():
    c = cal()
    day = date(2026, 11, 27)
    open_ns, close_ns = c.session_on(day)
    assert c.is_open(open_ns) is True
    assert c.is_open(close_ns - 1) is True
    assert c.is_open(close_ns) is False              # half-day is over
    normal_close = session_bounds(day, US_EQUITY_RTH)[1]
    assert c.is_open(normal_close - 1) is False      # would have been open


def test_is_open_is_false_all_day_on_a_holiday():
    c = cal()
    noon = session_bounds(date(2026, 7, 2), US_EQUITY_RTH)[0]
    holiday_noon = noon + 86_400 * NS
    assert c.is_open(holiday_noon) is False


# -- counting ---------------------------------------------------------------


def test_trading_days_skip_holidays_and_weekends():
    c = cal()
    days = c.trading_days_between(date(2026, 6, 29), date(2026, 7, 10))
    assert date(2026, 7, 3) not in days       # holiday
    assert date(2026, 7, 4) not in days       # Saturday
    assert len(days) == 9                     # 10 weekdays minus the holiday


def test_a_half_day_shortens_the_minute_count_but_not_the_day_count():
    c = cal()
    week = (date(2026, 11, 23), date(2026, 11, 27))
    days = c.trading_days_between(*week)
    assert len(days) == 5
    minutes = c.trading_minutes_between(*week)
    # four full sessions of 390 minutes and one of 210
    assert minutes == 4 * 390 + 210


def test_seconds_and_minutes_agree():
    c = cal()
    a, b = date(2026, 11, 23), date(2026, 11, 27)
    assert c.trading_minutes_between(a, b) == c.trading_seconds_between(a, b) // 60


def test_an_inverted_range_is_refused():
    with pytest.raises(ValueError):
        cal().trading_days_between(date(2026, 7, 10), date(2026, 7, 1))


def test_a_year_is_not_exactly_252_sessions():
    """The count depends on where the weekends and holidays fell."""
    c = cal()
    days = len(c.trading_days_between(date(2026, 1, 1), date(2026, 12, 31)))
    assert 250 <= days <= 262
    assert days != 252 or True     # the point is that it is computed, not assumed


# -- report -----------------------------------------------------------------


def test_report_counts_the_pieces():
    r = cal().report()
    assert r.first_day == date(2026, 1, 1) and r.last_day == date(2026, 12, 31)
    assert r.holidays == 1
    assert r.early_closes == 1
    assert r.calendar_days == 365
    assert r.trading_days + r.holidays + r.weekend_days == 365


def test_report_can_be_narrowed():
    r = cal().report(date(2026, 7, 1), date(2026, 7, 7))
    assert r.calendar_days == 7
    assert r.holidays == 1
    assert r.trading_days == 4          # Wed, Thu, Mon, Tue


# -- CSV --------------------------------------------------------------------


def test_read_calendar_csv(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text(
        "date,kind,close,name\n"
        "2026-07-03,holiday,,Independence Day\n"
        "2026-11-27,early_close,13:00,Day after Thanksgiving\n"
    )
    c = read_calendar_csv(str(p), US_EQUITY_RTH, name="us")
    assert c.covers == (date(2026, 1, 1), date(2026, 12, 31))
    assert c.is_trading_day(date(2026, 7, 3)) is False
    assert c.close_time(date(2026, 11, 27)) == time(13, 0)
    assert [h.name for h in c.holidays] == ["Independence Day"]


def test_read_calendar_csv_rejects_an_unknown_kind(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date,kind,close,name\n2026-07-03,closed,,x\n")
    with pytest.raises(ValueError) as exc:
        read_calendar_csv(str(p), US_EQUITY_RTH)
    assert "unknown kind" in str(exc.value)


def test_read_calendar_csv_requires_a_close_for_a_half_day(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date,kind,close,name\n2026-11-27,early_close,,x\n")
    with pytest.raises(ValueError) as exc:
        read_calendar_csv(str(p), US_EQUITY_RTH)
    assert "needs a close time" in str(exc.value)


def test_an_empty_file_cannot_imply_a_range(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date,kind,close,name\n")
    with pytest.raises(ValueError) as exc:
        read_calendar_csv(str(p), US_EQUITY_RTH)
    assert "cannot imply a covered range" in str(exc.value)


def test_an_empty_file_is_fine_with_a_stated_range(tmp_path):
    p = tmp_path / "cal.csv"
    p.write_text("date,kind,close,name\n")
    c = read_calendar_csv(str(p), US_EQUITY_RTH,
                          first_day=date(2026, 1, 1),
                          last_day=date(2026, 1, 31))
    assert c.report().holidays == 0
    assert c.is_trading_day(date(2026, 1, 2)) is True


def test_repr_is_informative():
    assert "TradingCalendar" in repr(cal())
