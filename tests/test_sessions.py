"""Trading-session filtering tests, including DST and overnight windows."""
from datetime import datetime, time, timezone
from decimal import Decimal

import pytest

from mdnorm import (
    US_EQUITY_RTH,
    US_FUTURES_OVERNIGHT,
    EventType,
    MarketEvent,
    Pipeline,
    Session,
    Side,
    filter_session,
    group_by_session_date,
    in_session,
    parse_session,
    session_date,
    time_bars,
)

MIN_NS = 60_000_000_000


def ns(iso: str) -> int:
    """UTC ISO-8601 string -> epoch nanoseconds."""
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000


def trade(ts_ns, price="100"):
    return MarketEvent(
        symbol="SPY", venue="x", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal("1"), side=Side.BUY,
    )


# -- regular hours ---------------------------------------------------------

def test_rth_boundaries_are_half_open():
    # 2026-08-05 is a Wednesday. 09:30 EDT == 13:30 UTC, 16:00 EDT == 20:00 UTC.
    assert in_session(ns("2026-08-05T13:30:00Z"), US_EQUITY_RTH)      # open: in
    assert in_session(ns("2026-08-05T19:59:59Z"), US_EQUITY_RTH)
    assert not in_session(ns("2026-08-05T20:00:00Z"), US_EQUITY_RTH)  # close: out
    assert not in_session(ns("2026-08-05T13:29:59Z"), US_EQUITY_RTH)  # pre-market


def test_rth_survives_dst_change():
    """09:30 New York stays 09:30 in winter and summer."""
    # 2026-01-07 (Wed, EST, UTC-5): open is 14:30 UTC
    assert in_session(ns("2026-01-07T14:30:00Z"), US_EQUITY_RTH)
    assert not in_session(ns("2026-01-07T13:30:00Z"), US_EQUITY_RTH)
    # 2026-07-08 (Wed, EDT, UTC-4): open is 13:30 UTC
    assert in_session(ns("2026-07-08T13:30:00Z"), US_EQUITY_RTH)
    assert not in_session(ns("2026-07-08T12:30:00Z"), US_EQUITY_RTH)


def test_weekend_is_excluded():
    # 2026-08-08 is a Saturday, 15:00 UTC == 11:00 EDT (inside the clock window)
    assert not in_session(ns("2026-08-08T15:00:00Z"), US_EQUITY_RTH)


# -- overnight sessions ----------------------------------------------------

def test_overnight_window_spans_midnight():
    s = US_FUTURES_OVERNIGHT  # 18:00 -> 17:00 New York
    # Wed 2026-08-05 19:00 EDT == 23:00 UTC — after the open
    assert in_session(ns("2026-08-05T23:00:00Z"), s)
    # Thu 2026-08-06 03:00 EDT == 07:00 UTC — still the Wednesday session
    assert in_session(ns("2026-08-06T07:00:00Z"), s)
    # Thu 17:30 EDT == 21:30 UTC — the daily maintenance break
    assert not in_session(ns("2026-08-06T21:30:00Z"), s)


def test_overnight_session_date_groups_the_night_with_its_open():
    s = US_FUTURES_OVERNIGHT
    evening = ns("2026-08-05T23:00:00Z")   # Wed 19:00 EDT
    small_hours = ns("2026-08-06T07:00:00Z")  # Thu 03:00 EDT
    assert session_date(evening, s) == session_date(small_hours, s)
    assert str(session_date(evening, s)) == "2026-08-05"


def test_overnight_respects_opening_weekday():
    s = US_FUTURES_OVERNIGHT  # opens Sun..Thu
    # Friday 2026-08-07 19:00 EDT == 23:00 UTC: Friday is not an opening day
    assert not in_session(ns("2026-08-07T23:00:00Z"), s)
    # Sunday 2026-08-09 19:00 EDT == 23:00 UTC: Sunday opens the week
    assert in_session(ns("2026-08-09T23:00:00Z"), s)


def test_full_day_session():
    s = Session(time(0, 0), time(0, 0), "UTC", frozenset({0, 1, 2, 3, 4, 5, 6}))
    assert in_session(ns("2026-08-08T03:00:00Z"), s)
    assert in_session(ns("2026-08-08T23:59:59Z"), s)


# -- filtering and grouping ------------------------------------------------

def test_filter_session_on_events_and_bars():
    events = [
        trade(ns("2026-08-05T12:00:00Z")),  # 08:00 EDT, pre-market
        trade(ns("2026-08-05T14:00:00Z")),  # 10:00 EDT, in
        trade(ns("2026-08-05T18:00:00Z")),  # 14:00 EDT, in
        trade(ns("2026-08-05T22:00:00Z")),  # 18:00 EDT, after hours
    ]
    kept = filter_session(events, US_EQUITY_RTH)
    assert len(kept) == 2

    bars = time_bars(events, MIN_NS)
    assert len(filter_session(bars, US_EQUITY_RTH)) == 2


def test_group_by_session_date():
    events = [
        trade(ns("2026-08-05T14:00:00Z")),
        trade(ns("2026-08-05T15:00:00Z")),
        trade(ns("2026-08-06T14:00:00Z")),
        trade(ns("2026-08-05T02:00:00Z")),  # outside RTH — dropped
    ]
    grouped = group_by_session_date(events, US_EQUITY_RTH)
    assert [str(k) for k in sorted(grouped)] == ["2026-08-05", "2026-08-06"]
    assert len(grouped[sorted(grouped)[0]]) == 2


def test_pipeline_session_step():
    events = [
        trade(ns("2026-08-05T12:00:00Z")),
        trade(ns("2026-08-05T14:00:00Z")),
        trade(ns("2026-08-05T14:00:30Z")),
    ]
    pipe = Pipeline().session(US_EQUITY_RTH).time_bars(MIN_NS)
    bars = pipe.run(events)
    assert pipe.steps == ["session", "time_bars"]
    assert len(bars) == 1 and bars[0].trades == 2


# -- parsing and validation ------------------------------------------------

@pytest.mark.parametrize("text,start,end", [
    ("09:30-16:00", time(9, 30), time(16, 0)),
    ("18:00:30-17:00", time(18, 0, 30), time(17, 0)),
])
def test_parse_session(text, start, end):
    s = parse_session(text, "America/New_York")
    assert s.start == start and s.end == end and s.tz == "America/New_York"


@pytest.mark.parametrize("bad", ["0930-1600", "09:30", "25:00-16:00", "09:70-16:00", "aa:bb-16:00"])
def test_parse_session_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_session(bad)


def test_session_rejects_unknown_timezone():
    with pytest.raises(ValueError, match="unknown timezone"):
        Session(time(9, 0), time(17, 0), "Mars/Olympus")


def test_session_rejects_empty_days():
    with pytest.raises(ValueError):
        Session(time(9, 0), time(17, 0), "UTC", frozenset())
