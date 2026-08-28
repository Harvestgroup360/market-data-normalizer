"""Holidays, half-days, and the year that is not 252 sessions long.

:mod:`mdnorm.sessions` describes a recurring window — 09:30 to 16:00 New York,
weekdays. That is most of a trading calendar and not all of it. The remainder
is the exceptions, and they are the part that quietly breaks things::

    from mdnorm import TradingCalendar, read_calendar_csv

    cal = read_calendar_csv("us_equity_2026.csv", session=US_EQUITY_RTH)
    cal.is_trading_day(date(2026, 7, 3))     # False, and the file says so
    cal.trading_minutes_between(jan, dec)    # not 252 x 390

**A missing holiday looks exactly like missing data.** A pipeline that does not
know 4 July is a holiday sees a day-long gap and reports it as an outage, or
fills it, or drops the instrument for poor coverage. All three are wrong in a
way that costs an afternoon and teaches nothing, because the data was never
supposed to be there.

**A half-day is not half a problem.** An early close shortens the session
without changing anything else, so bars keep being generated against a
6.5-hour assumption, a volatility annualised on session length is overstated
for that day, and a staleness check fires on every instrument at once an hour
before it should.

**A calendar cannot answer outside the range it was given.** This is the
distinction the module is built around: a file listing 2026 holidays says
nothing about 2027, and a calendar that silently treats an unknown weekday as
open converts a missing file into a confident wrong answer. Every query
outside :attr:`TradingCalendar.covers` raises rather than guesses, which is
noisy exactly once and then correct.

**252 is a convention, not a count.** The number of sessions in a year depends
on where the weekends and holidays fell, and the number of *minutes* depends
on how many of those sessions closed early. Both are computable from a
calendar, and both are constants that rescale every annualised figure in a
report while leaving its shape untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from .fileio import open_text
from .sessions import Session, session_bounds

__all__ = [
    "Holiday",
    "EarlyClose",
    "CalendarReport",
    "TradingCalendar",
    "read_calendar_csv",
]

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class Holiday:
    """A date on which the venue does not trade at all."""

    day: date
    name: str = ""


@dataclass(frozen=True, slots=True)
class EarlyClose:
    """A date on which the venue closes before its usual time."""

    day: date
    close: time
    name: str = ""

    def __post_init__(self) -> None:
        if self.close.tzinfo is not None:
            raise ValueError("close must be a naive local time")


@dataclass(frozen=True, slots=True)
class CalendarReport:
    """What the calendar covers and what it contains."""

    first_day: date
    last_day: date
    trading_days: int
    holidays: int
    early_closes: int
    weekend_days: int

    @property
    def calendar_days(self) -> int:
        return (self.last_day - self.first_day).days + 1


class TradingCalendar:
    """A recurring session plus the exceptions to it, over a stated range.

    ``covers`` is the closed range of dates the source file described. Nothing
    outside it can be answered, because the absence of a holiday in a file that
    never claimed to cover that year is not evidence that the venue was open.
    """

    __slots__ = ("session", "name", "_holidays", "_early", "_first", "_last")

    def __init__(
        self,
        session: Session,
        *,
        first_day: date,
        last_day: date,
        holidays: Iterable[Holiday] = (),
        early_closes: Iterable[EarlyClose] = (),
        name: str = "",
    ) -> None:
        if last_day < first_day:
            raise ValueError("last_day must not precede first_day")
        self.session = session
        self.name = name
        self._first = first_day
        self._last = last_day
        self._holidays: Dict[date, Holiday] = {h.day: h for h in holidays}
        self._early: Dict[date, EarlyClose] = {e.day: e for e in early_closes}
        overlap = set(self._holidays) & set(self._early)
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(
                f"{example} is listed as both a holiday and an early close; "
                "a venue that is shut cannot also close early"
            )
        for day in list(self._holidays) + list(self._early):
            if not (first_day <= day <= last_day):
                raise ValueError(
                    f"{day} lies outside the stated range "
                    f"{first_day}..{last_day}"
                )

    # -- coverage ----------------------------------------------------------

    @property
    def covers(self) -> Tuple[date, date]:
        """The closed range of dates this calendar can answer for."""
        return self._first, self._last

    def _check(self, day: date) -> None:
        if not (self._first <= day <= self._last):
            raise ValueError(
                f"{day} is outside this calendar ({self._first}..{self._last}); "
                "extend the source file rather than assuming the venue was open"
            )

    # -- querying ----------------------------------------------------------

    def is_trading_day(self, day: date) -> bool:
        """Whether the venue opened at all on ``day``."""
        self._check(day)
        if day in self._holidays:
            return False
        return day.weekday() in self.session.days

    def close_time(self, day: date) -> Optional[time]:
        """The local closing time on ``day``, or ``None`` if it did not trade."""
        if not self.is_trading_day(day):
            return None
        early = self._early.get(day)
        return early.close if early else self.session.end

    def session_on(self, day: date) -> Optional[Tuple[int, int]]:
        """The UTC nanosecond span of the session opening on ``day``.

        ``None`` on a non-trading day. An early close shortens the span; the
        open is unaffected, which is what an early close means.
        """
        if not self.is_trading_day(day):
            return None
        open_ns, close_ns = session_bounds(day, self.session)
        early = self._early.get(day)
        if early is None:
            return open_ns, close_ns
        shortened = Session(start=self.session.start, end=early.close,
                            tz=self.session.tz, days=self.session.days)
        return session_bounds(day, shortened)

    def is_open(self, ts_ns: int) -> bool:
        """Whether the venue was trading at ``ts_ns``.

        Both the day it opened on and the previous day are considered, so an
        overnight session that began yesterday is handled.
        """
        as_utc = datetime.fromtimestamp(ts_ns // _NS_PER_S, tz=timezone.utc)
        local_day = as_utc.astimezone(self.session.zone).date()
        for day in (local_day, local_day - timedelta(days=1)):
            if not (self._first <= day <= self._last):
                continue
            span = self.session_on(day)
            if span and span[0] <= ts_ns < span[1]:
                return True
        self._check(local_day)
        return False

    # -- counting ----------------------------------------------------------

    def trading_days_between(self, start: date, end: date) -> List[date]:
        """Every trading day in the closed range ``start..end``."""
        self._check(start)
        self._check(end)
        if end < start:
            raise ValueError("end must not precede start")
        days: List[date] = []
        day = start
        while day <= end:
            if self.is_trading_day(day):
                days.append(day)
            day += timedelta(days=1)
        return days

    def trading_seconds_between(self, start: date, end: date) -> int:
        """Total seconds the venue was open across the closed range.

        This is the figure an annualisation actually needs, and it is not
        ``sessions x session_length`` whenever the range contains a half-day.
        """
        total = 0
        for day in self.trading_days_between(start, end):
            span = self.session_on(day)
            if span:
                total += (span[1] - span[0]) // _NS_PER_S
        return total

    def trading_minutes_between(self, start: date, end: date) -> int:
        """Total whole minutes the venue was open across the closed range."""
        return self.trading_seconds_between(start, end) // 60

    def report(self, start: Optional[date] = None,
               end: Optional[date] = None) -> CalendarReport:
        """What the calendar holds over its own range, or a part of it."""
        first = start or self._first
        last = end or self._last
        self._check(first)
        self._check(last)
        trading = self.trading_days_between(first, last)
        weekend = 0
        day = first
        while day <= last:
            if day.weekday() not in self.session.days:
                weekend += 1
            day += timedelta(days=1)
        return CalendarReport(
            first_day=first,
            last_day=last,
            trading_days=len(trading),
            holidays=sum(1 for d in self._holidays if first <= d <= last),
            early_closes=sum(1 for d in self._early if first <= d <= last),
            weekend_days=weekend,
        )

    @property
    def holidays(self) -> Tuple[Holiday, ...]:
        return tuple(self._holidays[d] for d in sorted(self._holidays))

    @property
    def early_closes(self) -> Tuple[EarlyClose, ...]:
        return tuple(self._early[d] for d in sorted(self._early))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"TradingCalendar(name={self.name!r}, "
                f"covers={self._first}..{self._last}, "
                f"holidays={len(self._holidays)}, "
                f"early_closes={len(self._early)})")


def read_calendar_csv(
    path: str,
    session: Session,
    *,
    first_day: Optional[date] = None,
    last_day: Optional[date] = None,
    date_column: str = "date",
    kind_column: str = "kind",
    close_column: str = "close",
    name_column: str = "name",
    name: str = "",
) -> TradingCalendar:
    """Read exceptions from CSV: one row per holiday or early close.

    ``kind`` is ``holiday`` or ``early_close``; an early close also needs a
    local ``close`` time in ``HH:MM``. The covered range defaults to the first
    and last day of the year each exception falls in, which is the range such
    files describe in practice — pass ``first_day`` and ``last_day`` to state
    it exactly rather than relying on that inference.
    """
    import csv

    holidays: List[Holiday] = []
    early: List[EarlyClose] = []
    seen_years = set()
    with open_text(path) as fh:
        for row in csv.DictReader(fh):
            day = date.fromisoformat(row[date_column].strip())
            seen_years.add(day.year)
            label = row.get(name_column, "").strip()
            kind = row[kind_column].strip().lower()
            if kind == "holiday":
                holidays.append(Holiday(day, label))
            elif kind == "early_close":
                raw = row.get(close_column, "").strip()
                if not raw:
                    raise ValueError(
                        f"{day}: an early close needs a close time")
                hh, mm = raw.split(":")[:2]
                early.append(EarlyClose(day, time(int(hh), int(mm)), label))
            else:
                raise ValueError(
                    f"unknown kind {kind!r}; expected 'holiday' or "
                    "'early_close'")
    if not seen_years and (first_day is None or last_day is None):
        raise ValueError(
            "an empty calendar file cannot imply a covered range; pass "
            "first_day and last_day")
    first = first_day or date(min(seen_years), 1, 1)
    last = last_day or date(max(seen_years), 12, 31)
    return TradingCalendar(session, first_day=first, last_day=last,
                           holidays=holidays, early_closes=early, name=name)
