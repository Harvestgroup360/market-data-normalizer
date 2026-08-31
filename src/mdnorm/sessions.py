"""Trading sessions and calendar filtering.

Raw feeds run around the clock; research rarely should. Overnight prints,
weekend maintenance windows and pre-market crossings all distort features
that were meant to describe regular trading hours.

A :class:`Session` describes a recurring local-time window — regular US
equity hours, an overnight futures session, a weekday-only crypto slice —
and the helpers below decide which events or bars belong to it. Timezones
and daylight-saving transitions are handled by :mod:`zoneinfo`, so a
09:30 New York open stays 09:30 in both January and July. Standard
library only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import FrozenSet, Iterable, List, Sequence, Tuple, Union, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .bars import Bar
from .schema import MarketEvent

_NS_PER_S = 1_000_000_000

#: Monday–Friday, matching :meth:`datetime.date.weekday` numbering.
WEEKDAYS: FrozenSet[int] = frozenset({0, 1, 2, 3, 4})


@dataclass(frozen=True, slots=True)
class Session:
    """A recurring trading window expressed in local exchange time.

    ``start`` and ``end`` are local wall-clock times in ``tz``:

    - ``start < end`` — an intraday window, e.g. 09:30–16:00 (regular US
      equity hours).
    - ``start > end`` — an overnight window that opens on one calendar day
      and closes on the next, e.g. 18:00–17:00 (CME-style futures).
    - ``start == end`` — a full 24-hour day.

    ``days`` selects which local calendar days *open* a session, using
    :meth:`datetime.date.weekday` numbering (Monday = 0). For an overnight
    session this is the day the window opens, so a Friday-evening open is
    included by ``4`` even though it closes on Saturday.
    """

    start: time
    end: time
    tz: str = "UTC"
    days: FrozenSet[int] = field(default=WEEKDAYS)

    def __post_init__(self) -> None:
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise ValueError("session times must be naive local times")
        if not self.days:
            raise ValueError("session must include at least one weekday")
        if any(d not in range(7) for d in self.days):
            raise ValueError("days must be weekday numbers 0..6 (Monday = 0)")
        try:
            ZoneInfo(self.tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {self.tz!r}") from exc

    @property
    def is_overnight(self) -> bool:
        """True when the window crosses local midnight."""
        return self.start > self.end

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.tz)


def _local(ts_ns: int, session: Session) -> datetime:
    """Convert epoch nanoseconds to an aware datetime in the session's zone."""
    seconds, nanos = divmod(ts_ns, _NS_PER_S)
    utc = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=nanos // 1_000
    )
    return utc.astimezone(session.zone)


#: Anything with a timestamp this module knows how to read.
_Item = TypeVar("_Item", bound=Union[MarketEvent, Bar])


def _timestamp_of(item: Union[MarketEvent, Bar]) -> int:
    if isinstance(item, Bar):
        return item.start_ns
    if isinstance(item, MarketEvent):
        return item.ts_ns
    raise TypeError(f"unsupported item type: {type(item).__name__}")


def in_session(ts_ns: int, session: Session) -> bool:
    """Return True if ``ts_ns`` falls inside ``session``.

    The window is half-open: the opening time is included, the closing
    time is not.
    """
    local = _local(ts_ns, session)
    local_time = local.timetz().replace(tzinfo=None)

    if session.start == session.end:  # full day
        return local.weekday() in session.days

    if not session.is_overnight:
        if not (session.start <= local_time < session.end):
            return False
        return local.weekday() in session.days

    # Overnight: after the open it belongs to today's session, before the
    # close it belongs to the session that opened on the previous day.
    if local_time >= session.start:
        return local.weekday() in session.days
    if local_time < session.end:
        return (local.date() - timedelta(days=1)).weekday() in session.days
    return False


def session_date(ts_ns: int, session: Session) -> date:
    """Return the local calendar date on which ``ts_ns``'s session opened.

    Useful as a grouping key: for an overnight session, prints made after
    midnight are attributed to the previous calendar day, so a whole
    trading day stays in one bucket.
    """
    local = _local(ts_ns, session)
    if session.is_overnight and local.timetz().replace(tzinfo=None) < session.end:
        return local.date() - timedelta(days=1)
    return local.date()


def session_bounds(day: date, session: Session) -> Tuple[int, int]:
    """The UTC nanosecond span of the session that *opens* on ``day``.

    Returns ``(open_ns, close_ns)``, half-open like every other interval in
    this library. For an overnight session the close falls on the following
    local date; for a 24-hour session it is exactly one local day later, which
    is not always twenty-four hours — daylight-saving days are 23 or 25 hours
    long and the arithmetic here follows the calendar rather than a constant.

    ``day`` is not checked against ``session.days``: a caller asking for the
    bounds of a Sunday has a reason, and silently returning something else
    would be worse than answering the question asked.

    On the two hours a year that a local clock repeats or skips, the earlier
    of an ambiguous pair is used and a skipped time resolves forward, which is
    :mod:`zoneinfo`'s default. Sessions that open inside a transition are rare
    and worth knowing about rather than guessing at.
    """
    zone = session.zone
    open_local = datetime.combine(day, session.start).replace(tzinfo=zone)
    if session.start == session.end:
        close_day = day + timedelta(days=1)
    elif session.is_overnight:
        close_day = day + timedelta(days=1)
    else:
        close_day = day
    close_local = datetime.combine(close_day, session.end).replace(tzinfo=zone)
    open_ns = int(open_local.timestamp()) * _NS_PER_S
    close_ns = int(close_local.timestamp()) * _NS_PER_S
    if close_ns <= open_ns:  # pragma: no cover - guarded by Session validation
        raise ValueError("session close must follow its open")
    return open_ns, close_ns


def filter_session(items: Iterable[_Item], session: Session) -> List[_Item]:
    """Keep only the events or bars that fall inside ``session``.

    Bars are judged by their opening timestamp, events by their own. The
    element type survives: filtering bars gives back bars, not a union that
    every caller then has to narrow again.
    """
    return [it for it in items if in_session(_timestamp_of(it), session)]


def group_by_session_date(
    items: Sequence[Union[MarketEvent, Bar]], session: Session
) -> dict:
    """Group events or bars into ``{trading_date: [items]}``.

    Items outside the session are dropped, and insertion order follows the
    input, so each bucket keeps its original ordering.
    """
    out: dict = {}
    for it in items:
        ts = _timestamp_of(it)
        if not in_session(ts, session):
            continue
        out.setdefault(session_date(ts, session), []).append(it)
    return out


def parse_session(
    window: str, tz: str = "UTC", days: FrozenSet[int] = WEEKDAYS
) -> Session:
    """Parse ``"09:30-16:00"`` into a :class:`Session`.

    Accepts ``HH:MM`` or ``HH:MM:SS`` on either side of the dash.
    """
    if "-" not in window:
        raise ValueError(f"invalid session window {window!r} (expected HH:MM-HH:MM)")
    raw_start, _, raw_end = window.partition("-")

    def _parse(value: str) -> time:
        parts = value.strip().split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(f"invalid time {value!r} (expected HH:MM or HH:MM:SS)")
        nums = [int(p) for p in parts] + [0]
        if not (0 <= nums[0] < 24 and 0 <= nums[1] < 60 and 0 <= nums[2] < 60):
            raise ValueError(f"invalid time {value!r}")
        return time(nums[0], nums[1], nums[2])

    return Session(start=_parse(raw_start), end=_parse(raw_end), tz=tz, days=days)


#: Regular US equity trading hours (09:30–16:00 America/New_York, Mon–Fri).
US_EQUITY_RTH = Session(time(9, 30), time(16, 0), "America/New_York", WEEKDAYS)

#: CME-style overnight futures session (18:00–17:00 America/New_York).
US_FUTURES_OVERNIGHT = Session(
    time(18, 0), time(17, 0), "America/New_York", frozenset({6, 0, 1, 2, 3})
)
