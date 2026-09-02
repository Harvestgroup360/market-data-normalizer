"""The shape of a trading day, measured without reading the rest of the year.

Volume, spread, volatility and trade count all follow a pronounced curve
within a session: heavy at the open, thin in the middle, heavy again into the
close. Any statistic computed across a day without removing that curve is
mostly measuring the time of day::

    from mdnorm import Sample, US_EQUITY_RTH, session_profile, deseasonalise

    shape = session_profile(volumes, US_EQUITY_RTH, bucket_ns=5 * 60 * 10**9)
    shape.factor_at(offset_ns)                 # 3.4x at the open, 0.4x at noon
    adjusted = deseasonalise(volumes, US_EQUITY_RTH,
                             bucket_ns=5 * 60 * 10**9, min_sessions=20)

**Removing the shape is the easy half. Not reading the future while you do it
is the hard half.** The usual recipe estimates one profile over the whole
sample and divides every day by it, including the first. That profile was
built from days that had not happened yet, so an "unusually heavy open" in
January is being judged against a curve that already contains December. The
adjusted series is smoother than anything that could have been computed at the
time, and smoother inputs make better-looking signals.

:func:`expanding_profiles` gives each session a profile built only from the
sessions before it. :func:`deseasonalise` uses it. :func:`session_profile` builds the
full-sample version, kept deliberately — it is the right tool for describing
what a market does and the wrong one for adjusting a backtest, and shipping
both is what makes the difference measurable rather than arguable.

**There is no default bucket size.** Five minutes over a six-and-a-half-hour
session is 78 buckets; the same five minutes over a venue that never closes is
288. Finer buckets describe the curve better and put fewer observations in
each, and where that trade sits is a property of your data rather than of this
module.

**A bucket with too little evidence reports nothing rather than the average.**
Filling a thin bucket with the overall mean is the flattering choice: it makes
the adjusted series look well-behaved in exactly the places where nothing is
known about it. ``min_observations`` sets the bar and the bucket comes back
empty below it.

**A short session is not a quiet one.** Bucketing by offset from the open puts
a half-day's closing surge into a bucket that is the middle of the afternoon
on every other day, which contaminates both. Given a
:class:`~mdnorm.calendars.TradingCalendar` this module leaves early-close
sessions out of the profile and says how many it left out. Without one it
treats every day as full length, which is a claim, so the count of sessions
used is always reported.

**Nothing is emitted until there is enough history to emit it.** An expanding
profile over fewer than ``min_sessions`` days is not a rough estimate, it is
one day's noise, and dividing by it manufactures outliers rather than removing
them. Those early sessions are dropped, the same way a rolling window emits
nothing until it is full.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import (Dict, Iterable, Iterator, List, Optional, Sequence, Tuple)

from .calendars import TradingCalendar
from .sessions import Session, in_session, session_bounds, session_date

__all__ = [
    "Sample",
    "ProfileBucket",
    "SessionProfile",
    "ProfileLeak",
    "bucket_index",
    "session_profile",
    "expanding_profiles",
    "deseasonalise",
    "full_sample_deseasonalise",
    "profile_leak",
    "read_samples_csv",
]

_ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class Sample:
    """One observation of a per-interval quantity: volume, spread, range."""

    ts_ns: int
    value: Decimal


@dataclass(frozen=True, slots=True)
class ProfileBucket:
    """One slot of the trading day, and what was seen in it.

    ``value`` is ``None`` when the bucket held fewer than the required
    observations. That is not the same as a bucket whose observations summed
    to zero, and the two are kept apart on purpose.
    """

    index: int
    start_offset_ns: int
    end_offset_ns: int
    observations: int
    value: Optional[Decimal]


@dataclass(frozen=True, slots=True)
class SessionProfile:
    """The average shape of a session, bucket by bucket.

    ``sessions`` is how many trading days went into it and ``excluded`` how
    many were left out for being short. Both are reported because a profile
    is only as trustworthy as the number of days behind it, and that number
    is exactly what a caller cannot see from the curve itself.
    """

    bucket_ns: int
    buckets: Tuple[ProfileBucket, ...]
    sessions: int
    excluded: int = 0

    def __len__(self) -> int:
        return len(self.buckets)

    @property
    def observed(self) -> Tuple[ProfileBucket, ...]:
        """The buckets that cleared ``min_observations``."""
        return tuple(b for b in self.buckets if b.value is not None)

    @property
    def complete(self) -> bool:
        """True when every bucket in the session has a value."""
        return len(self.observed) == len(self.buckets)

    @property
    def empty_buckets(self) -> int:
        """How many buckets had too little evidence to report."""
        return len(self.buckets) - len(self.observed)

    @property
    def level(self) -> Optional[Decimal]:
        """The mean across the buckets that have values.

        This is the denominator :meth:`factor_at` divides by, so a profile
        with holes in it is normalised against the part that was observed
        rather than against an assumed zero for the rest.
        """
        seen = self.observed
        if not seen:
            return None
        total = sum((b.value for b in seen if b.value is not None), _ZERO)
        return total / len(seen)

    def at(self, offset_ns: int) -> Optional[Decimal]:
        """The profile value for a given offset into the session."""
        if offset_ns < 0:
            return None
        i = offset_ns // self.bucket_ns
        if i >= len(self.buckets):
            return None
        return self.buckets[i].value

    def factor_at(self, offset_ns: int) -> Optional[Decimal]:
        """How many times the session's own average this offset usually is.

        ``None`` where the bucket is empty or the profile has no level. A
        caller that treats a missing factor as 1.0 has quietly decided the
        unknown part of the day is average, which is the assumption this
        module exists to avoid making on anyone's behalf.
        """
        value = self.at(offset_ns)
        level = self.level
        if value is None or level is None or level == 0:
            return None
        return value / level


@dataclass(frozen=True, slots=True)
class ProfileLeak:
    """How far a full-sample profile differs from what was knowable.

    ``differ`` counts the samples whose adjustment factor changes between the
    two, and ``max_gap`` is the largest relative difference between the
    factor the full-sample profile applies and the one available at the time.
    """

    samples: int
    knowable: int
    differ: int
    median_gap: Optional[Decimal]
    max_gap: Optional[Decimal]
    worst_ts_ns: Optional[int]

    @property
    def differing_fraction(self) -> Optional[Decimal]:
        """Share of knowable samples the two profiles disagree about."""
        if self.knowable == 0:
            return None
        return Decimal(self.differ) / self.knowable


def bucket_index(ts_ns: int, session: Session, bucket_ns: int) -> Optional[int]:
    """Which bucket of its own session a timestamp falls in.

    ``None`` when the timestamp is outside the session. Offsets are measured
    from the moment the session opened on that trading date, so a venue whose
    open shifts with daylight saving keeps its buckets aligned to the bell
    rather than to the clock.
    """
    if bucket_ns <= 0:
        raise ValueError("bucket_ns must be positive")
    if not in_session(ts_ns, session):
        return None
    open_ns, _ = session_bounds(session_date(ts_ns, session), session)
    return (ts_ns - open_ns) // bucket_ns


def _session_length_ns(session: Session) -> int:
    """The nominal length of one session, from a day that is not special."""
    probe = date(2024, 1, 3)                       # a plain Wednesday
    open_ns, close_ns = session_bounds(probe, session)
    return close_ns - open_ns


def _irregular_days(
    days: Iterable[date], session: Session, calendar: Optional[TradingCalendar]
) -> set:
    """Dates whose session was not the regular one, per the calendar.

    That covers both an early close and a day the calendar says did not trade
    at all — a day with prints on it is then a data problem rather than a
    short session, but either way it is not a day whose shape belongs in the
    average. A date the calendar does not cover is left alone: refusing to
    answer outside its range is the calendar behaving correctly, and treating
    silence as a verdict would throw away real sessions.
    """
    if calendar is None:
        return set()
    out = set()
    for day in days:
        try:
            close = calendar.close_time(day)
        except (KeyError, ValueError):
            continue
        if close is None or close != session.end:
            out.add(day)
    return out


def _accumulate(
    samples: Iterable[Sample],
    session: Session,
    bucket_ns: int,
    calendar: Optional[TradingCalendar],
    include_short: bool,
) -> Tuple[Dict[date, Dict[int, List[Decimal]]], set]:
    """Group sample values by trading date and bucket, dropping short days."""
    by_day: Dict[date, Dict[int, List[Decimal]]] = {}
    for s in samples:
        if not in_session(s.ts_ns, session):
            continue
        day = session_date(s.ts_ns, session)
        open_ns, _ = session_bounds(day, session)
        i = (s.ts_ns - open_ns) // bucket_ns
        by_day.setdefault(day, {}).setdefault(i, []).append(s.value)

    excluded = set()
    if not include_short:
        excluded = _irregular_days(by_day, session, calendar)
        for day in excluded:
            by_day.pop(day, None)
    return by_day, excluded


def _build(
    by_day: Dict[date, Dict[int, List[Decimal]]],
    bucket_ns: int,
    n_buckets: int,
    min_observations: int,
    excluded: int,
) -> SessionProfile:
    totals: Dict[int, Decimal] = {}
    counts: Dict[int, int] = {}
    for buckets in by_day.values():
        for i, values in buckets.items():
            if not 0 <= i < n_buckets:
                continue
            totals[i] = totals.get(i, _ZERO) + sum(values, _ZERO)
            counts[i] = counts.get(i, 0) + len(values)

    out = []
    for i in range(n_buckets):
        n = counts.get(i, 0)
        value = totals[i] / n if n >= min_observations and n else None
        out.append(ProfileBucket(index=i, start_offset_ns=i * bucket_ns,
                          end_offset_ns=(i + 1) * bucket_ns,
                          observations=n, value=value))
    return SessionProfile(bucket_ns=bucket_ns, buckets=tuple(out),
                   sessions=len(by_day), excluded=excluded)


def session_profile(
    samples: Iterable[Sample],
    session: Session,
    *,
    bucket_ns: int,
    min_observations: int = 1,
    calendar: Optional[TradingCalendar] = None,
    include_short: bool = False,
) -> SessionProfile:
    """The average shape of the session across every day in ``samples``.

    This is the full-sample profile: it sees the whole input, including the
    part that comes after any particular day in it. That makes it the right
    thing for describing a market and the wrong thing for adjusting a series
    you intend to trade — use :func:`deseasonalise` for the second. It ships
    because a difference you cannot compute is a difference nobody checks.
    """
    if bucket_ns <= 0:
        raise ValueError("bucket_ns must be positive")
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")
    length = _session_length_ns(session)
    n_buckets = -(-length // bucket_ns)                 # ceiling division
    by_day, excluded = _accumulate(samples, session, bucket_ns, calendar,
                                   include_short)
    return _build(by_day, bucket_ns, n_buckets, min_observations,
                  len(excluded))


def expanding_profiles(
    samples: Iterable[Sample],
    session: Session,
    *,
    bucket_ns: int,
    min_sessions: int,
    min_observations: int = 1,
    calendar: Optional[TradingCalendar] = None,
    include_short: bool = False,
) -> Iterator[Tuple[date, SessionProfile]]:
    """Yield ``(trading_date, profile)`` using only the sessions before it.

    The profile handed out for a given day contains no observation from that
    day or any later one, so it is exactly what a process running at that
    day's open could have had. Days before ``min_sessions`` of history exist
    are not yielded at all: a profile built from three days is noise wearing
    a curve's clothes, and dividing by it invents outliers instead of
    removing them.
    """
    if min_sessions < 1:
        raise ValueError("min_sessions must be at least 1")
    length = _session_length_ns(session)
    n_buckets = -(-length // bucket_ns)
    by_day, excluded = _accumulate(samples, session, bucket_ns, calendar,
                                   include_short)

    history: Dict[date, Dict[int, List[Decimal]]] = {}
    for day in sorted(by_day):
        if len(history) >= min_sessions:
            yield day, _build(history, bucket_ns, n_buckets,
                              min_observations, len(excluded))
        history[day] = by_day[day]


def _apply(
    samples: Sequence[Sample],
    session: Session,
    profiles: Dict[date, SessionProfile],
) -> List[Sample]:
    out = []
    for s in samples:
        if not in_session(s.ts_ns, session):
            continue
        day = session_date(s.ts_ns, session)
        shape = profiles.get(day)
        if shape is None:
            continue
        open_ns, _ = session_bounds(day, session)
        bucket, level = shape.at(s.ts_ns - open_ns), shape.level
        if bucket is None or level is None or bucket == 0:
            continue
        # value / (bucket / level), rearranged: dividing by a factor that is
        # itself a quotient rounds twice, and this form rounds once. The
        # difference is a last digit, and a library that argues about last
        # digits elsewhere should not spend one here for nothing.
        out.append(Sample(s.ts_ns, s.value * level / bucket))
    return out


def deseasonalise(
    samples: Iterable[Sample],
    session: Session,
    *,
    bucket_ns: int,
    min_sessions: int,
    min_observations: int = 1,
    calendar: Optional[TradingCalendar] = None,
    include_short: bool = False,
) -> List[Sample]:
    """Divide out the shape of the day, using only what was knowable.

    Each sample is divided by the factor its own day's expanding profile
    gives for that offset. Samples in the first ``min_sessions`` days come
    back nowhere in the output, and neither do samples whose bucket had no
    factor — a dropped point is visible, whereas a point silently divided by
    one is a point that claims to have been adjusted.
    """
    rows = list(samples)
    profiles = dict(expanding_profiles(
        rows, session, bucket_ns=bucket_ns, min_sessions=min_sessions,
        min_observations=min_observations, calendar=calendar,
        include_short=include_short))
    return _apply(rows, session, profiles)


def full_sample_deseasonalise(
    samples: Iterable[Sample],
    session: Session,
    *,
    bucket_ns: int,
    min_observations: int = 1,
    calendar: Optional[TradingCalendar] = None,
    include_short: bool = False,
) -> List[Sample]:
    """The same adjustment done the usual way — one profile over everything.

    Kept so the two can be compared on one input. It is not deprecated and it
    is not a trap: for describing how a venue's day is shaped it is the
    better estimate, because it uses every observation. It is only wrong as
    an input to something that trades.
    """
    rows = list(samples)
    shape = session_profile(rows, session, bucket_ns=bucket_ns,
                    min_observations=min_observations, calendar=calendar,
                    include_short=include_short)
    days = {session_date(s.ts_ns, session) for s in rows
            if in_session(s.ts_ns, session)}
    return _apply(rows, session, {d: shape for d in days})


def profile_leak(
    samples: Iterable[Sample],
    session: Session,
    *,
    bucket_ns: int,
    min_sessions: int,
    min_observations: int = 1,
    tolerance: Decimal = Decimal("0.01"),
    calendar: Optional[TradingCalendar] = None,
    include_short: bool = False,
) -> ProfileLeak:
    """Measure what the full-sample profile claims over the knowable one.

    For every sample that both approaches can adjust, the two factors are
    compared and the relative gap recorded. ``tolerance`` is the size of
    disagreement worth counting — the default of one per cent is a starting
    point rather than a recommendation, because whether a one per cent shift
    in a normalisation matters depends entirely on what is downstream of it.

    The number to look at first is usually ``max_gap`` together with
    ``worst_ts_ns``: the largest disagreements cluster in the early sessions,
    which is exactly where a full-sample profile is drawing on the most
    future.
    """
    rows = [s for s in samples if in_session(s.ts_ns, session)]
    full = session_profile(rows, session, bucket_ns=bucket_ns,
                   min_observations=min_observations, calendar=calendar,
                   include_short=include_short)
    knowable = dict(expanding_profiles(
        rows, session, bucket_ns=bucket_ns, min_sessions=min_sessions,
        min_observations=min_observations, calendar=calendar,
        include_short=include_short))

    gaps: List[Decimal] = []
    differ = 0
    worst: Optional[Decimal] = None
    worst_ts: Optional[int] = None
    for s in rows:
        day = session_date(s.ts_ns, session)
        shape = knowable.get(day)
        if shape is None:
            continue
        open_ns, _ = session_bounds(day, session)
        offset = s.ts_ns - open_ns
        a, b = full.factor_at(offset), shape.factor_at(offset)
        if a is None or b is None or b == 0:
            continue
        gap = abs(a - b) / b
        gaps.append(gap)
        if gap > tolerance:
            differ += 1
        if worst is None or gap > worst:
            worst, worst_ts = gap, s.ts_ns

    gaps.sort()
    median = gaps[(len(gaps) - 1) // 2] if gaps else None
    return ProfileLeak(samples=len(rows), knowable=len(gaps), differ=differ,
                       median_gap=median, max_gap=worst, worst_ts_ns=worst_ts)


def read_samples_csv(
    path: str,
    *,
    ts_column: str = "ts_ns",
    value_column: str = "value",
) -> List[Sample]:
    """Read ``ts_ns,value`` rows into samples.

    A row missing either field is an error rather than a skip. A profile is
    an average, and a reader that quietly drops rows changes the average
    without saying so.
    """
    import csv

    from .fileio import open_text

    out: List[Sample] = []
    with open_text(path) as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                ts = int(row[ts_column])
                value = Decimal((row[value_column] or "").strip())
            except (KeyError, TypeError, ValueError, ArithmeticError):
                raise ValueError(
                    f"line {i}: both {ts_column} and {value_column} are "
                    "required and must parse")
            out.append(Sample(ts, value))
    if not out:
        raise ValueError("no samples in file")
    return out
