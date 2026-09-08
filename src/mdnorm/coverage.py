"""The absence of a row is not the absence of an event.

A feed that stops delivering looks exactly like a market that stops trading.
Both produce the same thing — nothing — and every downstream calculation
treats the silence as information::

    from mdnorm import coverage_report, panel_coverage

    rep = coverage_report(events, min_gap_ns=5 * MINUTE,
                          calendar=cal, halts=halts)
    rep.unexplained_ns          # 4h12m nobody has accounted for
    rep.explained_share         # 0.87
    panel_coverage(events, start, end, step_ns=DAY).full_share

**A gap has to be explained before it is a gap.** Most silences are ordinary:
the venue was shut, it was a Sunday, the instrument was halted. This module
subtracts those and reports what is left, because the residual is the only
part that says something about the feed. Handing back a raw gap count
without that subtraction produces a number that is mostly weekends.

**What is left is not a small correction.** A missing hour inside a session
becomes one long bar instead of sixty short ones, so the realised volatility
computed over it is understated and the return across it is attributed to
whichever period the grid decided to put it in. A missing day silently turns
a one-day return into a two-day return, which is a larger number in a series
of otherwise one-day numbers, and it lands wherever the data stopped.

**Names go missing when they are in trouble.** This is why the panel matters
more than the per-symbol figure. A cross-sectional rank or z-score computed
over "the instruments that printed" is computed over a universe whose size
changes, and the ones that drop out are rarely a random sample —
:func:`panel_coverage` counts the width at every point so the changes are
visible instead of averaged away.

**There is no default gap threshold.** Five minutes without a print is
remarkable on a liquid future and unremarkable on a corporate bond, so
``min_gap_ns`` is required. A threshold chosen for you turns the resulting
statistic into a property of that choice, which is the same objection
:mod:`mdnorm.halts` makes to inferring a pause and :mod:`mdnorm.staleness`
makes to interpreting a flat stretch.

**Without a calendar every night is a gap, and the report says so.** A
report built with no calendar carries ``calendar=False``: it is correct for a
venue that trades continuously and badly misleading for one that does not,
and the flag is there so the difference cannot be missed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .calendars import TradingCalendar
from .halts import Halt
from .schema import MarketEvent

__all__ = [
    "Gap",
    "CoverageReport",
    "PanelCoverage",
    "find_gaps",
    "explain_gaps",
    "coverage_report",
    "panel_coverage",
]

_NS_PER_S = 1_000_000_000


@dataclass(frozen=True, slots=True)
class Gap:
    """A stretch with no observations, split by what accounts for it.

    The three components are exhaustive and sum to :attr:`duration_ns`. They
    are times rather than labels because one gap is usually several things at
    once: a Friday-evening outage that runs into the weekend is part missing
    data and part closed venue, and calling the whole thing either one would
    be wrong.
    """

    symbol: str
    start_ns: int
    end_ns: int
    outside_session_ns: int = 0
    halt_ns: int = 0
    unexplained_ns: int = 0

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("a gap ends after it starts")

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def explained_ns(self) -> int:
        return self.outside_session_ns + self.halt_ns

    @property
    def is_explained(self) -> bool:
        """Whether nothing at all is left over."""
        return self.unexplained_ns == 0


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """What a feed delivered, against what was open."""

    symbols: int
    events: int
    span_ns: int
    gaps: int
    gap_ns: int
    outside_session_ns: int
    halt_ns: int
    unexplained_ns: int
    longest_unexplained_ns: int
    min_gap_ns: int
    calendar: bool = False

    @property
    def explained_share(self) -> Optional[Decimal]:
        """Share of gap time the calendar and the halts account for."""
        if self.gap_ns == 0:
            return None
        return Decimal(self.explained_ns) / self.gap_ns

    @property
    def explained_ns(self) -> int:
        return self.outside_session_ns + self.halt_ns

    @property
    def unexplained_share(self) -> Optional[Decimal]:
        """Unexplained time as a share of the span the events cover.

        Against the span rather than against the gaps, because the question
        is how much of the history is missing, not how much of the missing
        history is surprising.
        """
        if self.span_ns <= 0:
            return None
        return Decimal(self.unexplained_ns) / self.span_ns


@dataclass(frozen=True, slots=True)
class PanelCoverage:
    """How many instruments were present at each point of a grid."""

    points: int
    symbols: int
    counts: Tuple[int, ...]
    step_ns: int

    @property
    def full(self) -> int:
        """Grid points where every symbol printed."""
        return sum(1 for c in self.counts if c == self.symbols)

    @property
    def full_share(self) -> Optional[Decimal]:
        if self.points == 0:
            return None
        return Decimal(self.full) / self.points

    @property
    def narrowest(self) -> Optional[int]:
        return min(self.counts) if self.counts else None

    @property
    def widest(self) -> Optional[int]:
        return max(self.counts) if self.counts else None

    @property
    def median(self) -> Optional[int]:
        """The middle width, lower of the two on an even number of points."""
        if not self.counts:
            return None
        ordered = sorted(self.counts)
        return ordered[(len(ordered) - 1) // 2]


def find_gaps(
    events: Sequence[MarketEvent],
    *,
    min_gap_ns: int,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
) -> List[Gap]:
    """Every stretch of at least ``min_gap_ns`` with no observations.

    Per symbol, in timestamp order. ``min_gap_ns`` is required and has no
    default: what counts as a suspicious silence is a property of the
    instrument and the sampling interval, not of this library.

    Between consecutive events only, unless ``start_ns`` and ``end_ns`` state
    the period the sample was supposed to cover. **Give them.** Without them
    a feed that stops halfway through produces no gap at all — its last
    observation has nothing after it to be distant from — and a symbol that
    goes quiet and never returns is the one case worth catching, because
    instruments stop printing when something has happened to them. With the
    bounds, a symbol named in ``symbols`` that never appears is one gap the
    width of the whole period.
    """
    if min_gap_ns <= 0:
        raise ValueError("min_gap_ns must be positive")
    if start_ns is not None and end_ns is not None and end_ns <= start_ns:
        raise ValueError("end_ns must follow start_ns")

    by_symbol: Dict[str, List[int]] = {s: [] for s in (symbols or ())}
    for e in events:
        if start_ns is not None and e.ts_ns < start_ns:
            continue
        if end_ns is not None and e.ts_ns >= end_ns:
            continue
        if symbols is not None and e.symbol not in by_symbol:
            continue
        by_symbol.setdefault(e.symbol, []).append(e.ts_ns)

    out: List[Gap] = []
    for symbol in sorted(by_symbol):
        stamps = sorted(by_symbol[symbol])
        if start_ns is not None:
            stamps = [start_ns] + stamps
        if end_ns is not None:
            stamps = stamps + [end_ns]
        for a, b in zip(stamps, stamps[1:]):
            if b - a >= min_gap_ns:
                out.append(Gap(symbol=symbol, start_ns=a, end_ns=b,
                               unexplained_ns=b - a))
    out.sort(key=lambda g: (g.symbol, g.start_ns))
    return out


def _sessions_between(
    calendar: TradingCalendar, start_ns: int, end_ns: int
) -> List[Tuple[int, int]]:
    """Session spans overlapping ``[start_ns, end_ns)``, clipped to it."""
    first = datetime.fromtimestamp(start_ns // _NS_PER_S, tz=timezone.utc)
    last = datetime.fromtimestamp(end_ns // _NS_PER_S, tz=timezone.utc)
    day = first.astimezone(calendar.session.zone).date() - timedelta(days=1)
    stop = last.astimezone(calendar.session.zone).date() + timedelta(days=1)
    covers_first, covers_last = calendar.covers

    out: List[Tuple[int, int]] = []
    while day <= stop:
        if covers_first <= day <= covers_last:
            span = calendar.session_on(day)
            if span is not None:
                lo, hi = max(span[0], start_ns), min(span[1], end_ns)
                if lo < hi:
                    out.append((lo, hi))
        day += timedelta(days=1)
    return out


def _subtract(
    spans: Sequence[Tuple[int, int]], cuts: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """``spans`` with every part of ``cuts`` removed."""
    out = list(spans)
    for c_lo, c_hi in cuts:
        nxt: List[Tuple[int, int]] = []
        for lo, hi in out:
            if c_hi <= lo or c_lo >= hi:
                nxt.append((lo, hi))
                continue
            if lo < c_lo:
                nxt.append((lo, c_lo))
            if c_hi < hi:
                nxt.append((c_hi, hi))
        out = nxt
    return out


def _total(spans: Iterable[Tuple[int, int]]) -> int:
    return sum(hi - lo for lo, hi in spans)


def explain_gaps(
    gaps: Sequence[Gap],
    *,
    calendar: Optional[TradingCalendar] = None,
    halts: Sequence[Halt] = (),
) -> List[Gap]:
    """Attribute each gap to closed sessions, halts, and what is left.

    With no calendar every instant counts as trading time, which is right for
    a venue that never closes and wrong for one that does. The caller decides
    which they have; :func:`coverage_report` records the choice so a reader
    of the output can tell.
    """
    out: List[Gap] = []
    for g in gaps:
        whole = [(g.start_ns, g.end_ns)]
        if calendar is None:
            trading = whole
        else:
            trading = _sessions_between(calendar, g.start_ns, g.end_ns)
        outside = g.duration_ns - _total(trading)

        cuts = [(h.start_ns, h.end_ns) for h in halts if h.symbol == g.symbol]
        left = _subtract(trading, cuts)
        halted = _total(trading) - _total(left)

        out.append(Gap(symbol=g.symbol, start_ns=g.start_ns, end_ns=g.end_ns,
                       outside_session_ns=outside, halt_ns=halted,
                       unexplained_ns=_total(left)))
    return out


def coverage_report(
    events: Sequence[MarketEvent],
    *,
    min_gap_ns: int,
    calendar: Optional[TradingCalendar] = None,
    halts: Sequence[Halt] = (),
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    symbols: Optional[Sequence[str]] = None,
) -> CoverageReport:
    """Find the gaps, explain what can be explained, total the rest.

    ``span_ns`` is the stated period times the number of symbols when bounds
    are given, and otherwise each symbol's own extent summed. Either way it
    reads as instrument-time: two names over a week is two weeks, which is
    the quantity a panel is actually built from.

    Pass ``start_ns`` and ``end_ns`` whenever the sample was meant to cover a
    known period. They are what makes a feed that stopped visible.
    """
    explained = explain_gaps(
        find_gaps(events, min_gap_ns=min_gap_ns, start_ns=start_ns,
                  end_ns=end_ns, symbols=symbols),
        calendar=calendar, halts=halts)

    spans: Dict[str, List[int]] = {s: [] for s in (symbols or ())}
    for e in events:
        cur = spans.get(e.symbol)
        if symbols is not None and cur is None:
            continue
        if not cur:
            spans[e.symbol] = [e.ts_ns, e.ts_ns]
        else:
            cur[0] = min(cur[0], e.ts_ns)
            cur[1] = max(cur[1], e.ts_ns)
    if start_ns is not None and end_ns is not None:
        spans = {s: [start_ns, end_ns] for s in spans}

    return CoverageReport(
        symbols=len(spans),
        events=len(events),
        span_ns=sum(v[1] - v[0] for v in spans.values() if len(v) == 2),
        gaps=len(explained),
        gap_ns=sum(g.duration_ns for g in explained),
        outside_session_ns=sum(g.outside_session_ns for g in explained),
        halt_ns=sum(g.halt_ns for g in explained),
        unexplained_ns=sum(g.unexplained_ns for g in explained),
        longest_unexplained_ns=max((g.unexplained_ns for g in explained),
                                   default=0),
        min_gap_ns=min_gap_ns,
        calendar=calendar is not None,
    )


def panel_coverage(
    events: Sequence[MarketEvent],
    *,
    start_ns: int,
    end_ns: int,
    step_ns: int,
    symbols: Optional[Sequence[str]] = None,
) -> PanelCoverage:
    """Count how many symbols printed in each ``[t, t + step_ns)`` bucket.

    ``symbols`` states the intended universe. Left out, it is taken from the
    events, which measures how the width of the panel moved but cannot see a
    symbol that was absent for the entire period — and a name missing from
    the whole sample is the one worth knowing about.
    """
    if step_ns <= 0:
        raise ValueError("step_ns must be positive")
    if end_ns <= start_ns:
        raise ValueError("end_ns must follow start_ns")

    universe = (list(dict.fromkeys(symbols)) if symbols is not None
                else sorted({e.symbol for e in events}))
    index = {s: i for i, s in enumerate(universe)}
    n_points = -(-(end_ns - start_ns) // step_ns)     # ceiling division

    seen: List[set] = [set() for _ in range(n_points)]
    for e in events:
        if e.ts_ns < start_ns or e.ts_ns >= end_ns:
            continue
        i = index.get(e.symbol)
        if i is None:
            continue
        seen[(e.ts_ns - start_ns) // step_ns].add(i)

    return PanelCoverage(points=n_points, symbols=len(universe),
                         counts=tuple(len(s) for s in seen), step_ns=step_ns)
