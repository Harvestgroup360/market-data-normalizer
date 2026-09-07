"""A price you could not have traded at is not a price you traded at.

When an instrument is halted the tape goes quiet, and a backtest reading that
tape sees nothing unusual: the last print stands, the features keep updating
off it, and the next print arrives at a price that moved while nobody could
act. Every fill placed in that silence is a fill that could not have
happened::

    from mdnorm import halt_report, reopen_gaps, unfillable

    halt_report(events, halts).halted_share      # 3.1% of the session
    reopen_gaps(events, halts)[0].move_bps       # -1,840 bps across one pause
    unfillable(decisions, halts).value_share     # 34% of the P&L

**Halts are concentrated in exactly the wrong place.** An instrument is not
paused on a quiet afternoon. It is paused on the day of the earnings leak,
the guidance cut, the tender offer — the days with the largest moves in the
sample. So the fills a backtest invents during halts are not a random slice
of its trades; they are drawn from the fattest part of the tail, and they are
on the right side of it, because the strategy is reading a price that has not
yet absorbed the news.

**The reopening move belongs to nobody.** A stock halted at 41.20 and reopened
at 33.60 fell eighteen per cent without a single tradable print in between. A
strategy holding through it took the loss. A strategy that "entered" during
the pause did not: it entered at the stale price and was marked at the new
one, which is not a trade, it is a gift. :func:`reopen_gaps` reports every one
of those moves so the size of the gift can be seen.

**Prints during a halt are a feed problem, and they are named as such.** A
trade stamped inside a halt window is usually a late report of something that
executed before the pause, occasionally a cross that is allowed to print, and
sometimes a vendor with a broken clock. This module counts them and does not
guess which.

**Nothing is inferred.** There is no rule here that a long enough quiet
stretch is a halt. On an illiquid name that rule fires constantly, and the
resulting statistic is a property of the threshold rather than of the market
— see :mod:`mdnorm.staleness`, which counts flat stretches and equally
refuses to interpret them. Either the halt windows are supplied, or the
report says it has none.

Nothing here deletes anything: :func:`split_halted` hands back both halves.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import EventType, MarketEvent

__all__ = [
    "HaltKind",
    "Halt",
    "Decision",
    "ReopenGap",
    "HaltReport",
    "Unfillable",
    "halted",
    "split_halted",
    "exclude_halted",
    "halt_report",
    "reopen_gaps",
    "unfillable",
    "read_halts_csv",
]

_ZERO = Decimal(0)


class HaltKind(str, Enum):
    """Why the instrument stopped trading, as reported by the source."""

    REGULATORY = "regulatory"
    VOLATILITY = "volatility"
    LIMIT = "limit"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Halt:
    """A half-open window ``[start_ns, end_ns)`` during which one symbol
    could not be traded.

    Half-open like every other interval in this library, which here means a
    print stamped exactly at the reopening time is tradable and a print
    stamped exactly at the halt time is not. That is the correct reading: the
    reopening auction is the first thing that can be traded, and the instant
    the pause begins is already inside it.
    """

    symbol: str
    start_ns: int
    end_ns: int
    kind: HaltKind = HaltKind.UNKNOWN
    reason: str = ""

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("a halt ends after it starts")

    def __contains__(self, ts_ns: int) -> bool:
        return self.start_ns <= ts_ns < self.end_ns

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True, slots=True)
class Decision:
    """A moment at which a strategy would have acted, and what it was worth.

    ``value`` is whatever the caller is measuring — notional, position
    change, realised P&L. It is never interpreted, only summed, so the share
    it produces means whatever the input meant.
    """

    ts_ns: int
    symbol: str
    value: Decimal = _ZERO


@dataclass(frozen=True, slots=True)
class ReopenGap:
    """The move across a halt, which happened without a tradable price."""

    halt: Halt
    last_before: Optional[Decimal]
    first_after: Optional[Decimal]

    @property
    def move(self) -> Optional[Decimal]:
        if self.last_before is None or self.first_after is None:
            return None
        return self.first_after - self.last_before

    @property
    def move_bps(self) -> Optional[Decimal]:
        """The move in basis points of the last price before the halt."""
        move = self.move
        if move is None or not self.last_before:
            return None
        return move / self.last_before * 10_000


@dataclass(frozen=True, slots=True)
class HaltReport:
    """How much of a sample sat inside a halt, and what printed anyway."""

    halts: int
    symbols: int
    halted_ns: int
    covered_ns: int
    events: int
    events_during: int
    longest_ns: int

    @property
    def halted_share(self) -> Optional[Decimal]:
        """Halted time as a share of the span the events cover.

        Summed over symbols, so two symbols halted for the same hour count
        as two halted hours. On a single-symbol file it reads as wall-clock
        time; on a basket it reads as instrument-time, which is the quantity
        a portfolio is actually exposed to.
        """
        if self.covered_ns <= 0:
            return None
        return Decimal(self.halted_ns) / self.covered_ns

    @property
    def during_share(self) -> Optional[Decimal]:
        """Share of events stamped inside a halt window."""
        if self.events == 0:
            return None
        return Decimal(self.events_during) / self.events


@dataclass(frozen=True, slots=True)
class Unfillable:
    """How much of a strategy's activity landed where it could not trade."""

    decisions: int
    unfillable: int
    value: Decimal
    unfillable_value: Decimal
    unmatched_symbols: Tuple[str, ...] = ()

    @property
    def count_share(self) -> Optional[Decimal]:
        if self.decisions == 0:
            return None
        return Decimal(self.unfillable) / self.decisions

    @property
    def value_share(self) -> Optional[Decimal]:
        """The share that matters, and usually the larger of the two.

        Counts treat a decision worth a hundred dollars and one worth a
        hundred thousand as equal. A value share above a count share is the
        signature of the problem this module exists for: the decisions taken
        during halts were the big ones.
        """
        if not self.value:
            return None
        return self.unfillable_value / self.value


def _by_symbol(halts: Iterable[Halt]) -> Dict[str, List[Halt]]:
    out: Dict[str, List[Halt]] = {}
    for h in halts:
        out.setdefault(h.symbol, []).append(h)
    for v in out.values():
        v.sort(key=lambda h: h.start_ns)
    return out


def halted(
    ts_ns: int, symbol: str, halts: Sequence[Halt]
) -> Optional[Halt]:
    """The halt covering this instant for this symbol, or ``None``.

    A symbol with no halts in the input is not halted. That is not the same
    as a symbol whose halts were never loaded, and this function cannot tell
    the two apart — :func:`unfillable` reports which symbols it never saw a
    halt record for, so the difference is at least visible.
    """
    for h in halts:
        if h.symbol == symbol and ts_ns in h:
            return h
    return None


def split_halted(
    events: Iterable[MarketEvent], halts: Sequence[Halt]
) -> Tuple[List[MarketEvent], List[MarketEvent]]:
    """Return ``(tradable, during_halt)``, preserving input order in both."""
    index = _by_symbol(halts)
    tradable: List[MarketEvent] = []
    during: List[MarketEvent] = []
    for e in events:
        pool = index.get(e.symbol, ())
        if any(e.ts_ns in h for h in pool):
            during.append(e)
        else:
            tradable.append(e)
    return tradable, during


def exclude_halted(
    events: Iterable[MarketEvent], halts: Sequence[Halt]
) -> List[MarketEvent]:
    """The tradable half alone, for a pipeline that has decided."""
    return split_halted(events, halts)[0]


def halt_report(
    events: Sequence[MarketEvent], halts: Sequence[Halt]
) -> HaltReport:
    """Count halted time and the events that printed inside it.

    ``covered_ns`` is the span of the events themselves, per symbol, summed.
    Using the data's own extent rather than a session length keeps the share
    honest on a file that covers part of a day.
    """
    spans: Dict[str, List[int]] = {}
    for e in events:
        lo_hi = spans.get(e.symbol)
        if lo_hi is None:
            spans[e.symbol] = [e.ts_ns, e.ts_ns]
        else:
            lo_hi[0] = min(lo_hi[0], e.ts_ns)
            lo_hi[1] = max(lo_hi[1], e.ts_ns)
    covered = sum(hi - lo for lo, hi in spans.values())

    _, during = split_halted(events, halts)
    return HaltReport(
        halts=len(halts),
        symbols=len({h.symbol for h in halts}),
        halted_ns=sum(h.duration_ns for h in halts),
        covered_ns=covered,
        events=len(list(events)),
        events_during=len(during),
        longest_ns=max((h.duration_ns for h in halts), default=0),
    )


def reopen_gaps(
    events: Sequence[MarketEvent], halts: Sequence[Halt]
) -> List[ReopenGap]:
    """The last tradable price before each halt and the first one after it.

    Prints stamped inside the window are excluded from both sides, because a
    late report of a pre-halt execution is not the price the market reopened
    at and treating it as one understates the gap.
    """
    trades: Dict[str, List[MarketEvent]] = {}
    for e in events:
        if e.event_type is EventType.TRADE and e.price is not None:
            trades.setdefault(e.symbol, []).append(e)
    for v in trades.values():
        v.sort(key=lambda e: e.ts_ns)

    out: List[ReopenGap] = []
    for h in sorted(halts, key=lambda x: (x.symbol, x.start_ns)):
        pool = trades.get(h.symbol, ())
        before = [e.price for e in pool if e.ts_ns < h.start_ns]
        after = [e.price for e in pool if e.ts_ns >= h.end_ns]
        out.append(ReopenGap(halt=h,
                             last_before=before[-1] if before else None,
                             first_after=after[0] if after else None))
    return out


def unfillable(
    decisions: Sequence[Decision], halts: Sequence[Halt]
) -> Unfillable:
    """How many decisions, and how much value, fell inside a halt.

    Value is summed in absolute terms on both sides, so a short and a long
    of the same size do not cancel each other into a reassuring zero.
    """
    index = _by_symbol(halts)
    n = 0
    total = _ZERO
    bad = _ZERO
    unmatched = set()
    for d in decisions:
        pool = index.get(d.symbol)
        if pool is None:
            unmatched.add(d.symbol)
            pool = []
        total += abs(d.value)
        if any(d.ts_ns in h for h in pool):
            n += 1
            bad += abs(d.value)
    return Unfillable(decisions=len(decisions), unfillable=n, value=total,
                      unfillable_value=bad,
                      unmatched_symbols=tuple(sorted(unmatched)))


def read_halts_csv(
    path: str,
    *,
    symbol_column: str = "symbol",
    start_column: str = "start",
    end_column: str = "end",
    kind_column: str = "kind",
    reason_column: str = "reason",
) -> List[Halt]:
    """Read halt windows from ``symbol,start,end[,kind,reason]`` rows.

    ``kind`` is free text in most feeds; anything this module does not
    recognise becomes :attr:`HaltKind.UNKNOWN` rather than an error, since a
    vendor inventing a new code should not stop a pipeline that only needs
    the window.
    """
    import csv

    from .fileio import open_text

    out: List[Halt] = []
    with open_text(path) as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                raw = (row.get(kind_column) or "").strip().lower()
                try:
                    kind = HaltKind(raw)
                except ValueError:
                    kind = HaltKind.UNKNOWN
                out.append(Halt(
                    symbol=row[symbol_column],
                    start_ns=int(row[start_column]),
                    end_ns=int(row[end_column]),
                    kind=kind,
                    reason=(row.get(reason_column) or "").strip(),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {i}: {exc}")
    if not out:
        raise ValueError("no halts in file")
    return out
