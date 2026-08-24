"""A ticker is not an identifier.

:mod:`mdnorm.symbols` makes ``BTCUSDT`` and ``XBT/USD`` agree on a spelling.
That is a different problem from this one: the same spelling, at two different
times, meaning two different things.

Exchanges reuse ticker strings. A company delists and its symbol is reassigned
to an unrelated one; a fund closes and its letters come back on a new
instrument; a venue renames a pair and the old name reappears elsewhere. A
price history keyed on the string splices the two together, and the result is a
continuous series with no gap, no duplicate, and no error — the two halves are
simply about different companies::

    from mdnorm import SymbolAssignment, SymbolMap, key_by_instrument

    smap = SymbolMap([
        SymbolAssignment("ABC", "US0000000001", start_ns=t0, end_ns=t1),
        SymbolAssignment("ABC", "US0000000002", start_ns=t2),   # reused later
    ])
    smap.reused_symbols()          # [("ABC", 2)] — the finding
    rows, report = key_by_instrument(rows, smap)

**The bias is a join, not a bad value.** Every price in a spliced series
genuinely traded, at the timestamp it carries, under the ticker it carries.
Nothing in the data is wrong. What is wrong is the assumption that the column
header names one thing, and that assumption is made once, silently, when the
matrix is built.

**Reuse looks like a merger, and mergers look profitable.** A delisting is
usually a fall and a new listing usually starts at a normal price, so splicing
one onto the other inserts a jump. Half the time that jump is upward, and an
upward jump in a name your model was already holding is indistinguishable from
a takeover premium. The series does not look broken; it looks lucky.

**An interval, not a date.** :class:`SymbolAssignment` binds a symbol to an
instrument over a half-open interval, the same convention as
:class:`mdnorm.universe.Listing`: ``start_ns`` inclusive, ``end_ns`` exclusive.
Two assignments of one symbol that overlap in time are a contradiction rather
than something to merge, and constructing a :class:`SymbolMap` from them
raises.

**Refusing is better than warning.** :func:`series_segments` splits a series at
every point where its ticker changed instrument, and :func:`key_by_instrument`
re-keys rows by the identifier that was in force at their own timestamp. Both
report what they did. A pipeline that carries on across a reuse boundary with a
warning in a log is a pipeline that will do it again next quarter.

None of this can be inferred from prices. It comes from a reference-data file,
and this module is about applying one correctly rather than about producing it.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SymbolAssignment",
    "SymbolMap",
    "SymbolMapReport",
    "Segment",
    "key_by_instrument",
    "series_segments",
    "read_symbol_map_csv",
]


@dataclass(frozen=True, slots=True)
class SymbolAssignment:
    """One ticker bound to one instrument over a half-open interval.

    ``instrument_id`` is whatever stable identifier you trust — an ISIN, a
    FIGI, an internal key. It only has to outlive the ticker, which is the one
    thing a ticker does not do.

    ``end_ns`` is exclusive and ``None`` means the binding is still current as
    far as this record knows. As with :class:`mdnorm.universe.Listing`, that is
    a statement about the record rather than about the instrument.
    """

    symbol: str
    instrument_id: str
    start_ns: int
    end_ns: Optional[int] = None
    venue: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        if self.end_ns is not None and self.end_ns <= self.start_ns:
            raise ValueError(
                f"end_ns must be greater than start_ns for {self.symbol!r} "
                f"({self.instrument_id})"
            )

    def covers(self, ts_ns: int) -> bool:
        """Whether this binding was in force at ``ts_ns``."""
        if ts_ns < self.start_ns:
            return False
        return self.end_ns is None or ts_ns < self.end_ns

    @property
    def open_ended(self) -> bool:
        return self.end_ns is None


@dataclass(frozen=True, slots=True)
class SymbolMapReport:
    """What a symbol map contains, and where it is likely to bite.

    ``reused_symbols`` is the figure to look at first. If it is zero over a
    long history, either the market genuinely never recycled a ticker, or the
    file describes today's assignments only — and the second is far more
    common, which makes a zero here the same kind of finding as a purge that
    removes nothing.
    """

    assignments: int
    symbols: int
    instruments: int
    reused_symbols: int
    renamed_instruments: int
    open_ended: int


class SymbolMap:
    """Point-in-time ticker to instrument resolution.

    Overlapping assignments for one symbol are rejected on construction: a
    ticker that pointed at two instruments simultaneously is a broken
    reference file, and quietly picking one of them is how the error survives
    into a study.
    """

    __slots__ = ("_by_symbol", "_starts", "_by_instrument", "_count")

    def __init__(self, assignments: Iterable[SymbolAssignment]) -> None:
        by_symbol: Dict[str, List[SymbolAssignment]] = {}
        by_instrument: Dict[str, List[SymbolAssignment]] = {}
        count = 0
        for a in assignments:
            by_symbol.setdefault(a.symbol, []).append(a)
            by_instrument.setdefault(a.instrument_id, []).append(a)
            count += 1

        for symbol, group in by_symbol.items():
            group.sort(key=lambda a: a.start_ns)
            for earlier, later in zip(group, group[1:]):
                if earlier.end_ns is None or earlier.end_ns > later.start_ns:
                    raise ValueError(
                        f"overlapping assignments for symbol {symbol!r}: "
                        f"{earlier.instrument_id} is open or runs past the "
                        f"start of {later.instrument_id}. A ticker cannot name "
                        f"two instruments at the same time; close the earlier "
                        f"binding."
                    )
        for group in by_instrument.values():
            group.sort(key=lambda a: a.start_ns)

        self._by_symbol = by_symbol
        self._starts = {s: [a.start_ns for a in g] for s, g in by_symbol.items()}
        self._by_instrument = by_instrument
        self._count = count

    def __len__(self) -> int:
        return self._count

    @property
    def symbols(self) -> List[str]:
        return sorted(self._by_symbol)

    @property
    def instruments(self) -> List[str]:
        return sorted(self._by_instrument)

    # -- resolution ----------------------------------------------------------

    def assignment_at(self, symbol: str, ts_ns: int) -> Optional[SymbolAssignment]:
        """The binding in force for ``symbol`` at ``ts_ns``, or ``None``.

        ``None`` means the ticker named nothing at that moment as far as this
        map knows — before its first assignment, or inside a gap between two.
        It is deliberately not the nearest binding: a value stamped before a
        ticker existed belongs nowhere, and attaching it to the next owner is
        the splice this module exists to prevent.
        """
        group = self._by_symbol.get(symbol)
        if not group:
            return None
        i = bisect_right(self._starts[symbol], ts_ns) - 1
        if i < 0:
            return None
        candidate = group[i]
        return candidate if candidate.covers(ts_ns) else None

    def instrument_at(self, symbol: str, ts_ns: int) -> Optional[str]:
        """The instrument ``symbol`` named at ``ts_ns``, or ``None``."""
        a = self.assignment_at(symbol, ts_ns)
        return None if a is None else a.instrument_id

    def symbol_at(self, instrument_id: str, ts_ns: int) -> Optional[str]:
        """The ticker an instrument carried at ``ts_ns``, or ``None``.

        The inverse question, and the one a report needs: labelling a 2019 row
        with the name the instrument goes by today is how a chart ends up
        showing a company under a ticker it did not have.
        """
        for a in self._by_instrument.get(instrument_id, ()):
            if a.covers(ts_ns):
                return a.symbol
        return None

    def history(self, instrument_id: str) -> List[SymbolAssignment]:
        """Every ticker an instrument has carried, oldest first."""
        return list(self._by_instrument.get(instrument_id, ()))

    def assignments_of(self, symbol: str) -> List[SymbolAssignment]:
        """Every instrument a ticker has named, oldest first."""
        return list(self._by_symbol.get(symbol, ()))

    # -- diagnostics ---------------------------------------------------------

    def reused_symbols(self) -> List[Tuple[str, int]]:
        """Tickers that have named more than one instrument, with the count.

        This is the list that decides whether a price history keyed on symbol
        is usable at all.
        """
        out = []
        for symbol, group in self._by_symbol.items():
            distinct = {a.instrument_id for a in group}
            if len(distinct) > 1:
                out.append((symbol, len(distinct)))
        return sorted(out)

    def renamed_instruments(self) -> List[Tuple[str, int]]:
        """Instruments that have carried more than one ticker, with the count."""
        out = []
        for instrument, group in self._by_instrument.items():
            distinct = {a.symbol for a in group}
            if len(distinct) > 1:
                out.append((instrument, len(distinct)))
        return sorted(out)

    def report(self) -> SymbolMapReport:
        """Counts, including the ones that say the file may be present-day only."""
        return SymbolMapReport(
            assignments=self._count,
            symbols=len(self._by_symbol),
            instruments=len(self._by_instrument),
            reused_symbols=len(self.reused_symbols()),
            renamed_instruments=len(self.renamed_instruments()),
            open_ended=sum(1 for g in self._by_symbol.values()
                           for a in g if a.open_ended),
        )


# -- applying a map ----------------------------------------------------------


def key_by_instrument(
    rows: Sequence[Mapping[str, object]],
    symbol_map: SymbolMap,
    *,
    symbol_field: str = "symbol",
    ts_field: str = "ts_ns",
    target_field: str = "instrument_id",
    drop_unmapped: bool = True,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Re-key rows by the instrument their ticker named at their own timestamp.

    Returns the rewritten rows and a count of what happened: ``mapped``,
    ``unmapped`` and ``reassigned`` — the last being rows whose ticker pointed
    at a different instrument than the one it points at now. A ``reassigned``
    count above zero is the whole reason to do this.

    ``drop_unmapped`` removes rows the map cannot resolve. Keeping them is
    supported and is worse: a row whose instrument is unknown will be grouped
    with every other unknown one and treated as a single series.
    """
    latest: Dict[str, Optional[str]] = {}
    out: List[Dict[str, object]] = []
    counts = {"mapped": 0, "unmapped": 0, "reassigned": 0}

    for row in rows:
        symbol = row.get(symbol_field)
        ts = row.get(ts_field)
        if not isinstance(symbol, str) or not isinstance(ts, int):
            counts["unmapped"] += 1
            if not drop_unmapped:
                out.append(dict(row))
            continue
        instrument = symbol_map.instrument_at(symbol, ts)
        if instrument is None:
            counts["unmapped"] += 1
            if not drop_unmapped:
                out.append(dict(row))
            continue
        if symbol not in latest:
            group = symbol_map.assignments_of(symbol)
            latest[symbol] = group[-1].instrument_id if group else None
        if latest[symbol] is not None and instrument != latest[symbol]:
            counts["reassigned"] += 1
        counts["mapped"] += 1
        new = dict(row)
        new[target_field] = instrument
        out.append(new)

    return out, counts


@dataclass(frozen=True, slots=True)
class Segment:
    """A stretch of one ticker's history belonging to a single instrument."""

    symbol: str
    instrument_id: str
    start_index: int
    stop_index: int          # exclusive
    start_ns: int
    end_ns: int

    def __len__(self) -> int:
        return self.stop_index - self.start_index


def series_segments(
    symbol: str,
    timestamps: Sequence[int],
    symbol_map: SymbolMap,
) -> Tuple[List[Segment], int]:
    """Split a ticker's timestamps wherever it changed instrument.

    Returns the segments in time order and the number of timestamps the map
    could not resolve. More than one segment means the series must not be
    treated as one instrument — not that it needs a flag, but that any
    statistic spanning the boundary is meaningless, because it mixes two
    different companies.

    ``timestamps`` must be non-decreasing; a series that is out of order has a
    different problem and :func:`mdnorm.quality.find_issues` names it.
    """
    for a, b in zip(timestamps, timestamps[1:]):
        if b < a:
            raise ValueError("timestamps must be non-decreasing")

    segments: List[Segment] = []
    unresolved = 0
    current: Optional[str] = None
    start_i = 0
    start_ts = 0

    for i, ts in enumerate(timestamps):
        instrument = symbol_map.instrument_at(symbol, ts)
        if instrument is None:
            if current is not None:
                segments.append(Segment(symbol, current, start_i, i,
                                        start_ts, timestamps[i - 1]))
                current = None
            unresolved += 1
            continue
        if instrument != current:
            if current is not None:
                segments.append(Segment(symbol, current, start_i, i,
                                        start_ts, timestamps[i - 1]))
            current = instrument
            start_i = i
            start_ts = ts

    if current is not None and timestamps:
        segments.append(Segment(symbol, current, start_i, len(timestamps),
                                start_ts, timestamps[-1]))
    return segments, unresolved


# -- reading a reference file -------------------------------------------------


def read_symbol_map_csv(
    path: str,
    *,
    symbol_field: str = "symbol",
    instrument_field: str = "instrument_id",
    start_field: str = "start_ns",
    end_field: str = "end_ns",
    venue_field: str = "venue",
) -> List[SymbolAssignment]:
    """Read assignments from a CSV.

    An empty ``end_ns`` means still current. A row missing a symbol, an
    instrument or a start is an error rather than a row to skip: a reference
    file with holes in it produces a map that silently resolves fewer rows
    than you think, and the count of what failed to resolve is the only thing
    that would have told you.
    """
    import csv

    from .fileio import open_text

    out: List[SymbolAssignment] = []
    with open_text(path) as fh:
        for lineno, row in enumerate(csv.DictReader(fh), start=2):
            symbol = (row.get(symbol_field) or "").strip()
            instrument = (row.get(instrument_field) or "").strip()
            start = (row.get(start_field) or "").strip()
            if not symbol or not instrument or not start:
                raise ValueError(
                    f"{path}:{lineno}: needs {symbol_field}, "
                    f"{instrument_field} and {start_field}"
                )
            end = (row.get(end_field) or "").strip()
            venue = (row.get(venue_field) or "").strip() or None
            out.append(SymbolAssignment(
                symbol=symbol,
                instrument_id=instrument,
                start_ns=int(start),
                end_ns=int(end) if end else None,
                venue=venue,
            ))
    return out
