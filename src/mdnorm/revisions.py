"""Values that get corrected later, and the two different questions about them.

Every observation in this library so far has had one timestamp: when it
happened. A great deal of real data has two. A figure describes one period and
becomes knowable at another, and then it is revised — an exchange busts a
trade, a vendor backfills a gap, a statistical agency restates last quarter::

    from mdnorm import Revision, RevisionSeries

    series = RevisionSeries([
        Revision(event_ts_ns=q1, known_ts_ns=april, value=D("2.1")),
        Revision(event_ts_ns=q1, known_ts_ns=may,   value=D("1.6")),   # revised
    ])
    series.as_of(event_ts_ns=q1, known_ts_ns=april_20)   # 2.1, what you knew
    series.final(event_ts_ns=q1)                          # 1.6, what is true now

**Using the corrected value is look-ahead, and the timestamp does not show
it.** This is what makes revisions worse than an ordinary alignment mistake.
The row is dated correctly, the value is a real number that was genuinely
published, and nothing in the data indicates that the version sitting there was
not available until three weeks later. Every guard in :mod:`mdnorm.align`
passes. The study is still wrong.

**There are two honest questions and they need different objects.** "What was
the newest published number at time t" is a feature: use :meth:`known_series`,
which is keyed by publication time and can be joined like any other stream.
"What did the whole dataset look like at time t" is a vintage: use
:meth:`vintage_at`, which is keyed by event time and reproduces the table as it
appeared that day. Mixing them up produces a series that is correct row by row
and impossible in aggregate.

**How much this matters is measurable, so measure it.**
:meth:`revision_summary` reports how many events were ever revised and how far
the first release sat from the final value. On data where that number is small
the distinction is academic. On data where it is not, every backtest built on
final values has been reading answers.

A value cannot be known before the period it describes has started, so
``known_ts_ns`` must not precede ``event_ts_ns``. Something known in advance is
a forecast, which is a different object with different properties.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, Iterable, List, Optional, Tuple

from .align import AsOfSeries
from .features import _PRECISION
from .fileio import open_text
from .timeutil import epoch_to_ns, iso_to_ns

__all__ = [
    "Revision",
    "RevisionSummary",
    "RevisionSeries",
    "read_revisions_csv",
]


@dataclass(frozen=True, slots=True)
class Revision:
    """One published version of one observation.

    ``event_ts_ns`` is the period the value describes; ``known_ts_ns`` is when
    that version became available. The same event may appear many times with
    different ``known_ts_ns`` — that is what a revision is.
    """

    event_ts_ns: int
    known_ts_ns: int
    value: Decimal

    def __post_init__(self) -> None:
        if self.event_ts_ns < 0 or self.known_ts_ns < 0:
            raise ValueError("timestamps must be non-negative")
        if self.known_ts_ns < self.event_ts_ns:
            raise ValueError(
                "known_ts_ns cannot precede event_ts_ns; a value available "
                "before the period it describes is a forecast, not a revision"
            )


@dataclass(frozen=True, slots=True)
class RevisionSummary:
    """How much the corrections in a series actually move the numbers."""

    events: int
    revised_events: int
    max_absolute_change: Optional[Decimal]
    mean_absolute_change: Optional[Decimal]

    @property
    def revised_fraction(self) -> Optional[Decimal]:
        if self.events == 0:
            return None
        return Decimal(self.revised_events) / self.events


class RevisionSeries:
    """Every published version of every observation, queryable both ways."""

    __slots__ = ("name", "_by_event")

    def __init__(self, revisions: Iterable[Revision], *, name: str = "") -> None:
        self.name = name
        grouped: Dict[int, Dict[int, Decimal]] = {}
        for r in revisions:
            # Two versions stamped the same instant: the later input wins, the
            # same rule the rest of the library uses for duplicate timestamps.
            grouped.setdefault(r.event_ts_ns, {})[r.known_ts_ns] = r.value
        self._by_event: Dict[int, Tuple[List[int], List[Decimal]]] = {}
        for event, versions in grouped.items():
            known = sorted(versions)
            self._by_event[event] = (known, [versions[k] for k in known])

    # -- one observation ---------------------------------------------------

    @property
    def events(self) -> Tuple[int, ...]:
        """Every event timestamp present, in time order."""
        return tuple(sorted(self._by_event))

    def as_of(self, *, event_ts_ns: int, known_ts_ns: int) -> Optional[Decimal]:
        """The version of ``event_ts_ns`` that was published by ``known_ts_ns``.

        ``None`` when nothing about that event had been released yet, which is
        the honest answer rather than the first release brought forward.
        """
        entry = self._by_event.get(event_ts_ns)
        if entry is None:
            return None
        known, values = entry
        i = bisect_right(known, known_ts_ns)
        return values[i - 1] if i else None

    def first_release(self, event_ts_ns: int) -> Optional[Decimal]:
        """The value as it was first published, before any correction."""
        entry = self._by_event.get(event_ts_ns)
        return entry[1][0] if entry else None

    def final(self, event_ts_ns: int) -> Optional[Decimal]:
        """The newest version. Correct today, and unavailable at the time."""
        entry = self._by_event.get(event_ts_ns)
        return entry[1][-1] if entry else None

    def revision_count(self, event_ts_ns: int) -> int:
        """How many times the value changed after its first release."""
        entry = self._by_event.get(event_ts_ns)
        if entry is None:
            return 0
        values = entry[1]
        return sum(1 for a, b in zip(values, values[1:]) if a != b)

    def published_at(self, event_ts_ns: int) -> Tuple[int, ...]:
        """The times at which versions of this event appeared."""
        entry = self._by_event.get(event_ts_ns)
        return tuple(entry[0]) if entry else ()

    # -- the whole series --------------------------------------------------

    def vintage_at(self, known_ts_ns: int, *, name: str = "") -> AsOfSeries:
        """The dataset as it looked at ``known_ts_ns``, keyed by event time.

        This is the table someone would have printed that day: every event
        that had been released, each showing the version current then. Use it
        to reproduce a historical study, not to build a feature — a value keyed
        by event time says nothing about when it could be read.
        """
        rows = []
        for event in self._by_event:
            value = self.as_of(event_ts_ns=event, known_ts_ns=known_ts_ns)
            if value is not None:
                rows.append((event, value))
        return AsOfSeries(rows, name=name or self.name)

    def known_series(self, *, name: str = "") -> AsOfSeries:
        """The publication stream, keyed by when each version became knowable.

        At any moment this gives the newest number that had actually been
        released by then, which is what a feature is allowed to see. Joining
        it through :func:`mdnorm.align` is safe for the same reason every other
        stream is: the key is the time you could read it.
        """
        rows = []
        for event, (known, values) in self._by_event.items():
            for k, v in zip(known, values):
                rows.append((k, v))
        return AsOfSeries(rows, name=name or self.name)

    def revision_summary(self) -> RevisionSummary:
        """How far first releases sit from final values.

        A large number here means every study built on final values has been
        reading corrections that were not available at the time. A small one
        means the distinction is academic for this dataset — which is worth
        knowing rather than assuming in either direction.
        """
        deltas: List[Decimal] = []
        revised = 0
        for event in self._by_event:
            first = self.first_release(event)
            last = self.final(event)
            if first is None or last is None:
                continue
            if first != last:
                revised += 1
            deltas.append(abs(last - first))
        if not deltas:
            return RevisionSummary(len(self._by_event), 0, None, None)
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            mean = sum(deltas, Decimal(0)) / len(deltas)
        return RevisionSummary(
            events=len(self._by_event),
            revised_events=revised,
            max_absolute_change=max(deltas),
            mean_absolute_change=mean,
        )

    def __len__(self) -> int:
        return len(self._by_event)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RevisionSeries(name={self.name!r}, events={len(self._by_event)})"


def read_revisions_csv(
    path: str, *, ts_unit: Optional[str] = None
) -> List[Revision]:
    """Read ``event,known,value`` rows into :class:`Revision` objects.

    Timestamps are ISO-8601 unless ``ts_unit`` selects epoch parsing, matching
    the rest of the CLI. ``event_ts``/``known_ts`` are accepted as column names
    too, since both spellings turn up in vendor files.
    """
    import csv as _csv

    out: List[Revision] = []
    with open_text(path) as fh:
        for lineno, row in enumerate(_csv.DictReader(fh), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                event = (row.get("event") or row.get("event_ts") or "").strip()
                known = (row.get("known") or row.get("known_ts") or "").strip()
                value = (row.get("value") or "").strip()

                def parse(text: str) -> int:
                    return (epoch_to_ns(float(text), ts_unit) if ts_unit
                            else iso_to_ns(text))

                out.append(Revision(event_ts_ns=parse(event),
                                    known_ts_ns=parse(known),
                                    value=Decimal(value)))
            except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out
