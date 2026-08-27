"""Building an index membership record out of the files a vendor actually ships.

:mod:`mdnorm.universe` applies a membership record: give it who was tradable
when, and it will keep a cross-section honest. Producing that record is a
separate job, and it is where survivorship gets in::

    from mdnorm import IndexChange, MembershipHistory, Basis

    history = MembershipHistory.from_changes(changes)
    history.members_at(t, basis=Basis.EFFECTIVE)   # who was in the index
    history.report()                                # what the file cannot say

**Two dates, and they are not interchangeable.** An index addition is
announced on one day and takes effect on another, usually a few days apart.
Both are real and they answer different questions. *Who was in the index* is
answered by the effective date. *When could a strategy have known this was
coming* is answered by the announcement. Research that ranks the index by
effective date and trades the announcement effect has used one timestamp to
justify a decision the other one governs, and the file contains no hint of it
because both columns are correct.

**A snapshot cannot express a deletion.** Most vendors ship periodic lists —
here are the members as of this date. Names that left do not appear as
departures; they simply stop being present, and the last day they were in the
file is not the day they left. Building a history from snapshots therefore
produces changes whose timing is *bounded* rather than known, and this module
says so: :meth:`MembershipHistory.from_snapshots` records the interval the
change fell inside and refuses to pick a point in it.

**Building history from the newest snapshot alone is the classic error.** Every
name in today's index is a name that survived to today. A study run on that
list will show the index outperforming itself, and nothing in the data looks
wrong — the prices are real and the dates are right.
:func:`survivorship_gap` measures the distance between a point-in-time
membership and a today-list, in names and in rows, so the size of the problem
is a number rather than a worry.

Membership here is by instrument identifier, not ticker. A ticker that names
one company today named another in 2019, which is the subject of
:mod:`mdnorm.instruments`; feeding tickers into an index history reintroduces
that error one level up.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from .fileio import open_text
from .timeutil import iso_to_ns
from .universe import Listing, Universe

__all__ = [
    "Basis",
    "ChangeKind",
    "IndexChange",
    "IndexSnapshot",
    "InferredChange",
    "MembershipHistory",
    "MembershipReport",
    "survivorship_gap",
    "read_index_changes_csv",
]


class ChangeKind(str, Enum):
    """Whether an instrument joined the index or left it."""

    ADD = "add"
    DELETE = "delete"


class Basis(str, Enum):
    """Which of the two dates a membership question is asked against.

    ``EFFECTIVE`` answers "was this instrument in the index" — the composition
    a benchmark was actually computed from. ``ANNOUNCED`` answers "was this
    change public knowledge" — what a strategy was allowed to act on. Neither
    is the default in any general sense, so this module makes you name one.
    """

    EFFECTIVE = "effective"
    ANNOUNCED = "announced"


@dataclass(frozen=True, slots=True)
class IndexChange:
    """One addition or deletion, carrying both of its dates.

    ``announced_ns`` may equal ``effective_ns`` when a vendor publishes only
    one date, but it must not follow it: a change cannot take effect before
    anyone was told, and a file that says otherwise has its columns swapped.
    """

    instrument_id: str
    kind: ChangeKind
    effective_ns: int
    announced_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.instrument_id:
            raise ValueError("instrument_id must not be empty")
        if self.effective_ns < 0:
            raise ValueError("effective_ns must be non-negative")
        if self.announced_ns is not None:
            if self.announced_ns < 0:
                raise ValueError("announced_ns must be non-negative")
            if self.announced_ns > self.effective_ns:
                raise ValueError(
                    f"{self.instrument_id}: announced_ns follows effective_ns; "
                    "a change cannot take effect before it was announced"
                )

    def ts_on(self, basis: Basis) -> int:
        """The timestamp this change carries under ``basis``."""
        if basis is Basis.ANNOUNCED and self.announced_ns is not None:
            return self.announced_ns
        return self.effective_ns


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """The members a vendor listed as of one moment."""

    as_of_ns: int
    members: FrozenSet[str]

    def __post_init__(self) -> None:
        if self.as_of_ns < 0:
            raise ValueError("as_of_ns must be non-negative")


@dataclass(frozen=True, slots=True)
class InferredChange:
    """A change deduced from two snapshots, with the window it happened in.

    Snapshots do not record when a name left, only that it is no longer there.
    ``earliest_ns`` is the snapshot that still showed the old state and
    ``latest_ns`` the one that showed the new; the change happened somewhere in
    between and this object declines to guess where.
    """

    instrument_id: str
    kind: ChangeKind
    earliest_ns: int
    latest_ns: int

    @property
    def uncertainty_ns(self) -> int:
        """How wide the window around this change is."""
        return self.latest_ns - self.earliest_ns


@dataclass(frozen=True, slots=True)
class MembershipReport:
    """What the source file can and cannot support."""

    instruments: int
    changes: int
    additions: int
    deletions: int
    dated_by_snapshot: int
    max_uncertainty_ns: Optional[int]
    without_announcement: int
    never_removed: int
    first_ns: Optional[int]
    last_ns: Optional[int]

    @property
    def announcement_coverage(self) -> Optional[float]:
        """Share of changes that carry an announcement date."""
        if self.changes == 0:
            return None
        return (self.changes - self.without_announcement) / self.changes


class MembershipHistory:
    """Who was in the index, when, on whichever basis you name."""

    __slots__ = ("name", "_changes", "_inferred")

    def __init__(
        self,
        changes: Iterable[IndexChange],
        *,
        inferred: Iterable[InferredChange] = (),
        name: str = "",
    ) -> None:
        self.name = name
        self._changes: List[IndexChange] = sorted(
            changes, key=lambda c: (c.effective_ns, c.instrument_id, c.kind.value))
        self._inferred: Tuple[InferredChange, ...] = tuple(inferred)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_changes(
        cls, changes: Iterable[IndexChange], *, name: str = ""
    ) -> "MembershipHistory":
        """Build from an add/delete file, which is the form worth asking for."""
        return cls(changes, name=name)

    @classmethod
    def from_snapshots(
        cls, snapshots: Iterable[IndexSnapshot], *, name: str = ""
    ) -> "MembershipHistory":
        """Build from periodic member lists, keeping the timing uncertain.

        Each difference between consecutive snapshots becomes a change dated at
        the later snapshot — the first moment the new state is known to hold —
        and an :class:`InferredChange` recording the window it actually fell
        in. Dating it at the later snapshot is the conservative direction: it
        never claims membership earlier than the file can support.

        The first snapshot is treated as additions at its own timestamp.
        Nothing before it is knowable from this input, and the report says so
        rather than extending the first list backwards.
        """
        snaps = sorted(snapshots, key=lambda s: s.as_of_ns)
        changes: List[IndexChange] = []
        inferred: List[InferredChange] = []
        previous: Optional[IndexSnapshot] = None
        for snap in snaps:
            if previous is None:
                for iid in sorted(snap.members):
                    changes.append(IndexChange(iid, ChangeKind.ADD, snap.as_of_ns))
                previous = snap
                continue
            added = snap.members - previous.members
            removed = previous.members - snap.members
            for iid in sorted(added):
                changes.append(IndexChange(iid, ChangeKind.ADD, snap.as_of_ns))
                inferred.append(InferredChange(iid, ChangeKind.ADD,
                                               previous.as_of_ns, snap.as_of_ns))
            for iid in sorted(removed):
                changes.append(IndexChange(iid, ChangeKind.DELETE, snap.as_of_ns))
                inferred.append(InferredChange(iid, ChangeKind.DELETE,
                                               previous.as_of_ns, snap.as_of_ns))
            previous = snap
        return cls(changes, inferred=inferred, name=name)

    # -- querying ----------------------------------------------------------

    def members_at(self, ts_ns: int, *, basis: Basis) -> Tuple[str, ...]:
        """The index composition at ``ts_ns``, in identifier order.

        A change applies from its timestamp onwards: an instrument added
        effective at ``t`` is a member at ``t``, and one deleted effective at
        ``t`` is not.
        """
        current: Set[str] = set()
        for change in sorted(self._changes, key=lambda c: c.ts_on(basis)):
            if change.ts_on(basis) > ts_ns:
                break
            if change.kind is ChangeKind.ADD:
                current.add(change.instrument_id)
            else:
                current.discard(change.instrument_id)
        return tuple(sorted(current))

    def is_member(self, instrument_id: str, ts_ns: int, *, basis: Basis) -> bool:
        """Whether one instrument was in the index at ``ts_ns``."""
        state = False
        for change in sorted(self._changes, key=lambda c: c.ts_on(basis)):
            if change.ts_on(basis) > ts_ns:
                break
            if change.instrument_id == instrument_id:
                state = change.kind is ChangeKind.ADD
        return state

    def intervals_of(self, instrument_id: str, *, basis: Basis
                     ) -> Tuple[Tuple[int, Optional[int]], ...]:
        """Every spell this instrument spent in the index, half-open.

        An open end means the record has no departure for it, which is a
        statement about the file rather than about the instrument.
        """
        spans: List[Tuple[int, Optional[int]]] = []
        start: Optional[int] = None
        for change in sorted(self._changes, key=lambda c: c.ts_on(basis)):
            if change.instrument_id != instrument_id:
                continue
            ts = change.ts_on(basis)
            if change.kind is ChangeKind.ADD:
                if start is None:
                    start = ts
            elif start is not None:
                spans.append((start, ts))
                start = None
        if start is not None:
            spans.append((start, None))
        return tuple(spans)

    def to_universe(self, *, basis: Basis) -> Universe:
        """Hand the history to :mod:`mdnorm.universe` as tradable lifetimes.

        Every spell becomes a :class:`~mdnorm.universe.Listing`, so a
        cross-section can be masked to the index as it stood rather than as it
        stands. Re-entries produce several listings for one identifier, which
        is what happened.
        """
        listings: List[Listing] = []
        for iid in self.instruments:
            for start, end in self.intervals_of(iid, basis=basis):
                listings.append(Listing(iid, start, end))
        return Universe(listings)

    @property
    def instruments(self) -> Tuple[str, ...]:
        """Every identifier the record has ever mentioned."""
        return tuple(sorted({c.instrument_id for c in self._changes}))

    @property
    def changes(self) -> Tuple[IndexChange, ...]:
        return tuple(self._changes)

    @property
    def inferred(self) -> Tuple[InferredChange, ...]:
        """Changes whose timing was deduced rather than stated."""
        return self._inferred

    def report(self) -> MembershipReport:
        """What this record supports, and where it is guessing.

        Two figures deserve attention. ``max_uncertainty_ns`` is how far a
        snapshot-derived change could be from where it is dated — on a monthly
        file that is a month, which is longer than many holding periods.
        ``never_removed`` counts identifiers with no departure on record; when
        that equals the whole index the file is a snapshot of today wearing a
        history's clothes, and cannot answer a point-in-time question at all.
        """
        adds = sum(1 for c in self._changes if c.kind is ChangeKind.ADD)
        dels = sum(1 for c in self._changes if c.kind is ChangeKind.DELETE)
        no_ann = sum(1 for c in self._changes if c.announced_ns is None)
        open_ended = sum(
            1 for iid in self.instruments
            if self.intervals_of(iid, basis=Basis.EFFECTIVE)
            and self.intervals_of(iid, basis=Basis.EFFECTIVE)[-1][1] is None)
        widths = [i.uncertainty_ns for i in self._inferred]
        return MembershipReport(
            instruments=len(self.instruments),
            changes=len(self._changes),
            additions=adds,
            deletions=dels,
            dated_by_snapshot=len(self._inferred),
            max_uncertainty_ns=max(widths) if widths else None,
            without_announcement=no_ann,
            never_removed=open_ended,
            first_ns=self._changes[0].effective_ns if self._changes else None,
            last_ns=self._changes[-1].effective_ns if self._changes else None,
        )

    def __len__(self) -> int:
        return len(self._changes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"MembershipHistory(name={self.name!r}, "
                f"changes={len(self._changes)}, "
                f"instruments={len(self.instruments)})")


def survivorship_gap(
    history: MembershipHistory,
    ts_ns: int,
    *,
    basis: Basis = Basis.EFFECTIVE,
    today_ns: Optional[int] = None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Compare the index as it stood with the index as it stands.

    Returns ``(missed, phantom)``. ``missed`` were members at ``ts_ns`` and are
    absent from the later list — the names a today-list study silently drops,
    and the ones that left for reasons that are rarely good. ``phantom`` are in
    the later list but were not members at ``ts_ns``, so a today-list study
    holds them before they joined.

    Both directions matter and they do not cancel. The first removes losers;
    the second adds winners early. A study using a today-list gets both at
    once, which is why the effect is large and one-directional.
    """
    then = set(history.members_at(ts_ns, basis=basis))
    end = today_ns
    if end is None:
        rep = history.report()
        end = rep.last_ns if rep.last_ns is not None else ts_ns
    now = set(history.members_at(end, basis=basis))
    return tuple(sorted(then - now)), tuple(sorted(now - then))


def read_index_changes_csv(
    path: str,
    *,
    instrument_column: str = "instrument_id",
    kind_column: str = "action",
    effective_column: str = "effective",
    announced_column: str = "announced",
    name: str = "",
) -> MembershipHistory:
    """Read an add/delete file into a history.

    Timestamps accept ISO-8601 or integer nanoseconds. The announcement column
    is optional and an empty cell is read as absent rather than as equal to the
    effective date — the two claims differ, and one of them is unverifiable.
    """
    import csv

    changes: List[IndexChange] = []
    with open_text(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_kind = row[kind_column].strip().lower()
            try:
                kind = ChangeKind(raw_kind)
            except ValueError:
                raise ValueError(
                    f"unknown action {raw_kind!r}; expected 'add' or 'delete'"
                ) from None
            announced = row.get(announced_column, "")
            changes.append(IndexChange(
                instrument_id=row[instrument_column].strip(),
                kind=kind,
                effective_ns=_to_ns(row[effective_column]),
                announced_ns=_to_ns(announced) if announced.strip() else None,
            ))
    return MembershipHistory(changes, name=name)


def _to_ns(raw: str) -> int:
    text = raw.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return iso_to_ns(text)
