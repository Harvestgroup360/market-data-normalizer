"""Comparing two sources that claim to describe the same thing.

:mod:`mdnorm.quality` inspects one feed and reports what looks wrong inside
it. A second feed asks a different question, and it is the one desks actually
use to decide whether a vendor can be trusted: where do these two disagree,
and by how much::

    from mdnorm import reconcile, suggest_shift

    report, mismatches = reconcile(primary, secondary,
                                   relative_tolerance=Decimal("0.0001"))
    report.agreement          # share of shared timestamps that matched
    suggest_shift(primary, secondary, max_shift_ns=SECOND)

**The two kinds of disagreement are not the same problem.** A timestamp
present in one feed and absent in the other is a coverage difference — a
dropped message, a filtered print, a venue one side does not carry. A
timestamp present in both with different values is a content difference, and
it means at least one of them is wrong about something you can check. Rolling
the two into a single "match rate" hides which of them you have, so this
module counts them separately and never adds them together.

**There is no default tolerance.** Two feeds of the same instrument will
differ in the last digits for reasons that are not errors, and any constant
that decides how much difference is acceptable is a judgement about your data
rather than a property of it. Called with no tolerance, :func:`reconcile`
requires exact equality — the strictest reading, and one that at least states
its own assumption.

**Zero overlap almost never means total disagreement.** It usually means a
clock offset. If one feed stamps at the venue and the other on receipt, exact
matching finds nothing in common and the naive conclusion is that the feeds
are unrelated, which is both alarming and wrong.
:func:`suggest_shift` looks for the constant offset that lines them up and
reports how much of the series it would explain, so that failure is diagnosed
rather than believed.

Everything here compares :class:`~mdnorm.align.AsOfSeries` values, so a bar
field, a normalized price stream and a vendor CSV all reconcile the same way
once they have been turned into a series.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple, cast

from .align import AsOfSeries, BarField
from .bars import Bar
from .features import _PRECISION

__all__ = [
    "MismatchKind",
    "Mismatch",
    "ReconcileReport",
    "ShiftSuggestion",
    "reconcile",
    "reconcile_bars",
    "suggest_shift",
]


class MismatchKind(str, Enum):
    """Why two sources disagreed at one timestamp."""

    #: The left source has an observation the right one does not.
    ONLY_LEFT = "only_left"
    #: The right source has an observation the left one does not.
    ONLY_RIGHT = "only_right"
    #: Both have one, and the values differ by more than the tolerance.
    VALUE = "value"


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One disagreement, with whatever each side had."""

    ts_ns: int
    kind: MismatchKind
    left: Optional[Decimal] = None
    right: Optional[Decimal] = None

    @property
    def difference(self) -> Optional[Decimal]:
        """``right - left`` when both sides have a value."""
        if self.left is None or self.right is None:
            return None
        return self.right - self.left

    @property
    def relative_difference(self) -> Optional[Decimal]:
        """The difference as a fraction of the left value.

        ``None`` when a side is missing or the left value is zero, because a
        relative difference against zero is not a number and reporting a large
        one would be an invention.
        """
        if self.left is None or self.right is None or self.left == 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return (self.right - self.left) / self.left


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """How far apart two sources are, with the two kinds kept apart."""

    left_points: int
    right_points: int
    common_timestamps: int
    agreed: int
    value_mismatches: int
    only_left: int
    only_right: int
    max_absolute_difference: Optional[Decimal]
    max_relative_difference: Optional[Decimal]
    worst: Optional[Mismatch]

    @property
    def agreement(self) -> Optional[Decimal]:
        """Share of shared timestamps whose values matched.

        Deliberately computed over shared timestamps only. Coverage gaps are
        reported separately because they have a different cause and a
        different fix, and folding them in would produce one number that
        answers neither question.
        """
        if self.common_timestamps == 0:
            return None
        return Decimal(self.agreed) / self.common_timestamps

    @property
    def coverage_difference(self) -> int:
        """Timestamps present in exactly one of the two sources."""
        return self.only_left + self.only_right


@dataclass(frozen=True, slots=True)
class ShiftSuggestion:
    """A constant offset that would line two sources up, and its support."""

    shift_ns: int
    matched: int
    considered: int

    @property
    def explains(self) -> Optional[Decimal]:
        """Share of the compared observations this shift would align."""
        if self.considered == 0:
            return None
        return Decimal(self.matched) / self.considered


def _within(left: Decimal, right: Decimal,
            absolute: Optional[Decimal],
            relative: Optional[Decimal]) -> bool:
    if left == right:
        return True
    if absolute is None and relative is None:
        return False
    diff = abs(right - left)
    if absolute is not None and diff <= absolute:
        return True
    if relative is not None and left != 0:
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            if diff / abs(left) <= relative:
                return True
    return False


def _pairs(series: AsOfSeries) -> Dict[int, Decimal]:
    # Every ts here came out of the series itself, so the lookup cannot miss.
    return {ts: cast(Decimal, series.at(ts)[0])
            for ts in _timestamps(series)}


def _timestamps(series: AsOfSeries) -> List[int]:
    # AsOfSeries keeps its timestamps sorted internally; read them without
    # reaching past the public surface any further than necessary.
    return list(series._ts)  # noqa: SLF001


def reconcile(
    left: AsOfSeries,
    right: AsOfSeries,
    *,
    absolute_tolerance: Optional[Decimal] = None,
    relative_tolerance: Optional[Decimal] = None,
    shift_right_ns: int = 0,
    limit: Optional[int] = None,
) -> Tuple[ReconcileReport, List[Mismatch]]:
    """Compare two sources observation by observation.

    Only timestamps present in both are compared for value; the rest are
    counted as coverage differences and listed with the side that had them.
    With no tolerance given, values must match exactly.

    ``shift_right_ns`` moves the right source forward in time before
    comparing, for the case where one feed stamps at the venue and the other
    on receipt. Use :func:`suggest_shift` to find the offset rather than
    guessing it, and state the one you used — a shift is an assumption about
    two clocks, not a cleanup step.

    ``limit`` caps the number of mismatches returned; the report always counts
    all of them, so a truncated list never quietly becomes a smaller problem.
    """
    if absolute_tolerance is not None and absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    if relative_tolerance is not None and relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    left_map = _pairs(left)
    right_map = {ts + shift_right_ns: v for ts, v in _pairs(right).items()}

    common = sorted(set(left_map) & set(right_map))
    only_left = sorted(set(left_map) - set(right_map))
    only_right = sorted(set(right_map) - set(left_map))

    mismatches: List[Mismatch] = []
    agreed = 0
    value_mismatches = 0
    max_abs: Optional[Decimal] = None
    max_rel: Optional[Decimal] = None
    worst: Optional[Mismatch] = None

    for ts in common:
        a, b = left_map[ts], right_map[ts]
        if _within(a, b, absolute_tolerance, relative_tolerance):
            agreed += 1
            continue
        value_mismatches += 1
        m = Mismatch(ts, MismatchKind.VALUE, a, b)
        diff = abs(b - a)
        if max_abs is None or diff > max_abs:
            max_abs = diff
            worst = m
        rel = m.relative_difference
        if rel is not None:
            rel = abs(rel)
            if max_rel is None or rel > max_rel:
                max_rel = rel
        if limit is None or len(mismatches) < limit:
            mismatches.append(m)

    for ts in only_left:
        if limit is None or len(mismatches) < limit:
            mismatches.append(Mismatch(ts, MismatchKind.ONLY_LEFT,
                                       left_map[ts], None))
    for ts in only_right:
        if limit is None or len(mismatches) < limit:
            mismatches.append(Mismatch(ts, MismatchKind.ONLY_RIGHT,
                                       None, right_map[ts]))

    report = ReconcileReport(
        left_points=len(left_map),
        right_points=len(right_map),
        common_timestamps=len(common),
        agreed=agreed,
        value_mismatches=value_mismatches,
        only_left=len(only_left),
        only_right=len(only_right),
        max_absolute_difference=max_abs,
        max_relative_difference=max_rel,
        worst=worst,
    )
    return report, mismatches


def reconcile_bars(
    left: Iterable[Bar],
    right: Iterable[Bar],
    *,
    field: BarField = BarField.CLOSE,
    absolute_tolerance: Optional[Decimal] = None,
    relative_tolerance: Optional[Decimal] = None,
    limit: Optional[int] = None,
) -> Tuple[ReconcileReport, List[Mismatch]]:
    """Compare one field of two bar series, keyed at each bar's end.

    Bars are keyed by ``end_ns`` for the same reason
    :meth:`~mdnorm.align.AsOfSeries.from_bars` uses it: a bar summarises an
    interval and is not knowable at its label. Two vendors that label bars
    differently would otherwise appear to disagree about everything.
    """
    return reconcile(
        AsOfSeries.from_bars(left, field=field),
        AsOfSeries.from_bars(right, field=field),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        limit=limit,
    )


def suggest_shift(
    left: AsOfSeries,
    right: AsOfSeries,
    *,
    max_shift_ns: int,
    sample: int = 500,
) -> Optional[ShiftSuggestion]:
    """Find a constant offset that would line the two sources up.

    For each of the first ``sample`` observations in the left source, the
    nearest right-hand timestamp within ``max_shift_ns`` is found and the
    difference recorded. The most common difference is returned along with how
    many of the compared observations it accounts for.

    ``None`` means no candidate offset was found at all — either source is
    empty, or nothing in the right source falls within ``max_shift_ns`` of
    anything in the left. A suggestion with a low ``explains`` is not an
    offset; it is scattered noise, and treating it as one would move a series
    on no evidence.

    This does not shift anything. It reports what a shift would buy, because
    a clock difference is a fact about two systems that somebody should
    confirm rather than a value to be fitted.
    """
    if max_shift_ns < 0:
        raise ValueError("max_shift_ns must be non-negative")
    if sample <= 0:
        raise ValueError("sample must be positive")

    from bisect import bisect_left

    left_ts = _timestamps(left)[:sample]
    right_ts = _timestamps(right)
    if not left_ts or not right_ts:
        return None

    diffs: Counter = Counter()
    considered = 0
    for ts in left_ts:
        i = bisect_left(right_ts, ts)
        best: Optional[int] = None
        for j in (i - 1, i):
            if 0 <= j < len(right_ts):
                d = right_ts[j] - ts
                if abs(d) <= max_shift_ns and (best is None or abs(d) < abs(best)):
                    best = d
        if best is not None:
            considered += 1
            diffs[best] += 1
    if not diffs:
        return None
    shift, matched = diffs.most_common(1)[0]
    # reconcile() shifts the right source forward, so the offset it needs is
    # the negative of the difference measured here.
    return ShiftSuggestion(shift_ns=-shift, matched=matched,
                           considered=considered)
