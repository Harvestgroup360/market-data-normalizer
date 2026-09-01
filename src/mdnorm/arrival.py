"""When the venue says it happened, and when you found out.

Every observation in a live system has at least two timestamps: the one the
venue put on it, and the one your process read it at. Research keys on the
first, which quietly assumes the second is the same::

    from mdnorm import Arrival, delay_report, as_received, view_gap

    report = delay_report(arrivals)     # what your transport actually costs
    knowable = as_received(arrivals)    # the series as you really had it
    view_gap(arrivals, grid)            # what keying on the venue stamp buys

:meth:`~mdnorm.align.AsOfSeries.delayed` has always let you shift a series by
a delay you supply, with a docstring saying a delay of zero is a claim about
your infrastructure rather than a default. This module is the missing half:
measuring the delay you actually have, from data that carries both stamps.

**A venue timestamp is not an arrival.** Keying research on it is a claim that
information reached you instantly, and the error is one-directional: every
signal looks actionable slightly earlier than it was, every cross-venue lead
is inflated by the difference in transport, and a fill is priced at a quote
that had not reached the machine placing the order. None of it fails; the
result is simply better than it should be.

**There is no default delay.** If a file carries no receipt column, this
module cannot invent one. State it, and the report will record that the figure
was assumed rather than observed — the two are not interchangeable and a
report that hides which one it used is worse than no report.

**A negative delay is a fact, not an outlier.** Receipt before the venue stamp
means the two clocks disagree, which is a real property of the setup and is
usually the more interesting finding. It is counted and reported separately
and never clamped to zero, because clamping turns a clock problem into a
latency figure that looks fine.

**The mean latency is the least useful summary there is.** A transport
distribution has a tail, and the mean mostly measures it. The report gives the
median and the 95th percentile by nearest rank over the observed values — no
interpolation, so every figure it prints is a delay that actually happened.

**Out-of-order arrivals are counted rather than sorted away.** Messages
overtake each other, and a pipeline that sorts on receipt without saying how
often it had to is hiding the evidence that its sequencing matters.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence

from .align import AsOfSeries

__all__ = [
    "Arrival",
    "DelayReport",
    "ViewGap",
    "delay_report",
    "as_received",
    "as_stamped",
    "view_gap",
    "read_arrivals_csv",
]


@dataclass(frozen=True, slots=True)
class Arrival:
    """One observation, with both of the times that describe it.

    ``received_ns`` is deliberately allowed to precede ``venue_ns``. That
    combination means the clocks disagree, and refusing to represent it would
    only mean the disagreement never gets measured.
    """

    venue_ns: int
    received_ns: int
    value: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.venue_ns < 0 or self.received_ns < 0:
            raise ValueError("timestamps must be non-negative")

    @property
    def delay_ns(self) -> int:
        """Receipt minus venue. Negative when the clocks disagree."""
        return self.received_ns - self.venue_ns


@dataclass(frozen=True, slots=True)
class DelayReport:
    """What the transport between the venue and this process costs."""

    observations: int
    negative: int
    out_of_order: int
    min_ns: Optional[int]
    median_ns: Optional[int]
    p95_ns: Optional[int]
    max_ns: Optional[int]
    assumed: bool = False

    @property
    def clock_skew_share(self) -> Optional[Decimal]:
        """Share of observations that arrived before they happened.

        Anything above zero is worth chasing before the latency figures are
        used for anything, because a clock that runs backwards on some
        messages is not measuring the same thing on the rest.
        """
        if self.observations == 0:
            return None
        return Decimal(self.negative) / self.observations

    @property
    def tail_ratio(self) -> Optional[Decimal]:
        """p95 over the median: how heavy the tail is.

        A ratio near one is a well-behaved link. A large one says the typical
        case and the bad case are different problems, and that sizing a
        timeout or a staleness window off the median will fire constantly.
        """
        if self.median_ns is None or self.p95_ns is None or self.median_ns <= 0:
            return None
        return Decimal(self.p95_ns) / self.median_ns


@dataclass(frozen=True, slots=True)
class ViewGap:
    """How much a venue-stamped view claims over a receipt-stamped one."""

    grid_points: int
    differ: int
    earliest_gain_ns: Optional[int]
    largest_gain_ns: Optional[int]

    @property
    def share(self) -> Optional[Decimal]:
        """Fraction of grid points where the two views disagree."""
        if self.grid_points == 0:
            return None
        return Decimal(self.differ) / self.grid_points


def _nearest_rank(sorted_values: Sequence[int], percentile: int) -> int:
    """The value at ``percentile`` by nearest rank, never interpolated.

    Interpolating between two observed latencies produces a number that never
    happened, which is a poor thing to put in a report about what did.
    """
    n = len(sorted_values)
    idx = max(0, min(n - 1, (percentile * n + 99) // 100 - 1))
    return sorted_values[idx]


def delay_report(
    arrivals: Iterable[Arrival],
    *,
    assume_delay_ns: Optional[int] = None,
) -> DelayReport:
    """Measure the delay between the venue stamp and the receipt stamp.

    ``assume_delay_ns`` is for the case where the data carries no receipt
    column at all. Passing it produces a report describing that assumption,
    with ``assumed`` set — it is not a fallback that quietly fills in for
    missing evidence, it is a way of writing down what you decided to believe.
    """
    items = list(arrivals)
    if assume_delay_ns is not None:
        if assume_delay_ns < 0:
            raise ValueError("an assumed delay must be non-negative")
        return DelayReport(
            observations=len(items), negative=0, out_of_order=0,
            min_ns=assume_delay_ns, median_ns=assume_delay_ns,
            p95_ns=assume_delay_ns, max_ns=assume_delay_ns, assumed=True)

    if not items:
        return DelayReport(0, 0, 0, None, None, None, None)

    delays = sorted(a.delay_ns for a in items)
    negative = sum(1 for d in delays if d < 0)

    by_venue = sorted(items, key=lambda a: a.venue_ns)
    disordered = sum(1 for prev, cur in zip(by_venue, by_venue[1:])
                     if cur.received_ns < prev.received_ns)

    return DelayReport(
        observations=len(delays),
        negative=negative,
        out_of_order=disordered,
        min_ns=delays[0],
        median_ns=_nearest_rank(delays, 50),
        p95_ns=_nearest_rank(delays, 95),
        max_ns=delays[-1],
    )


def _valued(arrivals: Iterable[Arrival]) -> List[Arrival]:
    out = [a for a in arrivals if a.value is not None]
    if not out:
        raise ValueError(
            "these arrivals carry no values, so there is no series to build; "
            "delay_report works on timestamps alone if that is all you have")
    return out


def as_received(arrivals: Iterable[Arrival], *, name: str = "") -> AsOfSeries:
    """The series keyed at the moment each value reached this process.

    This is the one a strategy could have acted on. Where two values arrive at
    the same instant the later one in the input wins, which is
    :class:`~mdnorm.align.AsOfSeries` behaving as it always does: a correction
    arriving with the same stamp supersedes what it corrects.
    """
    return AsOfSeries(((a.received_ns, a.value) for a in _valued(arrivals)
                       if a.value is not None), name=name)


def as_stamped(arrivals: Iterable[Arrival], *, name: str = "") -> AsOfSeries:
    """The series keyed at the venue timestamp — the optimistic view.

    Kept deliberately, and not because it is wrong to want it: it is the right
    series for asking what the market did. It is the wrong one for asking what
    you could have done, and having both is what lets the difference between
    those two questions be measured instead of argued about.
    """
    return AsOfSeries(((a.venue_ns, a.value) for a in _valued(arrivals)
                       if a.value is not None), name=name)


def view_gap(arrivals: Iterable[Arrival],
             grid: Sequence[int]) -> ViewGap:
    """Compare the two views on a time grid, and say what the first one buys.

    At every grid point, the venue-stamped series is asked what it knew and
    the receipt-stamped series is asked the same. Where they differ, the
    optimistic view is holding a value that had not arrived yet.

    ``largest_gain_ns`` is the largest amount of foresight found at any grid
    point: how much sooner the optimistic view showed a value than the process
    could have had it. It is the number to compare against the horizon a
    signal acts on — a quarter of a second of unearned foresight is nothing to
    a daily rebalance and everything to a queue position.

    Where the two views happen to hold the same value they are not counted as
    differing, even if they are holding it for different reasons. The question
    here is what the view showed, not which row it came from.
    """
    items = _valued(arrivals)
    stamped, received = as_stamped(items), as_received(items)

    by_venue = sorted(items, key=lambda a: a.venue_ns)
    venue_ts = [a.venue_ns for a in by_venue]

    differ = 0
    earliest: Optional[int] = None
    largest: Optional[int] = None

    for t in grid:
        optimistic, _ = stamped.at(t)
        knowable, _ = received.at(t)
        if optimistic == knowable:
            continue
        differ += 1
        if earliest is None or t < earliest:
            earliest = t
        i = bisect_right(venue_ts, t)
        if i:
            # The value on show at t is still in flight; this is how far.
            lead = by_venue[i - 1].received_ns - t
            if lead > 0 and (largest is None or lead > largest):
                largest = lead

    return ViewGap(grid_points=len(grid), differ=differ,
                   earliest_gain_ns=earliest, largest_gain_ns=largest)


def read_arrivals_csv(
    path: str,
    *,
    venue_column: str = "venue_ns",
    received_column: str = "received_ns",
    value_column: Optional[str] = "value",
) -> List[Arrival]:
    """Read ``venue_ns,received_ns[,value]`` rows.

    ``value_column`` may be ``None`` for a file of timestamps alone, which is
    all :func:`delay_report` needs. A row missing either timestamp is an error
    rather than a skip: this module exists to compare two clocks, and a row
    that has only one of them cannot take part in that.
    """
    import csv

    from .fileio import open_text

    out: List[Arrival] = []
    with open_text(path) as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                venue = int(row[venue_column])
                received = int(row[received_column])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"line {i}: both {venue_column} and {received_column} are "
                    "required; a row with one clock cannot compare two")
            value = None
            if value_column is not None:
                raw = (row.get(value_column) or "").strip()
                if raw:
                    value = Decimal(raw)
            out.append(Arrival(venue, received, value))
    if not out:
        raise ValueError("no arrivals in file")
    return out
