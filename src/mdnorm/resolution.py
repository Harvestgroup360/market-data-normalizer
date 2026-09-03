"""Stored in nanoseconds is not the same as measured in nanoseconds.

Every timestamp in this library is an integer nanosecond. That is a storage
decision made once, and it says nothing about the feed. A vendor that stamps
to the millisecond and hands you nanoseconds has multiplied by a million; the
extra six digits are zeros, and every one of them is a claim nobody made::

    from mdnorm import detect_resolution, classification_risk

    res = detect_resolution(ts)         # granularity 1 ms, not 1 ns
    res.tied_share                      # 31% of events share a timestamp
    classification_risk(events)         # what that costs the side inference

**Resolution is detectable, and detection is divisibility.** A feed stamped to
the millisecond leaves every timestamp a multiple of a million nanoseconds.
This module walks the decimal ladder — nanosecond, ten, hundred, microsecond,
and so on up to a second — and reports the coarsest unit that divides
everything it was given. Only decimal units are considered, because those are
the units a clock is actually read in; a divisor of 2,000,000 would be an
arithmetic fact rather than a statement about the venue.

**Divisibility needs enough observations to mean anything.** If a feed really
were nanosecond-resolution, the chance that twenty independent timestamps are
all multiples of ten is one in 10^20, so twenty is plenty — but three is not,
and a module that answers confidently from three has told you about its own
arithmetic. Below ``min_observations`` the answer is that the resolution is
undetermined, which is a different thing from one nanosecond.

**A tie is not an ordering.** Events sharing a timestamp happened in some
order and the file records one of them. That order is the order the writer
happened to use — a sort that was not stable, a queue that interleaved, a
batch flushed in whatever sequence the buffer held. Reading it as sequence
information is reading the writer rather than the market.

**Where it costs something is trade classification.** The quote rule matches
each trade against the quote in force, and an as-of join takes the newest
quote at or before the trade. When that quote carries the same timestamp as
the trade, whether it was actually in force first is not in the data.
:func:`classification_risk` counts those trades, and then re-runs the rule
against the last quote that is provably earlier, so you get both the exposure
and the number of classifications that actually change.

Nothing here re-sorts, jitters, or fills in a finer timestamp. The resolution
you have is the resolution you have; this module only declines to let it be
mistaken for a better one.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .micro import quote_rule
from .schema import EventType, MarketEvent

__all__ = [
    "Resolution",
    "ClassificationRisk",
    "LADDER",
    "detect_resolution",
    "tie_groups",
    "order_is_determined",
    "classification_risk",
    "read_timestamps_csv",
]

#: The units a clock is read in, coarsest last. A feed stamped in one of
#: these leaves every timestamp a multiple of it.
LADDER: Tuple[int, ...] = (
    1,                      # nanosecond
    10,
    100,                    # the Windows FILETIME tick
    1_000,                  # microsecond
    10_000,
    100_000,
    1_000_000,              # millisecond
    10_000_000,
    100_000_000,
    1_000_000_000,          # second
)


@dataclass(frozen=True, slots=True)
class Resolution:
    """What a feed's timestamps can actually distinguish.

    ``granularity_ns`` is ``None`` when there was not enough evidence to say,
    which is deliberately not the same answer as one nanosecond.
    """

    observations: int
    distinct: int
    granularity_ns: Optional[int]
    ties: int
    tied_observations: int
    largest_tie: int

    @property
    def undetermined(self) -> bool:
        """True when the sample was too small to support any answer."""
        return self.granularity_ns is None

    @property
    def tied_share(self) -> Optional[Decimal]:
        """Share of observations that share a timestamp with another.

        This is the part of the file whose order is a property of the writer
        rather than of the market.
        """
        if self.observations == 0:
            return None
        return Decimal(self.tied_observations) / self.observations

    @property
    def overstated_digits(self) -> Optional[int]:
        """How many of the nine sub-second digits are padding.

        Six here means the feed stamps to the millisecond and the file is
        carrying six zeros that look like precision.
        """
        if self.granularity_ns is None:
            return None
        return len(str(self.granularity_ns)) - 1


@dataclass(frozen=True, slots=True)
class ClassificationRisk:
    """What an undetermined ordering costs the side inference.

    ``same_tick`` is the exposure: trades whose matched quote carries a
    timestamp the data cannot place before or after them. ``changed`` is the
    realised part: of those, how many are classified differently once the
    rule is restricted to a quote that is provably earlier.
    """

    trades: int
    classified: int
    same_tick: int
    changed: int
    granularity_ns: Optional[int]

    @property
    def exposed_share(self) -> Optional[Decimal]:
        """Share of trades resting on a same-tick quote."""
        if self.trades == 0:
            return None
        return Decimal(self.same_tick) / self.trades

    @property
    def changed_share(self) -> Optional[Decimal]:
        """Share of trades whose side actually moves. Never above one.

        A small number here is not reassurance. The trades it covers are the
        ones nearest the mid, which are also the ones an effective-spread or
        order-flow-imbalance figure is most sensitive to.
        """
        if self.trades == 0:
            return None
        return Decimal(self.changed) / self.trades


def detect_resolution(
    timestamps: Iterable[int],
    *,
    min_observations: int = 20,
) -> Resolution:
    """Find the coarsest decimal unit that divides every timestamp given.

    ``min_observations`` counts distinct timestamps, not rows: a thousand
    copies of one value is one observation of the clock, and treating it as a
    thousand would let a single round number decide the answer.
    """
    if min_observations < 1:
        raise ValueError("min_observations must be at least 1")

    counts: Dict[int, int] = {}
    total = 0
    for ts in timestamps:
        if ts < 0:
            raise ValueError("timestamps must be non-negative")
        counts[ts] = counts.get(ts, 0) + 1
        total += 1

    tie_sizes = [n for n in counts.values() if n > 1]
    ties = len(tie_sizes)
    tied = sum(tie_sizes)
    largest = max(tie_sizes) if tie_sizes else (1 if counts else 0)

    distinct = len(counts)
    # A timestamp of zero divides by everything and is evidence of nothing.
    informative = [ts for ts in counts if ts > 0]
    granularity: Optional[int] = None
    if len(informative) >= min_observations:
        granularity = 1
        for unit in LADDER:
            if all(ts % unit == 0 for ts in informative):
                granularity = unit

    return Resolution(observations=total, distinct=distinct,
                      granularity_ns=granularity, ties=ties,
                      tied_observations=tied, largest_tie=largest)


def tie_groups(timestamps: Iterable[int]) -> Iterator[Tuple[int, int]]:
    """Yield ``(timestamp, count)`` for every value that appears more than once.

    Ordered by timestamp. The counts are what an ordering-sensitive step —
    the tick rule, a sequence check, a lead-lag at this scale — is being
    asked to take on faith.
    """
    counts: Dict[int, int] = {}
    for ts in timestamps:
        counts[ts] = counts.get(ts, 0) + 1
    for ts in sorted(counts):
        if counts[ts] > 1:
            yield ts, counts[ts]


def order_is_determined(a_ns: int, b_ns: int, granularity_ns: int) -> bool:
    """Whether the data can say which of two events came first.

    False when both fall in the same tick of the clock that stamped them —
    including when the two timestamps differ, if that difference is finer
    than the feed can actually resolve.
    """
    if granularity_ns <= 0:
        raise ValueError("granularity_ns must be positive")
    return a_ns // granularity_ns != b_ns // granularity_ns


def _is_trade(e: MarketEvent) -> bool:
    return e.event_type is EventType.TRADE and e.price is not None


def _is_quote(e: MarketEvent) -> bool:
    return (e.event_type is EventType.QUOTE
            and e.bid_price is not None and e.ask_price is not None)


def classification_risk(
    events: Iterable[MarketEvent],
    *,
    granularity_ns: Optional[int] = None,
    min_observations: int = 20,
) -> ClassificationRisk:
    """Measure how much of the quote-rule side inference rests on a tie.

    For every trade, the quote an as-of join would use is compared with the
    last quote that is provably in an earlier tick. ``same_tick`` counts the
    trades where those differ — the exposure — and ``changed`` counts the
    subset whose classification is not the same under both.

    ``granularity_ns`` is detected from the events when not supplied. If it
    cannot be determined the report comes back with the counts at zero and
    ``granularity_ns`` as ``None``, rather than assuming one nanosecond and
    reporting a reassuring figure it has no basis for.
    """
    items = list(events)
    if granularity_ns is None:
        detected = detect_resolution((e.ts_ns for e in items),
                                     min_observations=min_observations)
        granularity_ns = detected.granularity_ns
    elif granularity_ns <= 0:
        raise ValueError("granularity_ns must be positive")

    trades = [e for e in items if _is_trade(e)]
    if granularity_ns is None:
        return ClassificationRisk(trades=len(trades), classified=0,
                                  same_tick=0, changed=0, granularity_ns=None)

    quotes = sorted((e for e in items if _is_quote(e)), key=lambda e: e.ts_ns)
    quote_ts = [e.ts_ns for e in quotes]

    def quote_at(ts: int) -> Optional[MarketEvent]:
        i = bisect_right(quote_ts, ts)
        return quotes[i - 1] if i else None

    classified = same_tick = changed = 0
    for t in trades:
        price: Decimal = t.price  # type: ignore[assignment]
        used = quote_at(t.ts_ns)
        # The last quote that cannot have been printed in the trade's own tick.
        tick_start = (t.ts_ns // granularity_ns) * granularity_ns
        earlier = quote_at(tick_start - 1) if tick_start else None

        side_used = (quote_rule(price, used.bid_price, used.ask_price)
                     if used is not None else None)
        if side_used is not None:
            classified += 1
        if used is earlier:
            continue
        same_tick += 1
        side_earlier = (quote_rule(price, earlier.bid_price, earlier.ask_price)
                        if earlier is not None else None)
        if side_used != side_earlier:
            changed += 1

    return ClassificationRisk(trades=len(trades), classified=classified,
                              same_tick=same_tick, changed=changed,
                              granularity_ns=granularity_ns)


def read_timestamps_csv(path: str, *, ts_column: str = "ts_ns") -> List[int]:
    """Read one column of integer nanosecond timestamps.

    A row whose timestamp does not parse is an error rather than a skip: the
    whole question here is what the set of timestamps is divisible by, and a
    reader that quietly drops the awkward ones answers a different question.
    """
    import csv

    from .fileio import open_text

    out: List[int] = []
    with open_text(path) as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                out.append(int(row[ts_column]))
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"line {i}: {ts_column} is required and must be an "
                    "integer number of nanoseconds")
    if not out:
        raise ValueError("no timestamps in file")
    return out
