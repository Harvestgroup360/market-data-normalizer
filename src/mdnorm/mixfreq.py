"""Joining a slow series onto a fast one without reading it early.

A daily number and a one-minute grid meet constantly in research: a daily
close, a settlement price, an overnight risk figure, a vendor factor. The
join itself is the ordinary as-of problem :mod:`mdnorm.align` already solves.
What makes the mixed-frequency case its own module is that the slow series
usually arrives labelled with the period it *describes* rather than the moment
it became *knowable*, and those differ by a whole session::

    from mdnorm import Period, PeriodSeries, leak_report

    series = PeriodSeries.from_sessions(daily_closes, US_EQUITY_RTH)
    feature = series.knowable_series()      # safe to join
    report = leak_report(series, grid)      # what the naive join would cost

**A daily bar labelled Tuesday is not knowable on Tuesday morning.** It is
knowable once Tuesday's session has closed, which is Tuesday evening — and
often later still, because the number has to be published. Join it by its
label and every minute of Tuesday sees a value that summarises, among other
things, the rest of Tuesday. The series is real, the dates are right, and the
feature quietly contains its own future.

**The size of that error is not the same as its danger.** Reading a close
seven hours early is worth very little on a slow signal and everything on a
fast one, and no rule of thumb decides which you have. So this module does not
argue: :func:`leak_report` counts the grid points where the naive join would
have shown a value that did not yet exist, and how far ahead the worst of them
was read. On a weekly rebalance the count will be large and the consequence
small. On a five-minute signal the same count is the whole result.

**Publication lag is a separate claim from the session close.** A settlement
price exists when the session closes; it reaches you when the vendor sends it.
``publication_lag_ns`` is where that goes, and it defaults to zero because a
lag of zero is a statement about your data feed that only you can make. A
default of "fifteen minutes, everyone uses that" would be a plausible constant
that rescales an entire study, which this library declines to supply here for
the same reason it declines to supply an annualisation factor.

The output of this module is an ordinary :class:`~mdnorm.align.AsOfSeries`
keyed by knowability, so it drops straight into :func:`mdnorm.align.align`
beside every fast stream. Nothing new joins anything; the key is simply honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .align import AsOfSeries, BarField
from .bars import Bar
from .fileio import open_text
from .sessions import Session, session_bounds, session_date
from .timeutil import iso_to_ns

__all__ = [
    "Period",
    "PeriodSeries",
    "LeakReport",
    "leak_report",
    "read_periods_csv",
]


@dataclass(frozen=True, slots=True)
class Period:
    """One low-frequency observation and the span it describes.

    ``start_ns`` and ``end_ns`` are half-open: ``end_ns`` is the first instant
    the period is over, and therefore the earliest instant the value could be
    computed from complete data. The label a vendor ships — usually the date
    the period opened — is ``start_ns``, and it is the wrong key for a join.
    """

    start_ns: int
    end_ns: int
    value: Decimal

    def __post_init__(self) -> None:
        if self.start_ns < 0:
            raise ValueError("period timestamps must be non-negative")
        if self.end_ns <= self.start_ns:
            raise ValueError(
                "end_ns must follow start_ns; a period that ends when it "
                "starts describes nothing"
            )


@dataclass(frozen=True, slots=True)
class LeakReport:
    """What joining on the label would have shown that was not yet knowable.

    ``leaking_points`` is the count of grid points where the naive join
    produces a value whose period had not finished (or had finished but not
    yet been published). ``max_lead_ns`` is the furthest any grid point ran
    ahead of the moment its value became knowable.
    """

    grid_points: int
    knowable_points: int
    label_points: int
    leaking_points: int
    max_lead_ns: Optional[int]

    @property
    def leaking_fraction(self) -> Optional[Decimal]:
        """Share of grid points the naive join would have got wrong."""
        if self.grid_points == 0:
            return None
        return Decimal(self.leaking_points) / self.grid_points


class PeriodSeries:
    """A slow series that knows when each of its values became readable."""

    __slots__ = ("name", "publication_lag_ns", "_periods")

    def __init__(
        self,
        periods: Iterable[Period],
        *,
        publication_lag_ns: int = 0,
        name: str = "",
    ) -> None:
        if publication_lag_ns < 0:
            raise ValueError("publication_lag_ns must be non-negative")
        self.name = name
        self.publication_lag_ns = publication_lag_ns
        # Two periods with the same span: the later input wins, matching the
        # duplicate-timestamp rule used everywhere else in the library.
        collapsed: Dict[Tuple[int, int], Decimal] = {}
        for p in periods:
            collapsed[(p.start_ns, p.end_ns)] = p.value
        self._periods: List[Period] = [
            Period(start, end, value)
            for (start, end), value in sorted(collapsed.items())
        ]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_sessions(
        cls,
        values: Mapping[date, Decimal],
        session: Session,
        *,
        publication_lag_ns: int = 0,
        name: str = "",
    ) -> "PeriodSeries":
        """Build from ``{trading_date: value}`` and the session it describes.

        The session supplies the close, so the caller does not have to know
        that a New York close is 21:00 UTC in January and 20:00 in July.
        """
        periods = []
        for day, value in values.items():
            start_ns, end_ns = session_bounds(day, session)
            periods.append(Period(start_ns, end_ns, value))
        return cls(periods, publication_lag_ns=publication_lag_ns, name=name)

    @classmethod
    def from_daily_bars(
        cls,
        bars: Iterable[Bar],
        session: Session,
        *,
        field: BarField = BarField.CLOSE,
        publication_lag_ns: int = 0,
        name: str = "",
    ) -> "PeriodSeries":
        """Build from daily bars, taking each bar's session close as its end.

        The bar's own ``end_ns`` is not used: a daily bar is frequently
        stamped midnight-to-midnight regardless of when the market was open,
        and that label would place the close hours after it existed on one
        side and hours before on the other. The session decides.
        """
        periods = []
        for b in bars:
            value = getattr(b, field.value)
            if value is None:
                continue
            day = session_date(b.start_ns, session)
            start_ns, end_ns = session_bounds(day, session)
            periods.append(Period(start_ns, end_ns, value))
        return cls(periods, publication_lag_ns=publication_lag_ns, name=name)

    # -- the two keys ------------------------------------------------------

    def knowable_series(self, *, name: str = "") -> AsOfSeries:
        """Keyed by when each value could first be read. Safe to join.

        The key is the period's end plus ``publication_lag_ns``. Passing this
        to :func:`mdnorm.align.align` alongside intraday streams gives, at
        every grid point, the newest slow value that actually existed then.
        """
        return AsOfSeries(
            ((p.end_ns + self.publication_lag_ns, p.value) for p in self._periods),
            name=name or self.name,
        )

    def labelled_series(self, *, name: str = "") -> AsOfSeries:
        """Keyed by the period's label. **This is the join that leaks.**

        Provided so the error can be measured rather than only described, and
        so a historical study built this way can be reproduced exactly before
        it is corrected. It is not a fallback for when the session is unknown.
        """
        return AsOfSeries(
            ((p.start_ns, p.value) for p in self._periods),
            name=name or self.name,
        )

    # -- querying ----------------------------------------------------------

    def at(
        self, ts_ns: int, *, max_age_ns: Optional[int] = None
    ) -> Tuple[Optional[Decimal], Optional[int]]:
        """The newest value knowable at ``ts_ns``, and its age.

        Same contract as :meth:`mdnorm.align.AsOfSeries.at`: ``(None, None)``
        means nothing had been published yet, ``(None, age)`` means the newest
        value was older than ``max_age_ns``.
        """
        return self.knowable_series().at(ts_ns, max_age_ns=max_age_ns)

    def knowable_at(self, period: Period) -> int:
        """The instant ``period``'s value could first be read."""
        return period.end_ns + self.publication_lag_ns

    @property
    def periods(self) -> Tuple[Period, ...]:
        """Every period, in time order."""
        return tuple(self._periods)

    def __len__(self) -> int:
        return len(self._periods)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"PeriodSeries(name={self.name!r}, n={len(self._periods)}, "
                f"lag_ns={self.publication_lag_ns})")


def leak_report(series: PeriodSeries, grid: Sequence[int]) -> LeakReport:
    """Count the grid points where the label join reads a value too early.

    Both joins are run over the same grid: one keyed by knowability, one by
    label. A point leaks when the label join returns a value and that value
    was not yet knowable at the point — either because its period had not
    closed, or because it had not been published.

    The count on its own does not say whether a study is ruined. A daily
    factor read seven hours early moves a monthly rebalance very little and a
    five-minute signal a great deal. What the report removes is the option of
    not knowing.
    """
    knowable = series.knowable_series()
    lag = series.publication_lag_ns
    # Which period each labelled value belongs to, so a leak can be measured
    # against the moment that value became readable rather than against the
    # newest value in the series.
    by_label = {p.start_ns: p for p in series.periods}
    label_ts = sorted(by_label)

    from bisect import bisect_right

    leaking = 0
    knowable_points = 0
    label_points = 0
    max_lead: Optional[int] = None
    for t in grid:
        if knowable.at(t)[0] is not None:
            knowable_points += 1
        i = bisect_right(label_ts, t)
        if i == 0:
            continue
        label_points += 1
        period = by_label[label_ts[i - 1]]
        readable_at = period.end_ns + lag
        if t < readable_at:
            leaking += 1
            lead = readable_at - t
            if max_lead is None or lead > max_lead:
                max_lead = lead
    return LeakReport(
        grid_points=len(grid),
        knowable_points=knowable_points,
        label_points=label_points,
        leaking_points=leaking,
        max_lead_ns=max_lead,
    )


def read_periods_csv(
    path: str,
    *,
    start_column: str = "start",
    end_column: str = "end",
    value_column: str = "value",
    publication_lag_ns: int = 0,
    name: str = "",
) -> PeriodSeries:
    """Read explicit period spans from CSV.

    Both boundary columns accept ISO-8601 timestamps or integer nanoseconds.
    A file that carries only a label and no end is not readable here on
    purpose: the end is the entire point, and inventing one from the label
    would reintroduce the error this module exists to remove. Build those
    with :meth:`PeriodSeries.from_sessions`, which derives the end from a
    session you stated.
    """
    import csv

    periods: List[Period] = []
    with open_text(path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            periods.append(
                Period(
                    _to_ns(row[start_column]),
                    _to_ns(row[end_column]),
                    Decimal(row[value_column].strip()),
                )
            )
    return PeriodSeries(
        periods, publication_lag_ns=publication_lag_ns, name=name
    )


def _to_ns(raw: str) -> int:
    text = raw.strip()
    if text.lstrip("-").isdigit():
        return int(text)
    return iso_to_ns(text)
