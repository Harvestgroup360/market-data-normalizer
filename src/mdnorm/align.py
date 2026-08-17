"""Aligning several instruments onto one time grid without leaking the future.

Research wants a matrix: one row per timestamp, one column per instrument.
Getting there from independent tick streams is an as-of join, and it is where
look-ahead bias enters a pipeline more often than anywhere else, because every
mistake here produces a *better* backtest rather than an error::

    from mdnorm import Field, align

    rows = align({"BTC": btc_events, "ETH": eth_events},
                 interval_ns=60_000_000_000,      # a one-minute grid
                 max_age_ns=5 * 60_000_000_000)   # nothing older than 5 minutes
    for r in rows:
        print(r.ts_ns, r.values["BTC"], r.values["ETH"], r.complete)

**Every value carries the time it was observed, and only values observed at or
before a grid point may appear on that row.** The join here is strictly
backward-looking: :meth:`AsOfSeries.at` uses ``ts <= t``, never the nearest
observation in either direction. "Nearest" is the single most expensive default
in this whole area — on a one-minute grid it lets a value from 09:30:20 be read
at 09:30:00, and twenty seconds of hindsight is enough to make a mediocre
signal look tradeable.

**A bar labelled by its start is not observable at its start.** A one-minute
bar labelled 09:30 contains everything that traded until 09:31, so joining it
to another series at 09:30 imports a minute of the future. :meth:`AsOfSeries.from_bars`
timestamps every bar at its *end* for that reason, which means a bar-derived
column is deliberately one interval behind the grid point that produced it.
This is the correction that most often turns a profitable backtest into a flat
one, and the flat one is the true one.

**Forward-filling has no natural end.** A stream that stops — a halt, a
delisting, a dropped subscription — will otherwise contribute its last known
price forever, and a frozen price is uncorrelated with everything, which reads
as diversification. Pass ``max_age_ns`` and a column that has gone quiet
becomes ``None`` instead of a fossil. The age is still reported, so a stale
column is distinguishable from one that never had data at all.

**A feed you receive late was not available on time.** Where a series reaches
you after a publication or network delay, :meth:`AsOfSeries.delayed` shifts its
observation times forward by that delay, so the alignment reflects when you
could actually have acted on it rather than when it was stamped at the source.

Nothing here interpolates, smooths, or invents a value between observations.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field as _dc_field
from decimal import Decimal
from enum import Enum
from typing import (Dict, Iterable, List, Mapping, Optional, Sequence, Tuple)

from .bars import Bar
from .schema import EventType, MarketEvent

__all__ = [
    "Field",
    "BarField",
    "AsOfSeries",
    "AlignedRow",
    "align",
    "align_on",
    "align_bars",
    "grid",
]

#: Refuse to build a grid larger than this many rows. A grid this size is
#: almost always a wrong interval rather than a real intention.
MAX_GRID_ROWS = 10_000_000


class Field(str, Enum):
    """Which number to take from an event stream."""

    PRICE = "price"   # last trade price
    MID = "mid"       # quote mid, requires both sides
    BID = "bid"
    ASK = "ask"


class BarField(str, Enum):
    """Which number to take from a bar. All are observable only at bar end."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VWAP = "vwap"


def _value_of(e: MarketEvent, f: Field) -> Optional[Decimal]:
    if f is Field.PRICE:
        return e.price if e.event_type is EventType.TRADE else None
    if e.event_type is not EventType.QUOTE:
        return None
    if f is Field.MID:
        return e.mid_price
    if f is Field.BID:
        return e.bid_price
    return e.ask_price


class AsOfSeries:
    """One instrument's observations, queryable as of a point in time.

    Construction sorts by timestamp and collapses repeats: when two
    observations share a timestamp the later one in the input wins, since a
    correction that arrives with the same stamp supersedes what it corrects.
    """

    __slots__ = ("name", "_ts", "_values")

    def __init__(
        self,
        observations: Iterable[Tuple[int, Decimal]],
        *,
        name: str = "",
    ) -> None:
        self.name = name
        collapsed: Dict[int, Decimal] = {}
        for ts, value in observations:
            if ts < 0:
                raise ValueError("observation timestamps must be non-negative")
            collapsed[ts] = value
        self._ts: List[int] = sorted(collapsed)
        self._values: List[Decimal] = [collapsed[t] for t in self._ts]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_events(
        cls,
        events: Iterable[MarketEvent],
        *,
        field: Field = Field.PRICE,
        name: str = "",
    ) -> "AsOfSeries":
        """Take one field from a normalized event stream.

        Events that do not carry the requested field are skipped rather than
        filled: a trade has no mid, and a one-sided quote has no mid either.
        """
        obs = []
        for e in events:
            v = _value_of(e, field)
            if v is not None:
                obs.append((e.ts_ns, v))
        return cls(obs, name=name)

    @classmethod
    def from_bars(
        cls,
        bars: Iterable[Bar],
        *,
        field: BarField = BarField.CLOSE,
        name: str = "",
    ) -> "AsOfSeries":
        """Take one field from a bar series, timestamped at each bar's end.

        A bar summarises an interval, so nothing about it is known until the
        interval is over — including its open, which is only final once no
        earlier trade can still arrive out of order. Using ``end_ns`` rather
        than the label is the whole point of this constructor: it is what
        stops a bar from being read before it closed.
        """
        obs = []
        for b in bars:
            v = getattr(b, field.value)
            if v is not None:
                obs.append((b.end_ns, v))
        return cls(obs, name=name)

    def delayed(self, by_ns: int) -> "AsOfSeries":
        """The same series as it would arrive with a delivery delay.

        Shifts every observation time forward by ``by_ns``, so a value stamped
        at the source at 09:30:00.000 and received 250ms later cannot be read
        before 09:30:00.250. Model the delay you actually have; a delay of
        zero is a claim about your infrastructure, not a default.
        """
        if by_ns < 0:
            raise ValueError("by_ns must be non-negative")
        return AsOfSeries(
            ((ts + by_ns, v) for ts, v in zip(self._ts, self._values)),
            name=self.name,
        )

    # -- querying ----------------------------------------------------------

    def at(
        self, ts_ns: int, *, max_age_ns: Optional[int] = None
    ) -> Tuple[Optional[Decimal], Optional[int]]:
        """The last value observed at or before ``ts_ns``, and its age.

        Returns ``(value, age_ns)``. Two different kinds of missing are
        distinguished on purpose: ``(None, None)`` means nothing had been
        observed yet at that time, while ``(None, age)`` means the newest
        observation was older than ``max_age_ns`` — a stale column, which is
        a data problem worth seeing rather than a gap at the start of history.
        """
        if max_age_ns is not None and max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        i = bisect_right(self._ts, ts_ns)
        if i == 0:
            return None, None
        age = ts_ns - self._ts[i - 1]
        if max_age_ns is not None and age > max_age_ns:
            return None, age
        return self._values[i - 1], age

    def __len__(self) -> int:
        return len(self._ts)

    @property
    def first_ts_ns(self) -> Optional[int]:
        return self._ts[0] if self._ts else None

    @property
    def last_ts_ns(self) -> Optional[int]:
        return self._ts[-1] if self._ts else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"AsOfSeries(name={self.name!r}, n={len(self._ts)}, "
                f"span=({self.first_ts_ns}, {self.last_ts_ns}))")


@dataclass(frozen=True, slots=True)
class AlignedRow:
    """One grid point: a value per column, plus how old each value was."""

    ts_ns: int
    values: Dict[str, Optional[Decimal]] = _dc_field(default_factory=dict)
    ages_ns: Dict[str, Optional[int]] = _dc_field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True when every column has a value."""
        return bool(self.values) and all(v is not None for v in self.values.values())

    @property
    def stale(self) -> Tuple[str, ...]:
        """Columns dropped for being older than the staleness window."""
        return tuple(
            k for k, v in self.values.items()
            if v is None and self.ages_ns.get(k) is not None
        )

    @property
    def missing(self) -> Tuple[str, ...]:
        """Columns with no observation at all at this point in time."""
        return tuple(
            k for k, v in self.values.items()
            if v is None and self.ages_ns.get(k) is None
        )


def grid(start_ns: int, end_ns: int, interval_ns: int) -> List[int]:
    """Grid points in ``[start_ns, end_ns)``, spaced ``interval_ns`` apart."""
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")
    if end_ns <= start_ns:
        raise ValueError("end_ns must be greater than start_ns")
    count = (end_ns - start_ns + interval_ns - 1) // interval_ns
    if count > MAX_GRID_ROWS:
        raise ValueError(
            f"the requested grid has {count} rows (limit {MAX_GRID_ROWS}); "
            "this is nearly always too small an interval rather than an "
            "intention — widen interval_ns or narrow the window"
        )
    return [start_ns + i * interval_ns for i in range(count)]


def _series_map(
    streams: Mapping[str, object], field: Field
) -> Dict[str, AsOfSeries]:
    out: Dict[str, AsOfSeries] = {}
    for name, stream in streams.items():
        if isinstance(stream, AsOfSeries):
            out[name] = stream
        else:
            out[name] = AsOfSeries.from_events(
                stream, field=field, name=name  # type: ignore[arg-type]
            )
    return out


def align_on(
    timestamps: Sequence[int],
    streams: Mapping[str, object],
    *,
    field: Field = Field.PRICE,
    max_age_ns: Optional[int] = None,
    require_all: bool = False,
) -> List[AlignedRow]:
    """Align streams as of timestamps you supply.

    Use this when the grid is not regular — one row per event of a reference
    instrument, per signal time, per fill. ``streams`` values may be event
    iterables or ready-made :class:`AsOfSeries`, which is how a delayed or
    bar-derived column joins the same matrix as a raw tick column.

    With ``require_all`` set, rows where any column is missing are dropped.
    That is usually what a feature matrix wants; leave it off and inspect
    :attr:`AlignedRow.stale` and :attr:`AlignedRow.missing` first, because how
    many rows a join throws away is itself a finding about the data.
    """
    series = _series_map(streams, field)
    rows: List[AlignedRow] = []
    for t in sorted(timestamps):
        values: Dict[str, Optional[Decimal]] = {}
        ages: Dict[str, Optional[int]] = {}
        for name, s in series.items():
            v, age = s.at(t, max_age_ns=max_age_ns)
            values[name] = v
            ages[name] = age
        row = AlignedRow(ts_ns=t, values=values, ages_ns=ages)
        if require_all and not row.complete:
            continue
        rows.append(row)
    return rows


def align(
    streams: Mapping[str, object],
    *,
    interval_ns: int,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    field: Field = Field.PRICE,
    max_age_ns: Optional[int] = None,
    require_all: bool = False,
) -> List[AlignedRow]:
    """Align streams onto a regular grid of ``interval_ns``.

    The default window starts at the first grid point at which any stream has
    data — rounded *up*, since a grid point before the first observation can
    only be empty — and ends at the first grid point at or after the last
    observation, so nothing observed is left out of every row. Pass
    ``start_ns`` and ``end_ns`` to pin the window instead, which is what you
    want when several runs have to line up row for row; pinning both and
    inverting them is an error, while a derived bound that closes the window
    early simply yields no rows.

    Returns an empty list when no stream carries the requested field.
    """
    series = _series_map(streams, field)
    firsts = [s.first_ts_ns for s in series.values() if s.first_ts_ns is not None]
    lasts = [s.last_ts_ns for s in series.values() if s.last_ts_ns is not None]
    if not firsts:
        return []
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")

    def _ceil(v: int) -> int:
        return -((-v) // interval_ns) * interval_ns

    # The default window runs from the first grid point that can hold data to
    # the first one at or after the last observation, so no leading row is
    # empty and no observation goes unrepresented.
    start = _ceil(min(firsts)) if start_ns is None else start_ns
    end = (_ceil(max(lasts)) + 1) if end_ns is None else end_ns
    if end <= start:
        if start_ns is not None and end_ns is not None:
            raise ValueError("end_ns must be greater than start_ns")
        # One bound was derived and the window came out empty; that is an
        # answer about the data, not a mistake by the caller.
        return []

    return align_on(
        grid(start, end, interval_ns),
        series,
        field=field,
        max_age_ns=max_age_ns,
        require_all=require_all,
    )


def align_bars(
    streams: Mapping[str, Sequence[Bar]],
    *,
    interval_ns: int,
    field: BarField = BarField.CLOSE,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
    max_age_ns: Optional[int] = None,
    require_all: bool = False,
) -> List[AlignedRow]:
    """Align bar series onto a grid, honouring the bar-end rule.

    Every bar is read at ``end_ns``, so a grid at the same interval as the
    bars produces a matrix in which each column is the last *closed* bar. That
    is one interval further back than a naive join on the bar label, and it is
    the version you can actually trade.
    """
    series: Dict[str, object] = {
        name: AsOfSeries.from_bars(bars, field=field, name=name)
        for name, bars in streams.items()
    }
    return align(
        series,
        interval_ns=interval_ns,
        start_ns=start_ns,
        end_ns=end_ns,
        max_age_ns=max_age_ns,
        require_all=require_all,
    )
