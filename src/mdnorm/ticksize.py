"""Prices live on a grid, and the grid is data.

A venue does not accept any price. It accepts multiples of a tick, the tick
depends on the price band and on the instrument, and both the bands and the
instrument's place in them change over time::

    from mdnorm import TickTable, TickBand, grid_report

    table = TickTable([TickBand(Decimal("0"), Decimal("0.0001")),
                       TickBand(Decimal("1"), Decimal("0.01"))])
    table.tick_at(Decimal("42.30"))          # 0.01
    grid_report(prices, table)               # do these prices sit on it?

**A price off the grid is telling you something.** Raw prints sit on the grid
by construction, because the venue would not have accepted them otherwise. So
a series that does not sit on the grid is not raw: it is a mid, a VWAP, an
average of venues, a back-adjusted history, or an error. Those have very
different consequences and they are indistinguishable by eye, which is why
:func:`grid_report` exists — it is one pass over the data and it answers a
question most pipelines never ask.

**Back-adjustment takes a series off the grid, permanently.** Dividing a
history by a split factor produces prices that were never quotable, and that
is correct — the adjusted series is a returns object, not a price object. It
stops being correct when someone rounds it back onto the grid to make it look
tidy, or feeds it to something that assumes a tradeable price. The grid is the
cheapest way to tell the two apart after the fact.

**There is no default tick size.** Not one hundredth, not anything. The
familiar penny is wrong below a dollar on most venues, wrong for sub-penny
programmes, wrong for crypto by orders of magnitude, and wrong for the same
instrument before the last tick-regime change. A constant here would be a
guess about a venue, applied silently to every price in a study.

**Rounding needs a stated rule, because ties are common here.** On a
continuous scale an exact half is a curiosity. On a tick grid it happens
constantly — a mid between two adjacent ticks is exactly a half-tick, every
time — so the tie rule is not a footnote, it is a systematic bias with a
direction. :class:`Rounding` has no default and no tie shortcut:
:meth:`TickTable.round` must be told what to do.

**Round against yourself or you are inventing edge.** A backtest that rounds
its target price to the nearest tick gets the better side of the grid half the
time for free. :meth:`TickTable.executable` rounds a buy down and a sell up —
away from the fill you wanted — because that is the version that cannot
flatter a result.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from .features import _PRECISION
from .schema import Side

__all__ = [
    "TickBand",
    "TickTable",
    "TickSchedule",
    "Rounding",
    "GridReport",
    "grid_report",
    "spread_in_ticks",
    "read_tick_table_csv",
]


class Rounding(str, Enum):
    """What to do with a price that is not on the grid."""

    #: Towards the next lower tick.
    DOWN = "down"
    #: Towards the next higher tick.
    UP = "up"
    #: To the closer tick; an exact half-tick goes down.
    NEAREST_DOWN = "nearest_down"
    #: To the closer tick; an exact half-tick goes up.
    NEAREST_UP = "nearest_up"


@dataclass(frozen=True, slots=True)
class TickBand:
    """From ``min_price`` upward, prices move in steps of ``tick``."""

    min_price: Decimal
    tick: Decimal

    def __post_init__(self) -> None:
        if self.min_price < 0:
            raise ValueError("min_price must not be negative")
        if self.tick <= 0:
            raise ValueError("tick must be positive")


@dataclass(frozen=True, slots=True)
class GridReport:
    """How well a set of prices matches a tick grid."""

    total: int
    on_grid: int
    off_grid: int
    below_table: int
    worst_offset: Optional[Decimal]
    example: Optional[Decimal]

    @property
    def share_on_grid(self) -> Optional[Decimal]:
        """Fraction of priced observations that sat on the grid."""
        considered = self.on_grid + self.off_grid
        if considered == 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(self.on_grid) / considered

    @property
    def looks_raw(self) -> Optional[bool]:
        """Whether these prices could have been accepted by the venue.

        ``True`` only when every price sat on the grid. Anything else is a
        derived series — a mid, an average, an adjusted history — or a file
        matched against the wrong tick table. The distinction is not made
        here, because it cannot be made from the prices alone.
        """
        if self.on_grid + self.off_grid == 0:
            return None
        return self.off_grid == 0


class TickTable:
    """A price grid: bands of increasing minimum price, each with a tick.

    Bands are half-open and ascending: a band starting at 1 governs prices
    from 1 up to the next band's start. Nothing below the first band can be
    answered, because a table that does not describe the penny stocks in your
    file is not evidence that they trade in pennies.
    """

    __slots__ = ("name", "_mins", "_ticks")

    def __init__(self, bands: Iterable[TickBand], *, name: str = "") -> None:
        ordered = sorted(bands, key=lambda b: b.min_price)
        if not ordered:
            raise ValueError("a tick table needs at least one band")
        mins = [b.min_price for b in ordered]
        if len(set(mins)) != len(mins):
            raise ValueError("two bands start at the same price")
        self.name = name
        self._mins: List[Decimal] = mins
        self._ticks: List[Decimal] = [b.tick for b in ordered]

    @property
    def bands(self) -> Tuple[TickBand, ...]:
        return tuple(TickBand(m, t) for m, t in zip(self._mins, self._ticks))

    @property
    def floor(self) -> Decimal:
        """The lowest price this table describes."""
        return self._mins[0]

    def tick_at(self, price: Decimal) -> Decimal:
        """The tick that applies at ``price``."""
        if price < self._mins[0]:
            raise ValueError(
                f"{price} is below this tick table, which starts at "
                f"{self._mins[0]}; extend the table rather than assuming the "
                "smallest tick applies")
        return self._ticks[bisect_right(self._mins, price) - 1]

    def on_grid(self, price: Decimal) -> bool:
        """Whether ``price`` is an exact multiple of its band's tick."""
        return self.offset(price) == 0

    def offset(self, price: Decimal) -> Decimal:
        """How far ``price`` sits above the tick at or below it."""
        tick = self.tick_at(price)
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return price - (price // tick) * tick

    def round(self, price: Decimal, mode: Rounding) -> Decimal:
        """Move ``price`` onto the grid under a stated rule.

        ``mode`` is required. The tie modes are separate members rather than a
        flag because an exact half-tick is not an edge case on a grid — a mid
        between adjacent ticks is one every time — so which way it goes is a
        systematic choice, and it should be visible at the call site.
        """
        if not isinstance(mode, Rounding):
            raise TypeError("mode must be a Rounding member")
        tick = self.tick_at(price)
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            lower = (price / tick).to_integral_value(ROUND_FLOOR) * tick
            upper = (price / tick).to_integral_value(ROUND_CEILING) * tick
            if lower == upper:
                return price
            if mode is Rounding.DOWN:
                return lower
            if mode is Rounding.UP:
                return upper
            below, above = price - lower, upper - price
            if below < above:
                return lower
            if above < below:
                return upper
            return lower if mode is Rounding.NEAREST_DOWN else upper

    def executable(self, price: Decimal, side: Side) -> Decimal:
        """Round to a price the venue would take, against the caller.

        A buy rounds down and a sell rounds up, so the rounding never improves
        the trade. Rounding to the nearest tick instead gives a backtest the
        better side of the grid about half the time, which is a real gain
        distributed evenly across every order and attributable to nothing.
        """
        if side is Side.BUY:
            return self.round(price, Rounding.DOWN)
        if side is Side.SELL:
            return self.round(price, Rounding.UP)
        raise ValueError("side must be BUY or SELL")

    def ticks_between(self, low: Decimal, high: Decimal) -> Decimal:
        """How many ticks separate two prices in the same band.

        Refused across a band boundary: the answer would depend on where the
        boundary was crossed, and a single number cannot carry that.
        """
        if high < low:
            raise ValueError("high must not precede low")
        t_low, t_high = self.tick_at(low), self.tick_at(high)
        if t_low != t_high:
            raise ValueError(
                f"{low} and {high} fall in bands with different ticks "
                f"({t_low} and {t_high}); a distance in ticks is not defined "
                "across a band boundary")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return (high - low) / t_low

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"TickTable(name={self.name!r}, bands={len(self._mins)}, "
                f"floor={self._mins[0]})")


class TickSchedule:
    """Tick tables that changed over time, queried as of a moment.

    Tick regimes are revised — pilot programmes, decimalisation, venue rule
    changes — so the grid a price had to sit on is a point-in-time fact like
    index membership or a holiday calendar. Asking before the first table is
    refused rather than answered with the oldest one.
    """

    __slots__ = ("_from", "_tables")

    def __init__(self, tables: Iterable[Tuple[int, TickTable]]) -> None:
        ordered = sorted(tables, key=lambda p: p[0])
        if not ordered:
            raise ValueError("a schedule needs at least one table")
        froms = [ts for ts, _ in ordered]
        if len(set(froms)) != len(froms):
            raise ValueError("two tables take effect at the same instant")
        if froms[0] < 0:
            raise ValueError("effective timestamps must be non-negative")
        self._from: List[int] = froms
        self._tables: List[TickTable] = [t for _, t in ordered]

    @property
    def effective_from_ns(self) -> Tuple[int, ...]:
        return tuple(self._from)

    def at(self, ts_ns: int) -> TickTable:
        """The table in force at ``ts_ns``."""
        if ts_ns < self._from[0]:
            raise ValueError(
                f"{ts_ns} precedes the first tick table in this schedule "
                f"({self._from[0]}); the grid before it is not described here")
        return self._tables[bisect_right(self._from, ts_ns) - 1]

    def __len__(self) -> int:
        return len(self._from)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TickSchedule(tables={len(self._from)})"


def grid_report(prices: Sequence[Decimal], table: TickTable) -> GridReport:
    """Check a set of prices against a grid, and say how many missed it.

    Prices below the table are counted separately rather than treated as
    failures: they are a statement that the table does not cover this file,
    which is a different problem from a series being derived.

    A run of raw prints from one venue should come back entirely on grid. When
    it does not, the usual causes are, in rough order of frequency: the series
    is a mid or a VWAP rather than a print; the history has been back-adjusted
    for splits and dividends; the file mixes venues with different grids; or
    the table is for the wrong period.
    """
    on = off = below = 0
    worst: Optional[Decimal] = None
    example: Optional[Decimal] = None
    for p in prices:
        if p < table.floor:
            below += 1
            continue
        o = table.offset(p)
        if o == 0:
            on += 1
            continue
        off += 1
        if worst is None or o > worst:
            worst, example = o, p
    return GridReport(total=len(prices), on_grid=on, off_grid=off,
                      below_table=below, worst_offset=worst, example=example)


def spread_in_ticks(bid: Decimal, ask: Decimal,
                    table: TickTable) -> Decimal:
    """The bid-ask spread measured in ticks rather than in currency.

    This is the form that compares across instruments and across time, and it
    is the form that shows when a spread has nothing left to give: an
    instrument quoted one tick wide is at the floor the venue permits, and any
    further "improvement" measured on it is measuring the tick.

    A spread below one tick is not a tight market. It means the two sides came
    from different places — a consolidated bid against a single-venue ask, a
    stale side, or a mid mistaken for a quote — so it is worth seeing rather
    than averaging away, and this function reports it as it is.
    """
    if ask < bid:
        raise ValueError("ask must not be below bid")
    return table.ticks_between(bid, ask)


def read_tick_table_csv(
    path: str,
    *,
    min_price_column: str = "min_price",
    tick_column: str = "tick",
    name: str = "",
) -> TickTable:
    """Read bands from a CSV of ``min_price,tick``.

    Rows may be in any order; they are sorted here. An empty file is an error
    rather than an empty grid, because a grid that admits every price is not a
    grid.
    """
    import csv

    from .fileio import open_text

    bands: List[TickBand] = []
    with open_text(path) as fh:
        for row in csv.DictReader(fh):
            bands.append(TickBand(Decimal(row[min_price_column].strip()),
                                  Decimal(row[tick_column].strip())))
    if not bands:
        raise ValueError("no bands in file; a tick table cannot be empty")
    return TickTable(bands, name=name)
