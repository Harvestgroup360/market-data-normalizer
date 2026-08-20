"""Which instruments existed at a given moment, and ranking across them.

The two biases handled elsewhere in this library are about time: a value read
before it was observable (:mod:`mdnorm.align`) and a label that overlaps the
block you are testing on (:mod:`mdnorm.labels`). This module is about the
third one, which is about membership::

    from mdnorm import Universe, cross_section, cross_sectional_rank

    pit = Universe.from_listings(listings)
    ranks = cross_section(rows, cross_sectional_rank, universe=pit)

**A universe assembled today did not exist in the past.** Take the instruments
that are listed and liquid now, pull their history, and rank them against each
other across ten years, and every name in the study is one that survived. The
ones that delisted, went to zero, got acquired or simply stopped trading are
absent — not because the data is missing, but because the list was built after
the fact. This is survivorship bias, and unlike a look-ahead bug it produces no
strange values anywhere: the numbers are all real, they are just the wrong
sample.

**Excluding a name is not the same as it having no data.** A symbol that has
not listed yet, and one that delisted last month, must be outside the
cross-section rather than sitting inside it as a blank. The difference matters
because a blank inside a cross-section is usually treated as missing at
random — dropped, imputed, or given a neutral rank — and the instruments that
disappear from a market are the opposite of random.

**The size of the cross-section changes, and that is correct.** Percentile
ranks are computed against the members present at that moment, so the
denominator moves as instruments list and delist. A fixed denominator would be
tidier and would mean a different thing on every row.

:class:`Universe` answers one question — which symbols were active at time
``t`` — and :func:`mask_to_universe` applies that answer to an aligned matrix,
reporting how many cells it removed. Ranking and standardising then happen
inside each row, across whatever was really there.

Nothing here selects instruments, weights them, or forms a portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import (Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

from .align import AlignedRow
from .features import _PRECISION
from .fileio import open_text
from .timeutil import epoch_to_ns, iso_to_ns

__all__ = [
    "Listing",
    "Universe",
    "mask_to_universe",
    "cross_sectional_rank",
    "cross_sectional_zscore",
    "cross_section",
    "read_listings_csv",
]


@dataclass(frozen=True, slots=True)
class Listing:
    """One instrument's tradable lifetime, as a half-open interval.

    ``listed_ns`` is inclusive and ``delisted_ns`` is exclusive, so an
    instrument is not a member on the day it stops trading. ``delisted_ns`` of
    ``None`` means still listed as far as this record knows — which is a claim
    about the record, not about the instrument.
    """

    symbol: str
    listed_ns: int
    delisted_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        if self.listed_ns < 0:
            raise ValueError("listed_ns must be non-negative")
        if self.delisted_ns is not None and self.delisted_ns <= self.listed_ns:
            raise ValueError(
                f"{self.symbol}: delisted_ns must be after listed_ns"
            )

    def active_at(self, ts_ns: int) -> bool:
        """Whether the instrument was tradable at ``ts_ns``."""
        if ts_ns < self.listed_ns:
            return False
        return self.delisted_ns is None or ts_ns < self.delisted_ns


class Universe:
    """The set of instruments tradable at any given moment."""

    __slots__ = ("_listings",)

    def __init__(self, listings: Iterable[Listing]) -> None:
        by_symbol: Dict[str, List[Listing]] = {}
        for listing in listings:
            by_symbol.setdefault(listing.symbol, []).append(listing)
        # A symbol may list, delist and relist; keep every interval.
        self._listings: Dict[str, Tuple[Listing, ...]] = {
            s: tuple(sorted(v, key=lambda x: x.listed_ns))
            for s, v in by_symbol.items()
        }

    @classmethod
    def from_listings(cls, listings: Iterable[Listing]) -> "Universe":
        return cls(listings)

    @property
    def symbols(self) -> Tuple[str, ...]:
        """Every symbol that ever appears, listed or not, in name order."""
        return tuple(sorted(self._listings))

    def members_at(self, ts_ns: int) -> Tuple[str, ...]:
        """Symbols tradable at ``ts_ns``, in name order.

        Name order rather than insertion order so that two runs over the same
        listings produce the same cross-section.
        """
        return tuple(sorted(
            s for s, ls in self._listings.items()
            if any(x.active_at(ts_ns) for x in ls)
        ))

    def contains(self, symbol: str, ts_ns: int) -> bool:
        return any(x.active_at(ts_ns) for x in self._listings.get(symbol, ()))

    def size_at(self, ts_ns: int) -> int:
        return len(self.members_at(ts_ns))

    def __len__(self) -> int:
        return len(self._listings)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Universe(symbols={len(self._listings)})"


def read_listings_csv(path: str, *, ts_unit: Optional[str] = None) -> List[Listing]:
    """Read ``symbol,listed,delisted`` rows into :class:`Listing` objects.

    An empty ``delisted`` cell means still listed. Timestamps are ISO-8601
    unless ``ts_unit`` selects epoch parsing, matching the rest of the CLI.
    """
    import csv as _csv

    out: List[Listing] = []
    with open_text(path) as fh:
        for lineno, row in enumerate(_csv.DictReader(fh), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                symbol = (row.get("symbol") or "").strip()
                listed = (row.get("listed") or "").strip()
                delisted = (row.get("delisted") or "").strip()

                def parse(text: str) -> int:
                    return (epoch_to_ns(float(text), ts_unit) if ts_unit
                            else iso_to_ns(text))

                out.append(Listing(
                    symbol=symbol,
                    listed_ns=parse(listed),
                    delisted_ns=parse(delisted) if delisted else None,
                ))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def mask_to_universe(
    rows: Sequence[AlignedRow], universe: Universe
) -> Tuple[List[AlignedRow], int]:
    """Blank every value whose symbol was not a member at that row's time.

    Returns the masked rows and the number of cells removed. That count is
    worth looking at: on a survivorship-free universe it is large, and if it
    comes back zero on a multi-year study the listing data is probably the
    present-day membership rather than a historical record.
    """
    masked: List[AlignedRow] = []
    removed = 0
    for row in rows:
        values: Dict[str, Optional[Decimal]] = {}
        ages: Dict[str, Optional[int]] = {}
        for name, value in row.values.items():
            if universe.contains(name, row.ts_ns):
                values[name] = value
                ages[name] = row.ages_ns.get(name)
            else:
                if value is not None:
                    removed += 1
                values[name] = None
                ages[name] = None
        masked.append(AlignedRow(ts_ns=row.ts_ns, values=values, ages_ns=ages))
    return masked, removed


# -- inside one row ----------------------------------------------------------


_Row = Mapping[str, Optional[Decimal]]


def _present(values: _Row) -> List[Tuple[str, Decimal]]:
    return [(k, v) for k, v in values.items() if v is not None]


def cross_sectional_rank(
    values: _Row, *, ascending: bool = True, pct: bool = False
) -> Dict[str, Optional[Decimal]]:
    """Rank the values within one row, averaging ties.

    Missing entries stay missing: they are not ranked last, and they are not
    given a middle rank. Something that was not trading has no place in the
    ordering at all, and putting it at either end is a decision the data does
    not support.

    With ``pct`` the result is scaled to ``[0, 1]`` against the number of
    values actually present, so the denominator follows the size of the
    cross-section. A single value has no percentile and returns ``None``.
    """
    present = _present(values)
    out: Dict[str, Optional[Decimal]] = {k: None for k in values}
    if not present:
        return out
    n = len(present)
    ordered = sorted(present, key=lambda kv: kv[1], reverse=not ascending)

    i = 0
    ranks: Dict[str, Decimal] = {}
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        # positions i..j share a value; give them the average rank
        average = Decimal(i + j + 2) / 2
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = average
        i = j + 1

    if not pct:
        out.update(ranks)
        return out
    if n == 1:
        return out
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for k, r in ranks.items():
            out[k] = (r - 1) / (n - 1)
    return out


def cross_sectional_zscore(
    values: _Row, *, ddof: int = 1
) -> Dict[str, Optional[Decimal]]:
    """Standardise the values within one row, against that row only.

    Uses the mean and standard deviation of the members present at this
    timestamp, so nothing from another moment in time enters. A row with fewer
    than ``ddof + 2`` values, or one where every value is identical, has no
    usable dispersion and returns ``None`` throughout rather than zeros.
    """
    if ddof < 0:
        raise ValueError("ddof must be non-negative")
    present = _present(values)
    out: Dict[str, Optional[Decimal]] = {k: None for k in values}
    n = len(present)
    if n - ddof < 2:
        return out
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        mean = sum((v for _, v in present), Decimal(0)) / n
        var = sum(((v - mean) ** 2 for _, v in present), Decimal(0)) / (n - ddof)
        if var == 0:
            return out
        sd = var.sqrt()
        for k, v in present:
            out[k] = (v - mean) / sd
    return out


def cross_section(
    rows: Sequence[AlignedRow],
    fn: Callable[[_Row], Dict[str, Optional[Decimal]]],
    *,
    universe: Optional[Universe] = None,
) -> List[AlignedRow]:
    """Apply a cross-sectional function to every row of an aligned matrix.

    Pass ``universe`` and each row is masked to its point-in-time membership
    before the function runs, which is the difference between ranking against
    the instruments that existed then and ranking against the ones that exist
    now. The ages of the original observations are carried through unchanged,
    since a rank is as old as the price it came from.
    """
    source = rows
    if universe is not None:
        source, _ = mask_to_universe(rows, universe)
    return [
        AlignedRow(ts_ns=row.ts_ns, values=fn(row.values), ages_ns=dict(row.ages_ns))
        for row in source
    ]
