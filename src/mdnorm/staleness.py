"""A price that stopped moving is not a price that stopped being risky.

:mod:`mdnorm.align` has warned since it was written that a frozen price is
uncorrelated with everything and therefore reads as diversification. It gave
no way to find out how much of that you have. This is that measurement::

    from mdnorm import staleness_report, smoothing_bias

    staleness_report(prices, min_run=3).unchanged_share   # 41% never moved
    bias = smoothing_bias(returns)
    bias.volatility_understated                           # 0.79x
    bias.sharpe_inflation                                 # 1.27x

**Repeated values are a fact; staleness is an interpretation.** An illiquid
instrument genuinely does not trade for an hour, and a vendor that carries
yesterday's mark forward produces exactly the same rows. Nothing here decides
which of those you have — it counts the runs and hands you the counts. What
makes a flat stretch suspicious is the instrument and the sampling interval,
which is why there is no default minimum run length.

**Smoothing is where the money is.** A reported series that partly reflects
the previous period's move is a moving average of the true one, and a moving
average has lower variance than what it averages. Lower measured volatility
with the same mean is a higher Sharpe ratio, a lower beta and a smaller
correlation with every other asset — all four moving in the flattering
direction at once, from one cause, with nothing in the arithmetic raising an
objection.

**The adjustment is a model, and it says so.** :func:`smoothing_bias` assumes
the reported return is a two-period weighted average of true returns, infers
the weights from the first-order autocorrelation, and reports the implied
understatement. That is a specific assumption about a specific mechanism. The
result carries ``modelled=True`` so it can never be mistaken for the run
counts, which are arithmetic.

**Where the model cannot fit, it refuses.** A two-period average cannot
produce a first-order autocorrelation above one half, so an observed value
past that is evidence of something other than simple smoothing — a genuine
trend, a stale feed with a longer memory, an overlapping sampling window. The
function returns nothing rather than a figure it cannot support.

Nothing here unsmooths a series in place, and nothing drops a repeated value.
Both would be corrections applied to data whose cause has not been
established.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterator, Optional, Sequence

from .independence import autocorrelation

__all__ = [
    "Run",
    "StalenessReport",
    "SmoothingBias",
    "runs",
    "staleness_report",
    "smoothing_bias",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_HALF = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class Run:
    """A maximal stretch over which the value did not change.

    ``length`` counts the observations, so a value printed once and then
    repeated twice is a run of three.
    """

    start: int
    length: int
    value: Decimal

    def __post_init__(self) -> None:
        if self.length < 1:
            raise ValueError("a run covers at least one observation")

    @property
    def repeats(self) -> int:
        """Observations after the first — the ones that carried no news."""
        return self.length - 1


@dataclass(frozen=True, slots=True)
class StalenessReport:
    """How much of a series never moved, and in what shapes."""

    observations: int
    unchanged: int
    runs: int
    longest_run: int
    in_runs: int
    min_run: int

    @property
    def unchanged_share(self) -> Optional[Decimal]:
        """Share of observations equal to the one before them.

        Counted over the transitions rather than the observations, since the
        first value has nothing to be unchanged from.
        """
        if self.observations < 2:
            return None
        return Decimal(self.unchanged) / (self.observations - 1)

    @property
    def in_runs_share(self) -> Optional[Decimal]:
        """Share of observations sitting inside a run of at least ``min_run``."""
        if self.observations == 0:
            return None
        return Decimal(self.in_runs) / self.observations


@dataclass(frozen=True, slots=True)
class SmoothingBias:
    """What a two-period average of the truth does to the statistics."""

    observations: int
    autocorrelation: Decimal
    weight_current: Optional[Decimal]
    weight_previous: Optional[Decimal]
    variance_ratio: Optional[Decimal]
    modelled: bool = True

    @property
    def fits(self) -> bool:
        """Whether the smoothing model can account for the autocorrelation."""
        return self.variance_ratio is not None

    @property
    def volatility_understated(self) -> Optional[Decimal]:
        """Measured volatility as a fraction of the volatility being measured.

        0.79 means the reported series moves four fifths as much as the thing
        it reports on.
        """
        if self.variance_ratio is None:
            return None
        return self.variance_ratio.sqrt()

    @property
    def sharpe_inflation(self) -> Optional[Decimal]:
        """How far a Sharpe ratio is overstated by the same smoothing.

        The mean survives an average unchanged and the volatility does not,
        so the ratio between them moves by exactly the reciprocal of
        :attr:`volatility_understated`. A correlation or a beta computed
        against an unsmoothed series is understated by the same factor.
        """
        under = self.volatility_understated
        if under is None or under == 0:
            return None
        return _ONE / under


def runs(values: Sequence[Decimal], *, min_run: int = 2) -> Iterator[Run]:
    """Yield maximal runs of an unchanged value, longest-first in the series.

    ``min_run`` is the length at which a flat stretch is worth reporting.
    There is no default that suits every instrument: two identical one-minute
    prints on a liquid future is a stall, and two identical daily marks on a
    corporate bond is a Tuesday.
    """
    if min_run < 1:
        raise ValueError("min_run must be at least 1")
    if not values:
        return

    start = 0
    for i in range(1, len(values) + 1):
        if i < len(values) and values[i] == values[start]:
            continue
        length = i - start
        if length >= min_run:
            yield Run(start=start, length=length, value=values[start])
        start = i


def staleness_report(
    values: Sequence[Decimal], *, min_run: int = 2
) -> StalenessReport:
    """Count the flat stretches without deciding what caused them."""
    found = list(runs(values, min_run=min_run))
    unchanged = sum(1 for i in range(1, len(values))
                    if values[i] == values[i - 1])
    return StalenessReport(
        observations=len(values),
        unchanged=unchanged,
        runs=len(found),
        longest_run=max((r.length for r in found), default=0),
        in_runs=sum(r.length for r in found),
        min_run=min_run,
    )


def smoothing_bias(
    returns: Sequence[Decimal], *, max_lag: int = 1
) -> SmoothingBias:
    """Infer how much a two-period average is hiding, from the autocorrelation.

    The model is ``reported[t] = a * true[t] + b * true[t-1]`` with
    ``a + b = 1`` and ``a >= b``, which is what a partly stale mark looks
    like: some of today's move lands today and the rest arrives tomorrow.
    Under it the first-order autocorrelation is ``a*b / (a^2 + b^2)`` and the
    variance ratio is ``a^2 + b^2``, so one determines the other.

    A negative autocorrelation is not smoothing — it is bid-ask bounce or
    mean reversion, and the model returns weights of one and zero rather than
    pretending the series is being inflated. An autocorrelation above one
    half cannot come from a two-period average at all, and the result comes
    back with ``fits`` False.
    """
    if len(returns) >= 2 and len(set(returns)) == 1:
        raise ValueError(
            "every return is identical, so there is no variation to "
            "attribute; a series that never moves is what staleness_report "
            "is for")
    rho = autocorrelation(returns, max_lag=max_lag)[0]

    if rho <= 0:
        return SmoothingBias(observations=len(returns), autocorrelation=rho,
                             weight_current=_ONE, weight_previous=_ZERO,
                             variance_ratio=_ONE)
    if rho > _HALF:
        return SmoothingBias(observations=len(returns), autocorrelation=rho,
                             weight_current=None, weight_previous=None,
                             variance_ratio=None)

    # k = b / a solves k^2 - k/rho + 1 = 0; the root at or below one is the
    # branch where most of the move still lands in its own period.
    disc = _ONE - 4 * rho * rho
    k = (_ONE - disc.sqrt()) / (2 * rho)
    a = _ONE / (_ONE + k)
    b = k / (_ONE + k)
    return SmoothingBias(observations=len(returns), autocorrelation=rho,
                         weight_current=a, weight_previous=b,
                         variance_ratio=a * a + b * b)
