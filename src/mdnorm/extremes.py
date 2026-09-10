"""A fat tail and a fat finger look identical in a price series.

One of them is the risk you are paid to carry and the other is a typo, and
nothing in the number distinguishes them. So this module measures what
removing them would cost and refuses to remove anything::

    from mdnorm import flag_extremes, clip_effect, tail_contribution

    flag_extremes(returns, sigma=5, robust=True)     # 30 found
    flag_extremes(returns, sigma=5)                  # 0 found
    clip_effect(returns, sigma=4).sharpe_inflation   # 1.34x

**The outliers hide inside the ruler used to find them.** A z-score divides
by a standard deviation computed from the same sample, and every extreme
observation inflates it. Thirty contaminated points in a thousand raised the
standard deviation by 1.42x on our worked example, which pushed every one of
them from six sigma to four and a bit — so at a five-sigma cut the classic
score found *none of them* and the robust score found *all thirty*. This is
called masking and it is not a corner case; it is what happens whenever
contamination is more than about one per cent.

**So the scale is computed two ways and you choose.** ``robust=True`` uses
the median and the median absolute deviation, which the extremes cannot move.
It is the right default for detection and the wrong one for description,
because a robust scale deliberately ignores the tail you may be trying to
measure. Both are offered and neither is assumed.

**Clipping always lowers the measured volatility, and what it does to the
Sharpe ratio depends on which side the tail was on.** Where the extremes are
roughly symmetric the mean survives and the ratio rises, which is the case
people have in mind. Where the tail is one-sided the clip takes the profit
with it, and the ratio falls — on the worked example in the README, 0.88x.
Both are distortions of the same size and neither is the safe direction, so
:func:`clip_effect` reports the shift without asserting a sign, and hands
back no data. Trimming a sample stays a decision somebody makes on purpose.

**The threshold is chosen after seeing the data.** That makes it an in-sample
decision applied to the whole history, and it is a parameter, which means it
belongs in a run manifest beside the trial count — see
:mod:`mdnorm.provenance`. There is no default sigma here for the same reason
:mod:`mdnorm.coverage` has no default gap and :mod:`mdnorm.halts` infers no
pause.

**Concentration is the question behind all of it.** If half the profit came
from nine days out of a thousand, the strategy is a bet on nine days, and
whether those days were real is the only question that matters.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Spread",
    "Extreme",
    "TailReport",
    "ClipEffect",
    "spread",
    "zscores",
    "flag_extremes",
    "tail_contribution",
    "concentration",
    "clip_effect",
    "winsorise",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
# The constant that makes the median absolute deviation match a standard
# deviation on normally distributed data. Without it the two scales are not
# comparable and a sigma means something different in each.
_MAD_TO_SIGMA = Decimal("1.4826")


@dataclass(frozen=True, slots=True)
class Spread:
    """A centre and a scale, and which pair of them was used."""

    centre: Decimal
    scale: Decimal
    robust: bool

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ValueError("a scale is not negative")

    @property
    def degenerate(self) -> bool:
        """Whether the scale is zero, so no score can be formed."""
        return self.scale == 0


@dataclass(frozen=True, slots=True)
class Extreme:
    """One flagged observation, with the score that flagged it."""

    index: int
    value: Decimal
    score: Decimal


@dataclass(frozen=True, slots=True)
class TailReport:
    """How much of a series lives in its largest few observations."""

    observations: int
    counted: int
    total: Decimal
    tail_total: Decimal

    @property
    def rest(self) -> Decimal:
        """The total with the counted observations taken out."""
        return self.total - self.tail_total

    @property
    def tail_share(self) -> Optional[Decimal]:
        """The tail's contribution as a share of the total.

        ``None`` when the total is zero, which is not a failure: a series
        that nets to nothing has no total for a part of it to be a share of,
        and returning zero there would read as "the tail contributed
        nothing".
        """
        if self.total == 0:
            return None
        return self.tail_total / self.total


@dataclass(frozen=True, slots=True)
class ClipEffect:
    """What winsorising would do, measured rather than done."""

    sigma: Decimal
    robust: bool
    observations: int
    clipped: int
    mean_before: Decimal
    mean_after: Decimal
    volatility_before: Decimal
    volatility_after: Decimal

    @property
    def clipped_share(self) -> Optional[Decimal]:
        if self.observations == 0:
            return None
        return Decimal(self.clipped) / self.observations

    @property
    def sharpe_before(self) -> Optional[Decimal]:
        if self.volatility_before == 0:
            return None
        return self.mean_before / self.volatility_before

    @property
    def sharpe_after(self) -> Optional[Decimal]:
        if self.volatility_after == 0:
            return None
        return self.mean_after / self.volatility_after

    @property
    def sharpe_shift(self) -> Optional[Decimal]:
        """The clipped Sharpe ratio as a multiple of the unclipped one.

        Above one when the extremes were roughly symmetric: the volatility
        falls, the mean is left more or less alone, and the ratio improves.
        Below one when the tail was one-sided, because the clip took the
        profit along with the risk. No sign is assumed here — a figure either
        side of one is the same statement, that a threshold somebody chose
        moved the headline number.

        Read it beside :attr:`sharpe_before`. A ratio is unstable when its
        denominator is near zero, so on a series with almost no mean this can
        be large or negative while both Sharpe ratios are negligible. That is
        arithmetic rather than a finding, and both figures are exposed so it
        cannot be mistaken for one.
        """
        before, after = self.sharpe_before, self.sharpe_after
        if before is None or after is None or before == 0:
            return None
        return after / before

    @property
    def volatility_understated(self) -> Optional[Decimal]:
        """Clipped volatility as a fraction of the volatility being measured."""
        if self.volatility_before == 0:
            return None
        return self.volatility_after / self.volatility_before


def _median(sorted_values: Sequence[Decimal]) -> Decimal:
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, _ZERO) / len(values)


def _pstdev(values: Sequence[Decimal], centre: Decimal) -> Decimal:
    var = sum(((v - centre) ** 2 for v in values), _ZERO) / len(values)
    return var.sqrt()


def spread(values: Sequence[Decimal], *, robust: bool) -> Spread:
    """The centre and scale of a series, computed one of two ways.

    ``robust=False`` is the mean and the population standard deviation.
    ``robust=True`` is the median and the median absolute deviation scaled by
    1.4826, which matches a standard deviation on normal data and, unlike
    one, cannot be moved by the observations being looked for.
    """
    if not values:
        raise ValueError("a spread needs at least one observation")
    if robust:
        ordered = sorted(values)
        centre = _median(ordered)
        deviations = sorted(abs(v - centre) for v in values)
        return Spread(centre=centre,
                      scale=_median(deviations) * _MAD_TO_SIGMA, robust=True)
    centre = _mean(values)
    return Spread(centre=centre, scale=_pstdev(values, centre), robust=False)


def zscores(values: Sequence[Decimal], *, robust: bool) -> List[Decimal]:
    """Signed scores in units of the chosen scale.

    Raises on a series with no spread at all, rather than returning zeros:
    a constant series has no scale, and every score in it would be a division
    by nothing dressed up as a finding.
    """
    s = spread(values, robust=robust)
    if s.degenerate:
        raise ValueError(
            "the scale is zero, so no observation can be scored against it; "
            "a series with no spread has no extremes, only values")
    return [(v - s.centre) / s.scale for v in values]


def flag_extremes(
    values: Sequence[Decimal], *, sigma: Decimal, robust: bool = False
) -> List[Extreme]:
    """Flag observations at or beyond ``sigma``, in input order.

    Nothing is removed and nothing is judged. ``sigma`` is required: what
    counts as extreme depends on the instrument and on what the series is
    for, and a threshold chosen here would make the result a property of
    this library rather than of the data.

    ``robust=True`` is strongly preferable for detection. With ordinary
    scores the flagged observations are themselves inflating the scale they
    are measured against, so a contaminated sample can report no extremes at
    all — see the module docstring for the worked case.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    scores = zscores(values, robust=robust)
    return [Extreme(index=i, value=v, score=z)
            for i, (v, z) in enumerate(zip(values, scores))
            if abs(z) >= sigma]


def tail_contribution(
    values: Sequence[Decimal], *, n: int
) -> TailReport:
    """How much of the total sits in the ``n`` largest observations by size.

    Ranked by absolute value, because the question is which observations
    dominate rather than which were profitable. The signed sum of those
    observations is reported, so a tail of large losses shows as a negative
    contribution rather than as a large one.
    """
    if n < 0:
        raise ValueError("n must not be negative")
    ranked = sorted(range(len(values)), key=lambda i: abs(values[i]),
                    reverse=True)[:n]
    return TailReport(
        observations=len(values),
        counted=len(ranked),
        total=sum(values, _ZERO),
        tail_total=sum((values[i] for i in ranked), _ZERO),
    )


def concentration(
    values: Sequence[Decimal], *, share: Decimal
) -> Optional[int]:
    """How few of the largest gains it takes to make ``share`` of the total.

    Returns ``None`` when the total is not positive, because "half of a
    negative total" is a sentence with no useful meaning. Counts only
    positive observations, largest first: the question is how much of the
    profit rests on how few days.
    """
    if not (0 < share <= 1):
        raise ValueError("share must be above zero and at most one")
    total = sum(values, _ZERO)
    if total <= 0:
        return None
    target = total * share
    running = _ZERO
    for i, v in enumerate(sorted((v for v in values if v > 0), reverse=True),
                          start=1):
        running += v
        if running >= target:
            return i
    return None


def clip_effect(
    values: Sequence[Decimal], *, sigma: Decimal, robust: bool = False
) -> ClipEffect:
    """Measure what winsorising at ``sigma`` would do, without doing it.

    No clipped series comes back. Trimming a sample is a decision with
    consequences for every statistic downstream, and a function that returned
    the trimmed data would make it the path of least resistance. Use
    :func:`winsorise` when the decision has been made deliberately.
    """
    if not values:
        raise ValueError("a clip effect needs at least one observation")
    clipped = winsorise(values, sigma=sigma, robust=robust)
    changed = sum(1 for a, b in zip(values, clipped) if a != b)
    before_centre = _mean(values)
    after_centre = _mean(clipped)
    s = spread(values, robust=robust)
    return ClipEffect(
        sigma=sigma,
        robust=robust,
        observations=len(values),
        clipped=changed,
        mean_before=before_centre,
        mean_after=after_centre,
        volatility_before=_pstdev(values, before_centre),
        volatility_after=_pstdev(clipped, after_centre),
    )


def winsorise(
    values: Sequence[Decimal], *, sigma: Decimal, robust: bool = False
) -> List[Decimal]:
    """Pull every observation back to ``sigma`` from the centre.

    Winsorising rather than dropping, so the sample size is unchanged and a
    later count still means what it says. This is the one function here that
    alters data, and it is separate from :func:`clip_effect` precisely so
    that using it is a visible act rather than a side effect of measuring.

    Whatever ``sigma`` you pass is a parameter chosen after seeing the data.
    Record it — :func:`mdnorm.manifest` exists for exactly this.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    s = spread(values, robust=robust)
    if s.degenerate:
        raise ValueError(
            "the scale is zero, so there is nothing to clip against; a "
            "series with no spread has no extremes, only values")
    lo, hi = s.centre - sigma * s.scale, s.centre + sigma * s.scale
    return [lo if v < lo else hi if v > hi else v for v in values]
