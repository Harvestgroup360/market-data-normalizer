"""A maximum drawdown is a maximum, and maxima grow with how long you look.

Every tear sheet reports one number for drawdown: the deepest one. It is the
worst single observation in the sample, which makes it an order statistic, and
order statistics are not comparable across samples of different lengths. Run
the same unchanged strategy for ten years instead of two and its worst
drawdown gets deeper, because there were more chances for a bad run to
happen::

    from mdnorm import underwater_report

    rep = underwater_report(equity)
    rep.deepest               # 0.183622 — the number that gets published
    rep.ulcer                 # 0.062179 — rests on every observation
    rep.underwater_share      # 0.7180 — the fraction of the sample below a peak
    rep.longest_underwater    # 289 observations, never recovered in sample

**The headline figure rests on one observation and the alternatives do not.**
:func:`ulcer_index` is the root mean square of the depth at every point and
:func:`pain_index` is its mean. Neither can be moved by a single day, both
fall when a decline is shallow or short, and both are defined without
reference to a maximum. A strategy whose ulcer index is close to its maximum
drawdown spent most of the sample near its worst; one where they are far apart
had a single bad week.

**Depth is not the part people live through.** A book down eight per cent for
three weeks and a book down eight per cent for three years report the same
drawdown. :attr:`UnderwaterReport.longest_underwater` and
:attr:`underwater_share` are the difference, and they are usually absent from
the report entirely.

**A drawdown still open at the end of the sample is not a recovered one.**
Nothing here closes an open decline at the final observation, because doing so
would turn "we do not know yet" into "it ended here", which is the flattering
reading.

**Resampling answers the length question, with an assumption we state.**
:func:`resampled_max_drawdown` draws from the caller's own returns to ask what
the worst drawdown looks like over a horizon they name. It assumes the returns
are exchangeable — that the order does not matter — which is exactly the
assumption :mod:`mdnorm.serial` exists to test. A positively autocorrelated
series has deeper drawdowns than its own reshuffling suggests, because losses
arrive in runs. So a resampled figure on a smoothed series is a lower bound on
the drawdown rather than an estimate of it, and that is the direction that
flatters.

Arithmetic here runs at forty significant digits, which is what the other
statistical modules in this library use and more than :mod:`mdnorm.metrics`,
whose thirty-four is tuned for a different job. The two agree on the deepest
drawdown to far more digits than any real series justifies, and the difference
past that is working precision rather than disagreement.

There is no default horizon, no default path count and no default seed. The
caller states all three, so the result is reproducible and so that the horizon
is a decision somebody made rather than one the library made for them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "UnderwaterReport",
    "ResampledDrawdowns",
    "underwater_curve",
    "time_under_water",
    "underwater_share",
    "longest_underwater",
    "pain_index",
    "ulcer_index",
    "depth_quantile",
    "resampled_max_drawdown",
    "underwater_report",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 40


def _check(curve: Sequence[Decimal]) -> None:
    if not curve:
        raise ValueError("an equity curve needs at least one observation")


@dataclass(frozen=True, slots=True)
class ResampledDrawdowns:
    """What the worst drawdown looks like over a horizon, by reshuffling.

    ``depths`` is sorted ascending, one entry per path. The quantiles are of
    that distribution, not of anything analytic.
    """

    paths: int
    periods: int
    depths: Tuple[Decimal, ...]

    @property
    def median(self) -> Decimal:
        return self.quantile(Decimal("0.5"))

    @property
    def worst(self) -> Decimal:
        return self.depths[-1]

    def quantile(self, level: Decimal) -> Decimal:
        """Linear-interpolated quantile of the simulated worst drawdowns.

        ``quantile(Decimal("0.95"))`` is the depth exceeded by one path in
        twenty — a statement about the resampling, not a confidence interval
        about the strategy.
        """
        if not (0 <= level <= 1):
            raise ValueError(f"level lies in [0, 1], got {level}")
        if len(self.depths) == 1:
            return self.depths[0]
        pos = level * Decimal(len(self.depths) - 1)
        low = int(pos)
        high = min(low + 1, len(self.depths) - 1)
        frac = pos - Decimal(low)
        return self.depths[low] + frac * (self.depths[high] - self.depths[low])


@dataclass(frozen=True, slots=True)
class UnderwaterReport:
    """How deep, how long, and how much of the sample was spent below a peak."""

    observations: int
    deepest: Decimal
    pain: Decimal
    ulcer: Decimal
    periods_under_water: int
    longest_underwater: int
    open_at_end: bool

    @property
    def underwater_share(self) -> Decimal:
        """Fraction of observations spent below a previous peak."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(self.periods_under_water) / Decimal(self.observations)

    @property
    def concentration(self) -> Optional[Decimal]:
        """Ulcer index over the deepest drawdown.

        Near one when the curve spent the sample close to its worst level;
        near zero when the maximum was a single excursion the rest of the
        sample knows nothing about. ``None`` when the curve never fell, which
        is a statement about the sample rather than a concentration of zero.
        """
        if self.deepest == 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.ulcer / self.deepest

    @property
    def never_fell(self) -> bool:
        """Whether the curve is monotone, in which case every figure is zero.

        A curve that never fell has no drawdown to report. Read this before
        reading any of the other numbers, because zeros here mean absence
        rather than excellence.
        """
        return self.deepest == 0


def underwater_curve(curve: Sequence[Decimal]) -> List[Decimal]:
    """Depth below the running maximum at every point, as a positive fraction.

    Zero where the curve is at a new high. The series this returns is what
    every other function in the module is computed from, and it is returned
    rather than hidden so a caller can plot it or apply their own statistic.

    A running peak of zero or below has no meaningful percentage depth, so the
    curve is refused rather than reported against a denominator that would
    make the numbers arbitrary.
    """
    _check(curve)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        out: List[Decimal] = []
        peak = curve[0]
        for i, value in enumerate(curve):
            if value > peak:
                peak = value
            if peak <= 0:
                raise ArithmeticError(
                    f"the running peak at observation {i} is {peak}; a "
                    "percentage depth below a non-positive peak is not "
                    "defined. Pass an equity curve rather than a series of "
                    "returns.")
            out.append((peak - value) / peak)
        return out


def time_under_water(curve: Sequence[Decimal]) -> int:
    """Observations spent strictly below a previous peak."""
    return sum(1 for d in underwater_curve(curve) if d > 0)


def underwater_share(curve: Sequence[Decimal]) -> Decimal:
    """Fraction of the sample spent below a previous peak."""
    depths = underwater_curve(curve)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return Decimal(sum(1 for d in depths if d > 0)) / Decimal(len(depths))


def longest_underwater(curve: Sequence[Decimal]) -> int:
    """The longest unbroken run below a previous peak.

    A run still open at the final observation is counted at its length so far.
    It is not extended, and it is not discarded either: the report carries
    ``open_at_end`` so a reader can tell whether the longest stretch had ended.
    """
    longest = current = 0
    for d in underwater_curve(curve):
        current = current + 1 if d > 0 else 0
        longest = max(longest, current)
    return longest


def pain_index(curve: Sequence[Decimal]) -> Decimal:
    """Mean depth below the running peak, over every observation.

    Unlike a maximum this uses the whole sample, so one bad day cannot set it
    and a long shallow decline is not free.
    """
    depths = underwater_curve(curve)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return sum(depths, _ZERO) / Decimal(len(depths))


def ulcer_index(curve: Sequence[Decimal]) -> Decimal:
    """Root mean square depth below the running peak.

    Martin and McCann (1989). The squaring weights deep declines more heavily
    than shallow ones while still reading every observation, which puts it
    between the pain index and the maximum in how much it cares about the
    worst stretch.
    """
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        depths = underwater_curve(curve)
        mean_square = sum((d * d for d in depths), _ZERO) / Decimal(len(depths))
        return mean_square.sqrt()


def depth_quantile(curve: Sequence[Decimal], *, level: Decimal) -> Decimal:
    """The depth that the curve was at or below for ``level`` of the sample.

    ``level=Decimal("0.95")`` is the depth exceeded on one observation in
    twenty. This is the maximum's robust cousin: it moves when a fifth of the
    sample moves, rather than when one observation does.
    """
    if not (0 <= level <= 1):
        raise ValueError(f"level lies in [0, 1], got {level}")
    depths = sorted(underwater_curve(curve))
    if len(depths) == 1:
        return depths[0]
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        pos = level * Decimal(len(depths) - 1)
        low = int(pos)
        high = min(low + 1, len(depths) - 1)
        frac = pos - Decimal(low)
        return depths[low] + frac * (depths[high] - depths[low])


def resampled_max_drawdown(returns: Sequence[Decimal], *, periods: int,
                           paths: int, seed: int) -> ResampledDrawdowns:
    """Worst drawdown over ``periods``, from reshuffling the caller's returns.

    Each path draws ``periods`` returns with replacement from the series
    supplied, compounds them into a curve, and records its deepest decline.
    The answer is a distribution rather than a number, because the worst
    drawdown of a sample is itself a random quantity and reporting one value
    for it is most of the problem this module describes.

    **The assumption is exchangeability and it is not innocent.** Drawing with
    replacement destroys any serial correlation in the series, and losses that
    arrive in runs make drawdowns deeper than independent losses do. On a
    positively autocorrelated series — which is what
    :mod:`mdnorm.serial` measures — the resampled distribution therefore sits
    shallower than the truth. It is a lower bound on the drawdown, which is
    the direction that flatters.

    ``periods``, ``paths`` and ``seed`` are all required. The seed makes the
    result reproducible; the other two are decisions about what question is
    being asked, and a library that chose them would be answering a different
    one.
    """
    _check(returns)
    for name, count in (("periods", periods), ("paths", paths)):
        if count < 1:
            raise ValueError(f"{name} must be positive, got {count}")
    for i, r in enumerate(returns):
        if r <= -1:
            raise ValueError(
                f"return {i} is {r}; a simple return at or below -1 empties "
                "the account and cannot be compounded. Convert log returns "
                "with mdnorm.to_simple first.")

    rng = random.Random(seed)
    n = len(returns)
    depths: List[Decimal] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for _ in range(paths):
            value = _ONE
            peak = _ONE
            worst = _ZERO
            for _ in range(periods):
                value *= _ONE + returns[rng.randrange(n)]
                if value > peak:
                    peak = value
                depth = (peak - value) / peak
                if depth > worst:
                    worst = depth
            depths.append(worst)
    return ResampledDrawdowns(paths=paths, periods=periods,
                              depths=tuple(sorted(depths)))


def underwater_report(curve: Sequence[Decimal]) -> UnderwaterReport:
    """Every measurement above, on one curve, in one call.

    Everything here is measured from the curve supplied. Nothing is
    extrapolated to a different sample length — that is what
    :func:`resampled_max_drawdown` is for, and it carries an assumption this
    does not.
    """
    depths = underwater_curve(curve)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        under = sum(1 for d in depths if d > 0)
        longest = current = 0
        for d in depths:
            current = current + 1 if d > 0 else 0
            longest = max(longest, current)
        mean_square = sum((d * d for d in depths), _ZERO) / Decimal(len(depths))
        return UnderwaterReport(
            observations=len(depths),
            deepest=max(depths),
            pain=sum(depths, _ZERO) / Decimal(len(depths)),
            ulcer=mean_square.sqrt(),
            periods_under_water=under,
            longest_underwater=longest,
            open_at_end=depths[-1] > 0,
        )
