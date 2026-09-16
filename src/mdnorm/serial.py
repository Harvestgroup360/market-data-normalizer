"""The square root of time is a claim about independence.

Every annualised Sharpe ratio in circulation was produced by multiplying a
per-period one by the square root of the calendar. That step is not a unit
conversion. It is an assumption — that the returns are serially uncorrelated —
and when the assumption fails the factor is simply the wrong number::

    from mdnorm import serial_report

    rep = serial_report(monthly, periods_per_year=12, max_lag=11)
    rep.naive_factor          # 3.4641 — the square root of twelve
    rep.corrected_factor      # 2.6148 — what this series supports
    rep.naive_annualised      # 1.2124
    rep.corrected_annualised  # 0.9152

**Positive serial correlation inflates the factor, and positive serial
correlation is the normal state of a smoothed series.** A monthly mark that
carries part of last month's move — an appraisal, a stale quote, a model price
on an instrument that did not trade — has an autocorrelation well above zero.
Its per-period volatility is understated, its per-period Sharpe is overstated,
and then the square root of twelve multiplies the overstatement rather than
correcting it.

**The correction is Lo's, and it is arithmetic rather than a model.** For a
horizon of ``q`` periods the honest factor is ``q / sqrt(q + 2 * sum over k of
(q - k) * rho_k)``. Substituting zero autocorrelation returns ``sqrt(q)``
exactly, which is the point: the familiar factor is the special case, not the
general one.

**The direction is not fixed, and we will not pretend otherwise.** Negative
autocorrelation — bid-ask bounce, a mean-reverting spread — makes the square
root of time *understate*. The report says which way this series goes rather
than assuming the flattering case, and :attr:`SerialReport.difference` is
named for a quantity that can land on either side of zero.

**Truncation is a choice and it is reported.** Lo's factor needs ``q - 1``
autocorrelations. Annualising daily returns means 251 of them, and past the
first handful those are estimated from too few pairs to mean anything.
Supplying fewer is normal and treats the rest as zero — which biases the
result toward the naive answer, the direction that flatters. So
:class:`ScalingFactor` carries the lags it used and the lags the horizon
called for, and says plainly when it was truncated.

**A variance ratio is the same fact stated without a Sharpe.** If a series
were independent, the variance of its ``q``-period returns would be ``q``
times the variance of its one-period returns. :func:`variance_ratio` reports
what it actually is. Above one is trending, below one is mean-reverting, and
neither is a verdict.

There is no default calendar, no default truncation lag and no threshold at
which an autocorrelation is called large. The caller states all three, for the
reason the rest of this library gives: a constant chosen in here would make
the answer partly a property of the library, and a reader would have no way of
knowing which part.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "ScalingFactor",
    "SerialReport",
    "scaling_factor",
    "annualise_sharpe",
    "variance_ratio",
    "long_run_variance",
    "serial_report",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_PRECISION = 40


def _check(values: Sequence[Decimal], *, minimum: int) -> None:
    if len(values) < minimum:
        raise ValueError(
            f"a series needs at least {minimum} observations, got "
            f"{len(values)}")


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, _ZERO) / Decimal(len(values))


@dataclass(frozen=True, slots=True)
class ScalingFactor:
    """What a per-period figure should be multiplied by, and what it usually is.

    ``naive`` is the square root of the horizon. ``corrected`` is Lo's factor
    given the autocorrelations supplied. They are equal exactly when those
    autocorrelations sum to zero under the weighting, which is not the same as
    every one of them being zero.
    """

    periods: int
    naive: Decimal
    corrected: Decimal
    lags_used: int
    lags_needed: int

    @property
    def truncated(self) -> bool:
        """Whether lags beyond those supplied were treated as zero.

        Treating them as zero pulls the corrected factor toward the naive one.
        On a positively autocorrelated series that is the flattering
        direction, so a truncated result is a lower bound on the correction
        rather than a neutral approximation of it.
        """
        return self.lags_used < self.lags_needed

    @property
    def ratio(self) -> Decimal:
        """Corrected over naive. Below one when the naive factor overstates."""
        return self.corrected / self.naive

    @property
    def naive_overstates(self) -> bool:
        """Whether the square root of time is the larger of the two.

        True for a positively autocorrelated series and false for a mean
        reverting one. Read it before reading :attr:`ratio`, because the
        interesting cases occur in both directions.
        """
        return self.naive > self.corrected


@dataclass(frozen=True, slots=True)
class SerialReport:
    """A return series, its serial correlation, and what that does to a year."""

    observations: int
    periods_per_year: int
    autocorrelations: Tuple[Decimal, ...]
    factor: ScalingFactor
    sharpe: Optional[Decimal]
    variance_ratio: Optional[Decimal]

    @property
    def naive_annualised(self) -> Optional[Decimal]:
        """The per-period Sharpe times the square root of the calendar."""
        if self.sharpe is None:
            return None
        return self.sharpe * self.factor.naive

    @property
    def corrected_annualised(self) -> Optional[Decimal]:
        """The per-period Sharpe times the factor this series supports."""
        if self.sharpe is None:
            return None
        return self.sharpe * self.factor.corrected

    @property
    def difference(self) -> Optional[Decimal]:
        """Naive less corrected.

        Deliberately not called an overstatement. It is positive on a
        positively autocorrelated series and negative on a mean reverting one,
        and a name that asserted a direction would be wrong half the time.
        :attr:`ScalingFactor.naive_overstates` is the flag that says which
        case this is.
        """
        naive = self.naive_annualised
        corrected = self.corrected_annualised
        if naive is None or corrected is None:
            return None
        return naive - corrected

    @property
    def first_order(self) -> Optional[Decimal]:
        """The lag-one autocorrelation, or ``None`` if none was computed."""
        return self.autocorrelations[0] if self.autocorrelations else None

    @property
    def thin_lags(self) -> bool:
        """Whether the deepest lag rests on fewer than thirty pairs.

        Flagged rather than corrected. An autocorrelation at lag k is computed
        from ``n - k`` pairs, and the estimate stops meaning much long before
        that count reaches zero. Where the line falls is a judgement about the
        series, so this reports the condition and leaves the judgement alone.
        """
        deepest = len(self.autocorrelations)
        return deepest > 0 and (self.observations - deepest) < 30


def scaling_factor(autocorrelations: Sequence[Decimal], *,
                   periods: int) -> ScalingFactor:
    """Lo's annualisation factor for a horizon of ``periods``.

    ``q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k)``, the factor from
    Lo (2002), *The Statistics of Sharpe Ratios*. Under independence every
    ``rho_k`` is zero, the sum vanishes and the factor is ``sqrt(q)``.

    Supply ``rho_1`` first. Fewer than ``q - 1`` autocorrelations is allowed
    and the remainder is treated as zero; the result records both counts so
    the assumption is visible. Extra autocorrelations past ``q - 1`` are
    ignored, because they carry no weight in the sum.

    Raises :class:`ArithmeticError` when the weighted sum drives the radicand
    to zero or below. That is a real state — strong negative autocorrelation
    at short lags can do it — and there is no factor to report for it, so it
    raises rather than returning something shaped like an answer.
    """
    if periods < 1:
        raise ValueError(f"periods must be positive, got {periods}")
    needed = periods - 1
    rhos = list(autocorrelations[:needed])
    for i, rho in enumerate(rhos, start=1):
        if not (-1 <= rho <= 1):
            raise ValueError(
                f"autocorrelation at lag {i} is {rho}; a correlation lies "
                "between -1 and 1")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        q = Decimal(periods)
        weighted = sum((Decimal(periods - k) * rho
                        for k, rho in enumerate(rhos, start=1)), _ZERO)
        radicand = q + _TWO * weighted
        if radicand <= 0:
            raise ArithmeticError(
                f"the weighted autocorrelation sum drives the variance of a "
                f"{periods}-period return to {radicand}, which is not a "
                "variance; there is no scaling factor for this series")
        return ScalingFactor(
            periods=periods,
            naive=q.sqrt(),
            corrected=q / radicand.sqrt(),
            lags_used=len(rhos),
            lags_needed=needed,
        )


def annualise_sharpe(sharpe: Decimal, *, periods_per_year: int,
                     autocorrelations: Sequence[Decimal]) -> Decimal:
    """A per-period Sharpe ratio carried to a year, honestly.

    The serially correct counterpart to :func:`mdnorm.annualise_sharpe`, which
    multiplies by the square root of the calendar and is right only when the
    returns are independent. Pass the same per-period ratio to both and the
    difference is what the independence assumption was worth.
    """
    factor = scaling_factor(autocorrelations, periods=periods_per_year)
    return sharpe * factor.corrected


def variance_ratio(values: Sequence[Decimal], *, horizon: int) -> Decimal:
    """Variance of ``horizon``-period returns over ``horizon`` times the
    variance of one-period returns.

    One under independence. Above one means the series trends — a move is
    more likely than not to be followed by another in the same direction, so
    the long-horizon variance grows faster than the calendar. Below one means
    it reverts.

    Overlapping windows are used, because non-overlapping ones discard most of
    the sample at any useful horizon. Both variances are computed about the
    same whole-series mean and divided by the same count convention, so the
    ratio is not distorted by a difference in estimator between numerator and
    denominator.

    This is a description, not a test. Whether a given departure from one is
    larger than sampling noise depends on the length of the series and on
    assumptions about the return process, and this library does not hold
    those.

    The estimator is biased toward one at long horizons, and the bias is not
    small: on four thousand draws from a process whose asymptotic ratio at
    twelve periods is 1.86, this returns about 1.66. The reason is that a
    ``q``-period window fits ``n - q + 1`` times rather than ``n``, and the
    windows share most of their observations. So a ratio near one is weak
    evidence of independence, while a ratio far from one is strong evidence
    against it — the error runs in the direction that makes a dependent
    series look independent.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    _check(values, minimum=horizon + 1)
    if horizon == 1:
        return _ONE

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        n = len(values)
        mu = _mean(values)
        short = sum(((v - mu) ** 2 for v in values), _ZERO) / Decimal(n)
        if short == 0:
            raise ArithmeticError(
                "the series does not vary, so its variance ratio would "
                "divide by zero")
        sums = [sum(values[i:i + horizon], _ZERO)
                for i in range(n - horizon + 1)]
        mu_long = mu * Decimal(horizon)
        long = (sum(((s - mu_long) ** 2 for s in sums), _ZERO)
                / Decimal(len(sums)))
        return long / (Decimal(horizon) * short)


def long_run_variance(values: Sequence[Decimal], *, max_lag: int) -> Decimal:
    """Newey-West long-run variance with Bartlett weights.

    ``gamma_0 + 2 * sum_{k=1}^{L} (1 - k / (L + 1)) * gamma_k``. The
    declining weights are not decoration: the raw sum of autocovariances can
    come out negative, which is not a variance, and the Bartlett taper
    guarantees it cannot.

    This is the variance that belongs under the square root in a standard
    error when the series is serially correlated. On an independent series it
    collapses to the ordinary variance, so the ratio of the two is a direct
    measure of what the correlation costs.

    ``max_lag`` has no default. How far the dependence runs is a property of
    the series, and a rule of thumb baked in here would quietly become a
    property of the library instead.
    """
    if max_lag < 0:
        raise ValueError(f"max_lag cannot be negative, got {max_lag}")
    _check(values, minimum=max_lag + 2)

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        n = len(values)
        mu = _mean(values)
        dev = [v - mu for v in values]

        def gamma(k: int) -> Decimal:
            return sum((dev[t] * dev[t - k] for t in range(k, n)),
                       _ZERO) / Decimal(n)

        total = gamma(0)
        for k in range(1, max_lag + 1):
            weight = _ONE - Decimal(k) / Decimal(max_lag + 1)
            total += _TWO * weight * gamma(k)
        return total


def serial_report(values: Sequence[Decimal], *, periods_per_year: int,
                  max_lag: int) -> SerialReport:
    """Everything above, on one series, in one call.

    The Sharpe ratio comes from :func:`mdnorm.sharpe_ratio` against a zero
    risk-free rate, and is ``None`` when that function declines to produce one
    — a constant series has no ratio, and a zero here would read as a result.
    The variance ratio is computed at a horizon of ``periods_per_year`` when
    the series is long enough to support it and is ``None`` otherwise.
    """
    from .independence import autocorrelation
    from .metrics import sharpe_ratio

    if periods_per_year < 1:
        raise ValueError(
            f"periods_per_year must be positive, got {periods_per_year}")
    if max_lag < 1:
        raise ValueError(f"max_lag must be positive, got {max_lag}")
    _check(values, minimum=max_lag + 2)

    rhos = autocorrelation(values, max_lag=max_lag)
    factor = scaling_factor(rhos, periods=periods_per_year)
    ratio: Optional[Decimal]
    try:
        ratio = variance_ratio(values, horizon=periods_per_year)
    except (ValueError, ArithmeticError):
        ratio = None
    return SerialReport(
        observations=len(values),
        periods_per_year=periods_per_year,
        autocorrelations=tuple(rhos),
        factor=factor,
        sharpe=sharpe_ratio(values),
        variance_ratio=ratio,
    )
