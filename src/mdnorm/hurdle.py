"""A return has to beat something, and what it beats is a decision.

Every performance statistic in this library measures a return against a
hurdle, and the hurdle is almost always left at zero or set to one constant
for the whole sample. Neither is free. Cash paid close to nothing for a decade
and then five per cent, so a strategy evaluated against zero is credited with
the cash return, and a strategy evaluated against one average rate is credited
or charged depending on *when* it made its money::

    from mdnorm import hurdle_comparison

    cmp = hurdle_comparison(returns, rates, ddof=1)
    cmp.sharpe_raw        # 0.075921 — no hurdle at all
    cmp.sharpe_constant   # 0.048163 — the mean rate, subtracted once
    cmp.sharpe_series     # 0.046004 — the rate that was actually paid
    cmp.rate_correlation  # 0.307 — why the last two differ

**Subtracting a constant and subtracting a series are different operations.**
The first moves the mean and leaves the volatility alone. The second changes
both, because a moving rate has a variance of its own and a covariance with
the returns. The two agree only when the rate never moved, and the direction
of the disagreement is the sign of that covariance — which is a property of
the sample, not something this module will predict for you.

**A quoted rate is an annual number and returns are not.** Turning five per
cent a year into a daily hurdle by dividing by the day count and by
compounding give different answers, and the gap runs the whole length of the
sample in one direction. :func:`per_period_rate` does both and makes you say
which; there is no default, for the same reason there is no default
annualisation factor anywhere else here.

**Day-count bases are not interchangeable.** Money-market rates — SOFR,
EURIBOR, most deposit quotes — are quoted on a 360-day year. Using one against
a 365-day calendar understates the hurdle by 365/360, which is 1.39 per cent
of the rate, every period, in the flattering direction.
:func:`rebase` converts between bases and refuses to guess which one a number
arrived on.

**A benchmark is a hurdle too, and a worse-behaved one.** :func:`active_returns`
and :func:`tracking_error` measure against a series somebody chose. Nothing
here will tell you the choice was right: an information ratio is a statement
about a strategy *and* about a benchmark, and the module reports both legs so
that stays visible.

What this module will not do is supply a rate. No cash curve ships here, for
the same reason no factor data ships with :mod:`mdnorm.exposure`: bundling one
would make every answer partly a property of whose curve we picked.

Arithmetic runs at forty significant digits, matching the other statistical
modules. Cross-module comparisons with :mod:`mdnorm.metrics`, which works at
thirty-four, agree far past anything a real series carries; the difference
beyond that is working precision rather than disagreement.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Optional, Sequence

__all__ = [
    "DayCount",
    "ACT_360",
    "ACT_365",
    "THIRTY_360",
    "ExcessReport",
    "ActiveReport",
    "HurdleComparison",
    "per_period_rate",
    "rebase",
    "excess_returns",
    "active_returns",
    "tracking_error",
    "information_ratio",
    "hurdle_comparison",
    "excess_report",
    "active_report",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 40


@dataclass(frozen=True, slots=True)
class DayCount:
    """A day-count basis: the denominator a quoted rate was divided by.

    Carried as an object rather than an integer so a rate cannot be rebased
    from a basis nobody stated. The three that cover most quotes are provided;
    build others directly.
    """

    name: str
    days: int

    def __post_init__(self) -> None:
        if self.days < 1:
            raise ValueError(f"a day count needs a positive year, got {self.days}")


ACT_360 = DayCount("ACT/360", 360)
ACT_365 = DayCount("ACT/365", 365)
THIRTY_360 = DayCount("30/360", 360)


def _check_pair(a: Sequence[Decimal], b: Sequence[Decimal],
                names: tuple[str, str]) -> None:
    if not a:
        raise ValueError(f"{names[0]} needs at least one observation")
    if len(a) != len(b):
        raise ValueError(
            f"{names[0]} has {len(a)} observations and {names[1]} has "
            f"{len(b)}; they must be the same series measured over the same "
            "periods. Align them before subtracting — mdnorm.align exists for "
            "this — rather than truncating to the shorter one here.")


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, _ZERO) / Decimal(len(values))


def _stdev(values: Sequence[Decimal], *, ddof: int) -> Optional[Decimal]:
    n = len(values)
    if n - ddof < 1:
        return None
    m = _mean(values)
    var = sum(((v - m) * (v - m) for v in values), _ZERO) / Decimal(n - ddof)
    if var <= 0:
        return None
    return var.sqrt()


@dataclass(frozen=True, slots=True)
class ExcessReport:
    """A return series measured against a rate series, period by period."""

    observations: int
    mean_return: Decimal
    mean_rate: Decimal
    mean_excess: Decimal
    volatility_return: Optional[Decimal]
    volatility_excess: Optional[Decimal]
    rate_volatility: Optional[Decimal]

    @property
    def rate_moved(self) -> bool:
        """Whether the hurdle varied at all over the sample.

        When this is false a constant rate and the series agree exactly, and
        the rest of this module has nothing to say that :mod:`mdnorm.metrics`
        does not.
        """
        return self.rate_volatility is not None and self.rate_volatility > 0

    @property
    def share_credited_to_cash(self) -> Optional[Decimal]:
        """Mean rate over mean return: how much of the gross was the hurdle.

        ``None`` when the mean return is not positive, because a share of a
        non-positive number reads as a magnitude and is not one. Above one
        means the hurdle exceeded the return.
        """
        if self.mean_return <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.mean_rate / self.mean_return


@dataclass(frozen=True, slots=True)
class ActiveReport:
    """A return series measured against a benchmark series."""

    observations: int
    mean_return: Decimal
    mean_benchmark: Decimal
    mean_active: Decimal
    tracking_error: Optional[Decimal]
    information_ratio: Optional[Decimal]

    @property
    def benchmark_share(self) -> Optional[Decimal]:
        """Mean benchmark over mean return, on the same terms as the hurdle.

        ``None`` when the mean return is not positive. A value near one says
        the strategy earned what the benchmark earned, which is a statement
        about both of them.
        """
        if self.mean_return <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.mean_benchmark / self.mean_return


@dataclass(frozen=True, slots=True)
class HurdleComparison:
    """The same Sharpe ratio under three hurdles, and why they differ.

    All three are per-period and none is annualised. See
    :func:`mdnorm.metrics.annualise_sharpe`, and :mod:`mdnorm.serial` before
    trusting the square root of the calendar on a smoothed series.
    """

    observations: int
    sharpe_raw: Optional[Decimal]
    sharpe_constant: Optional[Decimal]
    sharpe_series: Optional[Decimal]
    mean_rate: Decimal
    rate_volatility: Optional[Decimal]
    rate_correlation: Optional[Decimal]

    @property
    def constant_series_gap(self) -> Optional[Decimal]:
        """Constant-rate Sharpe minus series Sharpe.

        Called a gap and not an overstatement. Which one is larger depends on
        the sign of :attr:`rate_correlation` and on how the rate's own
        variance lands, both of which are properties of the sample. A positive
        gap means the constant-rate figure is the higher of the two here; it
        does not mean it is the higher one in general.
        """
        if self.sharpe_constant is None or self.sharpe_series is None:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.sharpe_constant - self.sharpe_series

    @property
    def zero_hurdle_gap(self) -> Optional[Decimal]:
        """Raw Sharpe minus series Sharpe: the cost of not subtracting a rate.

        This one does have a direction, and it is the direction that flatters:
        a non-negative mean rate can only make the raw figure the larger.
        """
        if self.sharpe_raw is None or self.sharpe_series is None:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.sharpe_raw - self.sharpe_series


def per_period_rate(quoted: Decimal, *, periods_per_year: Decimal,
                    compound: bool) -> Decimal:
    """Turn an annual rate into a per-period one, by a method you name.

    ``compound=True`` gives ``(1 + quoted) ** (1 / periods_per_year) - 1``,
    the rate that compounds to the quote over a year. ``compound=False``
    divides, which is what a money-market accrual convention actually does and
    what most quote sheets mean.

    **They are not interchangeable and there is no default.** Five per cent
    over 252 periods is 0.000198413 divided and 0.000193695 compounded; the
    difference is small per period, runs in one direction for the whole
    sample, and lands on the hurdle rather than on the return. Which one is
    right depends on how the rate you were handed was quoted, and this module
    cannot see that.
    """
    if periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year must be positive, got {periods_per_year}")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        n = Decimal(periods_per_year)
        q = Decimal(quoted)
        if not compound:
            return q / n
        if q <= -1:
            raise ArithmeticError(
                f"a quoted rate of {q} is at or below -100 per cent; the "
                "root that compounding needs is not defined there. Divide "
                "instead, with compound=False, or check the sign convention "
                "on the quote.")
        return (((_ONE + q).ln()) / n).exp() - _ONE


def rebase(rate: Decimal, *, quoted: DayCount, target: DayCount) -> Decimal:
    """Restate an annual rate from one day-count basis on to another.

    A rate quoted ACT/360 accrues ``rate / 360`` a day. Expressed on a 365-day
    year the same accrual is ``rate * 365 / 360``. Using the raw quote against
    a 365-day calendar therefore understates the hurdle by 1.39 per cent of
    itself, every period — small, one-directional, and on the flattering side.

    Neither basis is inferred. A rate arrives with a convention attached to
    where it came from, not to its value, and guessing it is how the error
    this function exists to stop gets made.
    """
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return Decimal(rate) * Decimal(target.days) / Decimal(quoted.days)


def excess_returns(returns: Sequence[Decimal],
                   rates: Sequence[Decimal]) -> List[Decimal]:
    """Period-by-period difference between a return and the rate it faced.

    Both are per-period figures over the same periods. The subtraction is
    arithmetic rather than geometric, which matches how every ratio in
    :mod:`mdnorm.metrics` treats an excess return; a geometric excess is a
    different quantity and belongs to the caller who wants it.
    """
    _check_pair(returns, rates, ("returns", "rates"))
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [Decimal(r) - Decimal(f) for r, f in zip(returns, rates)]


def active_returns(returns: Sequence[Decimal],
                   benchmark: Sequence[Decimal]) -> List[Decimal]:
    """Period-by-period difference between a return and a benchmark return."""
    _check_pair(returns, benchmark, ("returns", "benchmark"))
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [Decimal(r) - Decimal(b) for r, b in zip(returns, benchmark)]


def tracking_error(returns: Sequence[Decimal], benchmark: Sequence[Decimal],
                   *, ddof: int) -> Optional[Decimal]:
    """Standard deviation of the active return, per period.

    Not annualised, and ``ddof`` is required rather than assumed: the
    difference between dividing by ``n`` and by ``n - 1`` is visible on the
    short samples people compute tracking error over, which is most of them.
    ``None`` when the active series has no dispersion or is too short.
    """
    active = active_returns(returns, benchmark)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return _stdev(active, ddof=ddof)


def information_ratio(returns: Sequence[Decimal],
                      benchmark: Sequence[Decimal], *,
                      ddof: int) -> Optional[Decimal]:
    """Mean active return over its standard deviation, per period.

    ``None`` when the tracking error is zero or undefined. A strategy that
    tracks its benchmark exactly has no information ratio rather than an
    infinite one.
    """
    active = active_returns(returns, benchmark)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        sd = _stdev(active, ddof=ddof)
        if sd is None or sd == 0:
            return None
        return _mean(active) / sd


def _correlation(a: Sequence[Decimal], b: Sequence[Decimal]) -> Optional[Decimal]:
    ma, mb = _mean(a), _mean(b)
    sa = sum(((x - ma) * (x - ma) for x in a), _ZERO)
    sb = sum(((y - mb) * (y - mb) for y in b), _ZERO)
    if sa <= 0 or sb <= 0:
        return None
    cov = sum(((x - ma) * (y - mb) for x, y in zip(a, b)), _ZERO)
    return cov / (sa.sqrt() * sb.sqrt())


def excess_report(returns: Sequence[Decimal], rates: Sequence[Decimal],
                  *, ddof: int) -> ExcessReport:
    """Means and volatilities of the return, the rate and their difference."""
    excess = excess_returns(returns, rates)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        rs = [Decimal(r) for r in returns]
        fs = [Decimal(f) for f in rates]
        return ExcessReport(
            observations=len(rs),
            mean_return=_mean(rs),
            mean_rate=_mean(fs),
            mean_excess=_mean(excess),
            volatility_return=_stdev(rs, ddof=ddof),
            volatility_excess=_stdev(excess, ddof=ddof),
            rate_volatility=_stdev(fs, ddof=ddof),
        )


def active_report(returns: Sequence[Decimal], benchmark: Sequence[Decimal],
                  *, ddof: int) -> ActiveReport:
    """Means, tracking error and information ratio against a benchmark."""
    active = active_returns(returns, benchmark)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        rs = [Decimal(r) for r in returns]
        bs = [Decimal(b) for b in benchmark]
        te = _stdev(active, ddof=ddof)
        ir = None if te is None or te == 0 else _mean(active) / te
        return ActiveReport(
            observations=len(rs),
            mean_return=_mean(rs),
            mean_benchmark=_mean(bs),
            mean_active=_mean(active),
            tracking_error=te,
            information_ratio=ir,
        )


def hurdle_comparison(returns: Sequence[Decimal], rates: Sequence[Decimal],
                      *, ddof: int) -> HurdleComparison:
    """The same Sharpe ratio against no hurdle, a constant one and the series.

    The constant is the mean of the rate series supplied, which is the most
    favourable reading of what somebody using one number was trying to do. It
    is subtracted from the mean return and divided by the volatility of the
    *raw* returns, because that is what subtracting a constant does: it moves
    the centre and leaves the dispersion where it was.

    The series figure divides by the volatility of the excess series, which
    differs whenever the rate moved. :attr:`HurdleComparison.rate_correlation`
    is reported beside the two so the direction of the difference is legible
    from the sample rather than asserted here.
    """
    _check_pair(returns, rates, ("returns", "rates"))
    excess = excess_returns(returns, rates)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        rs = [Decimal(r) for r in returns]
        fs = [Decimal(f) for f in rates]
        mean_r, mean_f = _mean(rs), _mean(fs)
        sd_raw = _stdev(rs, ddof=ddof)
        sd_ex = _stdev(excess, ddof=ddof)
        return HurdleComparison(
            observations=len(rs),
            sharpe_raw=None if sd_raw is None else mean_r / sd_raw,
            sharpe_constant=(None if sd_raw is None
                             else (mean_r - mean_f) / sd_raw),
            sharpe_series=(None if sd_ex is None
                           else _mean(excess) / sd_ex),
            mean_rate=mean_f,
            rate_volatility=_stdev(fs, ddof=ddof),
            rate_correlation=_correlation(rs, fs),
        )
