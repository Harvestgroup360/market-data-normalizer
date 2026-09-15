"""The average of your returns is not the return you earned.

An average monthly return of one per cent does not annualise to 12.68 per
cent, which is what compounding the average gives. It annualises to what the
account did, and the difference is not a rounding error — it is the
volatility, converted into a number nobody reports::

    from mdnorm import compound_report, Convention

    rep = compound_report(monthly, convention=Convention.SIMPLE)
    rep.arithmetic            # 0.010000 — the number in the deck
    rep.geometric             # 0.009529 — the rate that actually compounds
    rep.drag                  # 0.000471 per month

    year = rep.annualised(periods_per_year=12)
    year.naive                # 0.126825 — the average, compounded
    year.actual               # 0.120539 — what the account did
    year.overstatement        # 0.006287 — sixty-three basis points

**The direction is guaranteed and the size is not.** The geometric mean never
exceeds the arithmetic one — that is an inequality, not a tendency — so
compounding the average always overstates. How much depends on the variance
alone.

**The gap is the variance, so the flattery grows with the risk.** To second
order the difference between the arithmetic and geometric means is σ²/2. A
strategy running at ten per cent annual volatility loses about half a point a
year to it; one running at forty per cent loses about eight. The riskier the
book, the more the reported average overstates what an investor received, which
is the wrong way round for a statistic to behave.

**Leverage is worse than proportional.** Doubling every period return roughly
quadruples the drag, because the variance term is squared while the mean is
only doubled. :func:`leverage_drag` applies the multiple and recomputes rather
than scaling the approximation, so the figure it reports is the arithmetic
rather than the rule of thumb.

**The approximation is reported beside the exact figure**, because σ²/2 is what
people quote and the two separate as soon as returns stop being small. Seeing
them side by side is the point: on monthly equity returns they agree to a basis
point, and on a series with twenty per cent periods they do not.

**Nothing here guesses your convention.** A log return and a simple return are
different numbers living in identically shaped files, and compounding one as
though it were the other is a silent error of exactly the kind this library
exists to catch. State :class:`Convention` on every call.

There is no default annualisation factor either, for the reason
:doc:`ROADMAP` already gives: the same series of minute bars annualises to two
different numbers on a 24/7 venue and a six-hour session, and both look
reasonable.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum
from typing import List, Optional, Sequence

__all__ = [
    "Convention",
    "Annualised",
    "CompoundReport",
    "arithmetic_mean",
    "geometric_mean",
    "compound",
    "naive_total",
    "variance_drag",
    "approximate_drag",
    "annualise_return",
    "to_log",
    "to_simple",
    "leverage_drag",
    "compound_report",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_PRECISION = 40


class Convention(str, Enum):
    """Which kind of return the series holds.

    ``SIMPLE`` is ``P_t / P_{t-1} - 1`` and compounds by multiplication.
    ``LOG`` is ``ln(P_t / P_{t-1})`` and compounds by addition. There is no
    default: the two are different numbers in identically shaped files.
    """

    SIMPLE = "simple"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class Annualised:
    """A per-period rate carried out to a year, two ways."""

    periods_per_year: int
    naive: Decimal
    actual: Decimal

    @property
    def overstatement(self) -> Decimal:
        """Naive less actual. Never negative: the geometric mean never exceeds
        the arithmetic one, and raising both to the same positive power keeps
        the order."""
        return self.naive - self.actual

    @property
    def overstatement_share(self) -> Optional[Decimal]:
        """The overstatement as a fraction of what was actually earned.

        ``None`` when the actual return is zero, which is not a failure: a
        year that netted to nothing has no total for an error to be a share
        of, and a number there would invite a reader to divide by it.
        """
        if self.actual == 0:
            return None
        return self.overstatement / self.actual


@dataclass(frozen=True, slots=True)
class CompoundReport:
    """What a series averaged, what it compounded to, and the gap."""

    observations: int
    convention: Convention
    arithmetic: Decimal
    geometric: Decimal
    volatility: Decimal
    actual_total: Decimal
    naive_total: Decimal

    @property
    def drag(self) -> Decimal:
        """Arithmetic mean less geometric mean, per period. Never negative."""
        return self.arithmetic - self.geometric

    @property
    def approximate_drag(self) -> Decimal:
        """The σ²/2 rule of thumb, for comparison with :attr:`drag`.

        Quoted here rather than used. It is a second-order expansion, so it
        agrees with the exact figure on small returns and parts company on
        large ones — and the size of the disagreement is itself worth seeing,
        since the approximation is what most reports are built on.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.volatility * self.volatility / _TWO

    @property
    def total_gap(self) -> Decimal:
        """Actual total less the sum of the returns.

        **The sign is not fixed, and that is worth knowing before quoting it.**
        Compounding adds every cross-product of the period returns, so on a
        series with a positive mean and modest volatility the account beats the
        sum, and on a volatile or losing one it trails. Only the comparison in
        :meth:`annualised` has a guaranteed direction. Reporting this one as
        "the overstatement" — which an earlier draft of this module did — is
        wrong about half the time.
        """
        return self.actual_total - self.naive_total

    def annualised(self, *, periods_per_year: int) -> Annualised:
        """Both means carried out to a year: the flattering one and the true one.

        ``naive`` compounds the arithmetic mean, which is the number most
        reports are built from. ``actual`` compounds the geometric mean, which
        is what the account did. The first is never smaller than the second.
        """
        return Annualised(
            periods_per_year=periods_per_year,
            naive=annualise_return(self.arithmetic,
                                   periods_per_year=periods_per_year),
            actual=annualise_return(self.geometric,
                                    periods_per_year=periods_per_year),
        )


def _check(values: Sequence[Decimal], convention: Convention) -> List[Decimal]:
    if not values:
        raise ValueError("a return series needs at least one observation")
    out = [Decimal(v) for v in values]
    if convention is Convention.SIMPLE:
        for i, v in enumerate(out):
            if v < -_ONE:
                raise ValueError(
                    f"observation {i} is {v}, which as a simple return means "
                    "losing more than everything. Either the series is log "
                    "returns — say so with Convention.LOG — or it is wrong, "
                    "and compounding it would produce a negative account "
                    "balance. Exactly -1 is allowed: an account can reach "
                    "zero")
    return out


def arithmetic_mean(values: Sequence[Decimal]) -> Decimal:
    """The plain average. The number that goes in the deck."""
    if not values:
        raise ValueError("a return series needs at least one observation")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return sum((Decimal(v) for v in values), _ZERO) / len(values)


def compound(values: Sequence[Decimal], *, convention: Convention) -> Decimal:
    """The total return the series actually produced.

    Multiplication for simple returns, addition then exponentiation for log
    returns. The result is a simple total either way, because that is the
    number an account statement shows.
    """
    vals = _check(values, convention)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        if convention is Convention.LOG:
            return sum(vals, _ZERO).exp() - _ONE
        growth = _ONE
        for v in vals:
            growth *= (_ONE + v)
        return growth - _ONE


def geometric_mean(
    values: Sequence[Decimal], *, convention: Convention
) -> Decimal:
    """The per-period simple rate that compounds to the same total.

    Reported as a simple rate whichever convention went in, so the two are
    comparable against each other and against the arithmetic mean.
    """
    vals = _check(values, convention)
    n = len(vals)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        if convention is Convention.LOG:
            return (sum(vals, _ZERO) / n).exp() - _ONE
        growth = _ONE
        for v in vals:
            growth *= (_ONE + v)
        if growth == 0:
            return -_ONE           # everything was lost; the rate is -100%
        return growth ** (_ONE / Decimal(n)) - _ONE


def naive_total(values: Sequence[Decimal]) -> Decimal:
    """The sum of the returns: what multiplying the average by ``n`` gives.

    It is not what the account did, but it does not err in a fixed direction
    either — compounding adds the cross-products, which help a positive series
    and hurt a volatile one. For the comparison that does have a guaranteed
    sign, see :meth:`CompoundReport.annualised`.
    """
    if not values:
        raise ValueError("a return series needs at least one observation")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return sum((Decimal(v) for v in values), _ZERO)


def _pstdev(values: Sequence[Decimal]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        n = len(values)
        mu = sum(values, _ZERO) / n
        var = sum(((v - mu) ** 2 for v in values), _ZERO) / n
        return var.sqrt()


def variance_drag(
    values: Sequence[Decimal], *, convention: Convention
) -> Decimal:
    """Arithmetic mean less geometric mean, computed exactly.

    Not the σ²/2 approximation — see :attr:`CompoundReport.approximate_drag`
    for that, reported beside this so the gap between them is visible.
    """
    return (arithmetic_mean(values)
            - geometric_mean(values, convention=convention))


def approximate_drag(values: Sequence[Decimal]) -> Decimal:
    """The σ²/2 rule of thumb on its own."""
    if not values:
        raise ValueError("a return series needs at least one observation")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        sigma = _pstdev([Decimal(v) for v in values])
        return sigma * sigma / _TWO


def annualise_return(
    per_period: Decimal, *, periods_per_year: int
) -> Decimal:
    """Compound a per-period simple rate into an annual one.

    ``(1 + r) ** periods - 1``, not ``r * periods``. There is no default for
    ``periods_per_year`` and there will not be one: the same series of minute
    bars annualises to two different numbers on a 24/7 venue and a six-hour
    session, and both look reasonable.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    if per_period <= -_ONE:
        raise ValueError(
            "a per-period rate at or below -1 cannot be compounded; the "
            "account is already empty")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return (_ONE + Decimal(per_period)) ** Decimal(periods_per_year) - _ONE


def to_log(values: Sequence[Decimal]) -> List[Decimal]:
    """Convert simple returns to log returns."""
    vals = _check(values, Convention.SIMPLE)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [(_ONE + v).ln() for v in vals]


def to_simple(values: Sequence[Decimal]) -> List[Decimal]:
    """Convert log returns to simple returns."""
    if not values:
        raise ValueError("a return series needs at least one observation")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [Decimal(v).exp() - _ONE for v in values]


def leverage_drag(
    values: Sequence[Decimal], *, multiple: Decimal, convention: Convention
) -> Decimal:
    """The drag on a series with every period return multiplied by ``multiple``.

    The multiple is applied and the drag recomputed, rather than the
    approximation being scaled: this is the arithmetic, not the rule of thumb.
    Expect roughly ``multiple ** 2`` times the unlevered figure, because the
    variance term is squared while the mean is only multiplied — which is why
    a levered product decays faster than its exposure suggests, and why the
    decay is not visible in the average.

    Only the returns are scaled. Financing, borrow and the path-dependence of a
    daily reset are real and are not modelled here; this is a lower bound on
    what leverage costs, not an estimate of it.
    """
    if multiple <= 0:
        raise ValueError("a leverage multiple is positive")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        scaled = [Decimal(multiple) * Decimal(v) for v in values]
    return variance_drag(scaled, convention=convention)


def compound_report(
    values: Sequence[Decimal], *, convention: Convention
) -> CompoundReport:
    """Everything at once: the two means, the two totals, and the gap."""
    vals = _check(values, convention)
    return CompoundReport(
        observations=len(vals),
        convention=convention,
        arithmetic=arithmetic_mean(vals),
        geometric=geometric_mean(vals, convention=convention),
        volatility=_pstdev(vals),
        actual_total=compound(vals, convention=convention),
        naive_total=naive_total(vals),
    )
