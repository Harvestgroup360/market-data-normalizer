"""Alpha is what is left after the things you already knew about.

A strategy with a Sharpe ratio worth reporting is sometimes a strategy, and
sometimes it is a factor everybody can buy, wearing a new name. Nothing in the
return series distinguishes them, because the return series of an exposure and
the return series of an edge look identical::

    from mdnorm import factor_regression, dominant_factor

    rep = factor_regression(strategy, {"market": mkt, "momentum": mom,
                                       "value": val})
    rep.r_squared          # 0.84
    rep.alpha              # 0.00002 per period — what no factor carried
    rep.alpha_t_stat       # 0.31
    rep.alpha_share        # 0.04 — four per cent of the mean return survives
    dominant_factor(rep)   # momentum, carrying 0.00038 of the mean

**The question is not whether the strategy made money.** It did; that is in
the accounts. The question is whether it made money for a reason the person
reading the report is being asked to pay for, and the honest form of that
question is how much of the mean return survives once the known exposures are
subtracted.

**The absence of an exposure is not evidence of alpha.** It is evidence about
your factor list. A residual that no factor explains means the strategy is
orthogonal to *the factors you supplied*, which is a much smaller claim than
the one people make with it. This module will never tell you a strategy has
alpha; it tells you how much survives a list you chose.

**No factor data ships with this library and none ever will.** Bundling a
factor set would make every answer partly a property of whose definition of
momentum we happened to vendor, and :doc:`ROADMAP` has already ruled out
tying the library to one feed. Bring your own series, state where they came
from, and record it — :mod:`mdnorm.provenance` exists for exactly this.

**Choosing a factor list after seeing the strategy is a search.** Four
candidate factors used in every combination are fifteen regressions, and the
one with the flattering residual is the best of fifteen. That count belongs in
a deflation: :mod:`mdnorm.multiverse` will enumerate the grid and hand you the
number.

**The loadings are full-sample and constant.** A strategy whose market
exposure was one in the first half and zero in the second has an average beta
of a half, which describes neither half. Nothing here detects that, because
detecting it means choosing a breakpoint and a breakpoint is a parameter —
run the regression over :mod:`mdnorm.windows` instead and look at whether the
betas move.

**One trap worth naming, because it is in the arithmetic rather than in the
data.** A least-squares residual computed with an intercept has a mean of
exactly zero, always. A Sharpe ratio on it is therefore zero whatever the
alpha was. :func:`residuals` returns that series, for looking at the shape of
what the factors missed; :func:`alpha_stream` returns the strategy with the
factor contributions removed and the intercept kept, whose mean *is* the
alpha, and that is the series to put a Sharpe ratio on.

The t-statistics assume the residuals are independent. When the strategy is
built on overlapping labels they are not, and :mod:`mdnorm.independence`
applies to these statistics exactly as it applies to any other.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FactorLoading",
    "ExposureReport",
    "factor_regression",
    "residuals",
    "alpha_stream",
    "dominant_factor",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 50
# Below this a pivot is indistinguishable from zero at the working precision,
# which for a normal-equations matrix means the columns are collinear.
_SINGULAR = Decimal("1e-30")


@dataclass(frozen=True, slots=True)
class FactorLoading:
    """One factor, its beta, and the part of the mean return it carries."""

    name: str
    beta: Decimal
    standard_error: Optional[Decimal]
    contribution: Decimal

    @property
    def t_stat(self) -> Optional[Decimal]:
        """Beta over its standard error.

        ``None`` when the standard error could not be formed — a regression
        with no residual degrees of freedom fits every point exactly and says
        nothing about any of them.
        """
        if self.standard_error is None or self.standard_error == 0:
            return None
        return self.beta / self.standard_error


@dataclass(frozen=True, slots=True)
class ExposureReport:
    """What survives a strategy's return once known exposures are subtracted."""

    observations: int
    loadings: Tuple[FactorLoading, ...]
    alpha: Decimal
    alpha_standard_error: Optional[Decimal]
    mean_return: Decimal
    r_squared: Decimal
    residual_variance: Decimal

    @property
    def alpha_t_stat(self) -> Optional[Decimal]:
        if self.alpha_standard_error is None or self.alpha_standard_error == 0:
            return None
        return self.alpha / self.alpha_standard_error

    @property
    def alpha_share(self) -> Optional[Decimal]:
        """The intercept as a share of the mean return.

        ``None`` when the mean return is zero: a share of nothing is a
        sentence with no meaning, and reporting one would invite a reader to
        divide by it.

        The figure can exceed one or go negative — that happens when the
        factors carry the return the other way and the intercept is making up
        the difference. Read it beside :attr:`r_squared` rather than alone.
        """
        if self.mean_return == 0:
            return None
        return self.alpha / self.mean_return

    @property
    def explained_share(self) -> Optional[Decimal]:
        """The share of the mean return the factors account for."""
        share = self.alpha_share
        return None if share is None else _ONE - share

    @property
    def factors(self) -> int:
        return len(self.loadings)

    @property
    def degrees_of_freedom(self) -> int:
        """Observations less the intercept and one per factor."""
        return self.observations - self.factors - 1

    @property
    def loading_map(self) -> "dict[str, Decimal]":
        return {l.name: l.beta for l in self.loadings}

    @property
    def crowded(self) -> bool:
        """Whether there are fewer than ten observations per estimated term.

        A regression with four factors on forty points has ten degrees of
        freedom and an R² that means very little. This reports the condition;
        it does not adjust anything, because the adjustment people reach for
        next is an adjusted R² and that hides the problem behind a smaller
        number rather than naming it.
        """
        return self.observations < 10 * (self.factors + 1)


def _invert(matrix: List[List[Decimal]], labels: Sequence[str]) -> List[List[Decimal]]:
    """Gauss-Jordan inverse with partial pivoting.

    Raises on a singular matrix and names the term whose column collapsed,
    because "singular matrix" tells a caller nothing they can act on and
    "these two factors are the same series" tells them everything.
    """
    n = len(matrix)
    aug = [row[:] + [(_ONE if i == j else _ZERO) for j in range(n)]
           for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < _SINGULAR:
            raise ValueError(
                f"the design matrix is singular at {labels[col]!r}: that "
                "column is a linear combination of the others, so its "
                "coefficient is not identified. Two factors that are the same "
                "series, or one that is the sum of two more, will do this")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            aug[r] = [v - factor * w for v, w in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def _prepare(
    strategy: Sequence[Decimal], factors: Mapping[str, Sequence[Decimal]]
) -> Tuple[List[str], List[List[Decimal]], List[Decimal]]:
    if not factors:
        raise ValueError(
            "a regression needs at least one factor; with none the residual "
            "is the return and the answer is already in front of you")
    names = list(factors)
    n = len(strategy)
    if n == 0:
        raise ValueError("the strategy has no observations")
    ragged = {k: len(v) for k, v in factors.items() if len(v) != n}
    if ragged:
        raise ValueError(
            f"the strategy has {n} observations and "
            f"{', '.join(f'{k} has {v}' for k, v in sorted(ragged.items()))}; "
            "align the series first rather than letting this truncate them, "
            "which would change which period the answer describes")
    if n < len(names) + 2:
        raise ValueError(
            f"{n} observations cannot support {len(names)} factors and an "
            "intercept; the fit would be exact and would mean nothing")
    for k in names:
        column = list(factors[k])
        if len(set(column)) == 1:
            raise ValueError(
                f"factor {k!r} is constant, so it is the intercept under "
                "another name and its coefficient is not identified")
    return names, [[Decimal(v) for v in factors[k]] for k in names], \
        [Decimal(v) for v in strategy]


def factor_regression(
    strategy: Sequence[Decimal], factors: Mapping[str, Sequence[Decimal]]
) -> ExposureReport:
    """Regress a strategy's returns on a set of factor returns.

    Ordinary least squares with an intercept, solved through the normal
    equations in :class:`~decimal.Decimal` at fifty digits, which keeps this a
    library with no runtime dependencies. The intercept is the alpha: the mean
    return that the supplied factors do not account for.

    Every series must already be on the same grid and the same length. This
    does no joining, because a join is a decision about missing observations
    and :mod:`mdnorm.align` is where that decision belongs.
    """
    names, columns, y = _prepare(strategy, factors)
    k = len(names)
    n = len(y)

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        # design matrix with the intercept first
        design = [[_ONE] + [columns[j][i] for j in range(k)] for i in range(n)]
        width = k + 1
        labels = ["intercept"] + names

        xtx = [[sum((design[i][a] * design[i][b] for i in range(n)), _ZERO)
                for b in range(width)] for a in range(width)]
        xty = [sum((design[i][a] * y[i] for i in range(n)), _ZERO)
               for a in range(width)]
        inverse = _invert(xtx, labels)
        beta = [sum((inverse[a][b] * xty[b] for b in range(width)), _ZERO)
                for a in range(width)]

        fitted = [sum((design[i][a] * beta[a] for a in range(width)), _ZERO)
                  for i in range(n)]
        resid = [y[i] - fitted[i] for i in range(n)]
        rss = sum((r * r for r in resid), _ZERO)

        mean_y = sum(y, _ZERO) / n
        tss = sum(((v - mean_y) ** 2 for v in y), _ZERO)
        r_squared = _ZERO if tss == 0 else _ONE - rss / tss

        dof = n - width
        if dof > 0:
            sigma2 = rss / dof
            errors: List[Optional[Decimal]] = []
            for a in range(width):
                var = sigma2 * inverse[a][a]
                errors.append(var.sqrt() if var > 0 else None)
        else:
            sigma2 = _ZERO
            errors = [None] * width

        loadings = tuple(
            FactorLoading(
                name=names[j],
                beta=beta[j + 1],
                standard_error=errors[j + 1],
                contribution=beta[j + 1] * (sum(columns[j], _ZERO) / n),
            )
            for j in range(k))

        return ExposureReport(
            observations=n,
            loadings=loadings,
            alpha=beta[0],
            alpha_standard_error=errors[0],
            mean_return=mean_y,
            r_squared=r_squared,
            residual_variance=sigma2,
        )


def _fit(
    strategy: Sequence[Decimal], factors: Mapping[str, Sequence[Decimal]]
) -> Tuple[List[Decimal], List[List[Decimal]], List[Decimal], int]:
    """Shared solve: returns the coefficients, the design, y and its width."""
    names, columns, y = _prepare(strategy, factors)
    k, n = len(names), len(y)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        design = [[_ONE] + [columns[j][i] for j in range(k)] for i in range(n)]
        width = k + 1
        xtx = [[sum((design[i][a] * design[i][b] for i in range(n)), _ZERO)
                for b in range(width)] for a in range(width)]
        xty = [sum((design[i][a] * y[i] for i in range(n)), _ZERO)
               for a in range(width)]
        inverse = _invert(xtx, ["intercept"] + names)
        beta = [sum((inverse[a][b] * xty[b] for b in range(width)), _ZERO)
                for a in range(width)]
        return beta, design, y, width


def residuals(
    strategy: Sequence[Decimal], factors: Mapping[str, Sequence[Decimal]]
) -> List[Decimal]:
    """The ordinary least-squares residual series.

    **Its mean is zero by construction**, because the fit includes an
    intercept and the intercept absorbs exactly that. So this is the series to
    look at for the *shape* of what the factors missed — its volatility, its
    drawdowns, whether the misses cluster in one year — and it is emphatically
    not the series to compute a Sharpe ratio on, which would be a division of
    zero by something and would read as "no alpha" whatever the alpha was.

    For a series whose mean *is* the alpha, use :func:`alpha_stream`.
    """
    beta, design, y, width = _fit(strategy, factors)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [y[i] - sum((design[i][a] * beta[a] for a in range(width)),
                           _ZERO) for i in range(len(y))]


def alpha_stream(
    strategy: Sequence[Decimal], factors: Mapping[str, Sequence[Decimal]]
) -> List[Decimal]:
    """The strategy with the factor contributions removed and the intercept
    kept: the return that would have been earned by a version of this strategy
    holding no exposure to any of the supplied factors.

    Its mean is the alpha, so this is the series a Sharpe ratio, a drawdown or
    a :mod:`mdnorm.windows` sweep belongs on. Running those on the raw
    strategy answers a question about the factors; running them on this
    answers the question the report is claiming to answer.
    """
    beta, design, y, width = _fit(strategy, factors)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return [y[i] - sum((design[i][a] * beta[a] for a in range(1, width)),
                           _ZERO) for i in range(len(y))]


def dominant_factor(report: ExposureReport) -> Optional[FactorLoading]:
    """The factor carrying the largest part of the mean return, by size.

    Ranked by :attr:`FactorLoading.contribution` rather than by beta or by
    t-statistic: a large loading on a factor that went nowhere carries no
    return, and a significant coefficient is a statement about precision
    rather than about magnitude. ``None`` when there are no factors, or when
    every contribution is exactly zero.
    """
    if not report.loadings:
        return None
    best = max(report.loadings, key=lambda l: abs(l.contribution))
    return None if best.contribution == 0 else best
