"""Five hundred names is not five hundred bets.

:mod:`mdnorm.independence` counts how many independent observations a
overlapping-label study really has, along the time axis. This is the same
question asked across the cross-section, and it is the one nobody asks::

    from mdnorm import correlation_matrix, breadth_report

    m = correlation_matrix(returns_by_symbol)   # 500 names
    rep = breadth_report(m)
    rep.names               # 500
    rep.average_correlation # 0.41
    rep.effective_bets      # 12.4
    rep.overstatement       # 40.3x

**Breadth enters performance arithmetic as a square root, and it is almost
always the wrong number.** The fundamental law of active management puts the
information ratio at the information coefficient times the square root of the
number of independent bets. Counting positions instead of bets does not
overstate the ratio a little: five hundred correlated names behaving like
twelve overstate √(500/12.4), which is a factor of six and a third.

**The error is silent because the position count is a fact.** There really are
five hundred names, the trades really happened, the reconciliation really
balances. Nothing in the books is wrong. What is wrong is the claim implied by
reporting a t-statistic against five hundred, and no line of the accounting
contradicts it.

**Two numbers are reported, and they are not two estimates of one thing.**
:func:`effective_bets` is the participation ratio of the correlation matrix's
eigenvalues, ``(Σλ)² / Σλ²`` — how concentrated risk is across independent
directions, in the sense of Meucci. :func:`effective_observations` is
``n / (1 + (n-1)ρ̄)`` — what an average of ``n`` correlated series is worth as
a sample size, which is the number a cross-sectional t-statistic needs.

They coincide only at the two extremes: both give ``n`` for the identity and
both give one when every correlation is one. Everywhere in between they differ
on purpose, and by a lot. Three names at ρ = 0.5 are two bets and one and a
half observations; as ``n`` grows at fixed ρ the first tends to ``1/ρ²`` and
the second to ``1/ρ``. Quote the one that matches the claim being made, and
if you are not sure which claim you are making, that is the finding.

**A correlation is an estimate, and a short sample flatters it downward.**
With fewer observations than names the sample correlation matrix is singular
and its eigenvalues are partly noise, which inflates the apparent number of
bets. :attr:`BreadthReport.observations` carries the sample size so the
reader can see whether it supports the claim; nothing here corrects for it,
because the correction depends on a model of the return process and this
library does not have one.

Nothing in this module allocates, sizes or ranks anything. It reports how many
independent bets a correlation structure contains and stops.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import (TYPE_CHECKING, Dict, List, Mapping, Optional,
                    Sequence, Tuple)

if TYPE_CHECKING:  # pragma: no cover
    from .independence import EffectiveSample

__all__ = [
    "CorrelationMatrix",
    "BreadthReport",
    "correlation_matrix",
    "average_correlation",
    "eigenvalues",
    "effective_bets",
    "effective_observations",
    "breadth_report",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 40
# The rotation is run until the off-diagonal mass stops shrinking. It cannot
# be driven to zero: decimal arithmetic at a fixed precision has a round-off
# floor, and on a forty-by-forty matrix that floor sits near 1e-23. Demanding
# anything below it spins for a hundred sweeps and then reports a failure that
# is a property of the arithmetic rather than of the matrix.
_TARGET = Decimal("1e-30")          # stop early if we ever get this far
_ACCEPTABLE = Decimal("1e-15")      # a floor above this is a real failure
_NEGLIGIBLE = Decimal("1e-32")      # skip an element already this small
_MAX_SWEEPS = 100


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """A symmetric matrix of correlations, with the names it was built from."""

    names: Tuple[str, ...]
    rows: Tuple[Tuple[Decimal, ...], ...]
    observations: Optional[int] = None

    def __post_init__(self) -> None:
        n = len(self.names)
        if n < 2:
            raise ValueError(
                "a correlation matrix needs at least two names; the breadth "
                "of one position is one")
        if len(set(self.names)) != n:
            raise ValueError("a name is listed twice")
        if len(self.rows) != n or any(len(r) != n for r in self.rows):
            raise ValueError(f"the matrix is not {n} by {n}")
        for i in range(n):
            if self.rows[i][i] != _ONE:
                raise ValueError(
                    f"the diagonal must be exactly one; row {i} "
                    f"({self.names[i]!r}) has {self.rows[i][i]}. A covariance "
                    "matrix is not a correlation matrix and the difference "
                    "changes every number here")
            for j in range(i + 1, n):
                if self.rows[i][j] != self.rows[j][i]:
                    raise ValueError(
                        f"the matrix is not symmetric at ({i}, {j}): "
                        f"{self.rows[i][j]} against {self.rows[j][i]}")
                if not (-_ONE <= self.rows[i][j] <= _ONE):
                    raise ValueError(
                        f"a correlation is between -1 and 1; ({i}, {j}) is "
                        f"{self.rows[i][j]}")

    @property
    def size(self) -> int:
        return len(self.names)

    def __getitem__(self, pair: Tuple[int, int]) -> Decimal:
        i, j = pair
        return self.rows[i][j]


@dataclass(frozen=True, slots=True)
class BreadthReport:
    """How many independent bets a cross-section actually contains."""

    names: int
    observations: Optional[int]
    average_correlation: Decimal
    effective_bets: Decimal
    effective_observations: Decimal

    @property
    def overstatement(self) -> Optional[Decimal]:
        """Position count as a multiple of the effective bet count."""
        if self.effective_bets == 0:
            return None
        return Decimal(self.names) / self.effective_bets

    @property
    def ratio_overstatement(self) -> Optional[Decimal]:
        """How much an information ratio computed on the position count is
        overstated, which is the square root of :attr:`overstatement`.

        Breadth enters the fundamental law under a square root, so a fortyfold
        error in the count is a sixfold error in the ratio. Sixfold is still
        the difference between a fund and a story.
        """
        over = self.overstatement
        if over is None or over < 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(over).sqrt()

    @property
    def thin_sample(self) -> bool:
        """Whether there are fewer observations than names.

        When there are, the sample correlation matrix is singular, some of its
        eigenvalues are noise, and :attr:`effective_bets` is biased upward —
        in the flattering direction. This reports the condition and does not
        correct for it.
        """
        return self.observations is not None and self.observations < self.names

    def as_sample(self) -> "EffectiveSample":
        """The cross-sectional count, shaped for :mod:`mdnorm.independence`.

        Carries :attr:`effective_observations` rather than
        :attr:`effective_bets`, because a t-statistic on a cross-sectional
        average is asking the variance-of-the-mean question. Marked
        ``estimated`` because a correlation matrix is an estimate and the
        count inherits every weakness of the sample it came from.
        """
        from .independence import EffectiveSample

        return EffectiveSample(nominal=self.names,
                               effective=self.effective_observations,
                               estimated=True)

    @property
    def dominated_by_one_factor(self) -> bool:
        """Whether the cross-section behaves as though it has a single driver.

        True when fewer than two effective bets survive in ``n`` names. At
        that point the portfolio is one position wearing many tickers, and
        every diversification statement made about it is a statement about
        the naming convention.
        """
        return self.names >= 2 and self.effective_bets < 2


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, _ZERO) / len(values)


def correlation_matrix(
    columns: Mapping[str, Sequence[Decimal]]
) -> CorrelationMatrix:
    """Build a correlation matrix from aligned return series.

    Every series must already be on the same grid and the same length: this
    does no joining, because a join is a decision about what to do with
    missing observations and :mod:`mdnorm.align` is where that decision
    belongs. Ragged input raises rather than being truncated to the shortest,
    which would silently change which period the answer describes.

    A series with no variance raises. Its correlation with anything is a
    division by zero, and returning zero there would read as "uncorrelated"
    when the truth is "not a series".
    """
    if len(columns) < 2:
        raise ValueError("a correlation matrix needs at least two series")
    names = tuple(columns)
    lengths = {len(columns[k]) for k in names}
    if len(lengths) != 1:
        raise ValueError(
            "the series have different lengths "
            f"({sorted(lengths)}); align them first rather than letting this "
            "truncate to the shortest")
    n_obs = lengths.pop()
    if n_obs < 2:
        raise ValueError("a correlation needs at least two observations")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        centred: Dict[str, List[Decimal]] = {}
        norms: Dict[str, Decimal] = {}
        for k in names:
            vals = [Decimal(v) for v in columns[k]]
            mu = _mean(vals)
            dev = [v - mu for v in vals]
            ss = sum((d * d for d in dev), _ZERO)
            if ss == 0:
                raise ValueError(
                    f"series {k!r} does not vary, so it has no correlation "
                    "with anything; a constant is not a return series")
            centred[k] = dev
            norms[k] = ss.sqrt()

        rows: List[Tuple[Decimal, ...]] = []
        for a in names:
            row: List[Decimal] = []
            for b in names:
                if a == b:
                    row.append(_ONE)
                    continue
                dot = sum((x * y for x, y in zip(centred[a], centred[b])),
                          _ZERO)
                r = dot / (norms[a] * norms[b])
                # clamp only the last-digit overshoot arithmetic can produce
                r = max(-_ONE, min(_ONE, r))
                row.append(r)
            rows.append(tuple(row))

    # symmetry can be lost in the last digit; take the upper triangle as given
    fixed = [list(r) for r in rows]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            fixed[j][i] = fixed[i][j]
    return CorrelationMatrix(names=names,
                             rows=tuple(tuple(r) for r in fixed),
                             observations=n_obs)


def average_correlation(matrix: CorrelationMatrix) -> Decimal:
    """The mean of the off-diagonal correlations.

    The diagonal is excluded because a name's correlation with itself is an
    identity rather than an observation, and including it would pull every
    average toward one by exactly the amount that hides the problem.
    """
    n = matrix.size
    total = _ZERO
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += matrix[i, j]
            count += 1
    return total / count


def eigenvalues(matrix: CorrelationMatrix) -> List[Decimal]:
    """The eigenvalues of the matrix, largest first.

    A cyclic Jacobi rotation, which needs nothing but arithmetic and a square
    root, so this stays a pure-Python library with exact decimal inputs.

    The sweep stops when the off-diagonal mass reaches the round-off floor of
    the working precision and stops shrinking, which is the only sensible
    stopping rule for fixed-precision arithmetic. If the floor it reaches is
    still large enough to matter, this raises rather than returning a
    half-diagonalised answer dressed up as eigenvalues.
    """
    n = matrix.size
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        a = [[Decimal(matrix[i, j]) for j in range(n)] for i in range(n)]
        off: Optional[Decimal] = None
        previous: Optional[Decimal] = None
        for _ in range(_MAX_SWEEPS):
            off = _ZERO
            for i in range(n):
                for j in range(i + 1, n):
                    off += abs(a[i][j])
            if off < _TARGET:
                break
            if previous is not None and off >= previous:
                break               # the round-off floor; no sweep will help
            previous = off
            for p in range(n):
                for q in range(p + 1, n):
                    if abs(a[p][q]) < _NEGLIGIBLE:
                        continue
                    theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
                    root = (theta * theta + _ONE).sqrt()
                    denom = theta + root if theta >= 0 else theta - root
                    t = _ONE / denom
                    c = _ONE / (t * t + _ONE).sqrt()
                    s = t * c
                    for k in range(n):
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = c * akp - s * akq
                        a[k][q] = s * akp + c * akq
                    for k in range(n):
                        apk, aqk = a[p][k], a[q][k]
                        a[p][k] = c * apk - s * aqk
                        a[q][k] = s * apk + c * aqk
        if off is None or off > _ACCEPTABLE:
            raise ArithmeticError(
                f"the rotation stalled with {off} of off-diagonal mass left "
                f"after {_MAX_SWEEPS} sweeps, which is too much to call these "
                "eigenvalues; the matrix is reported as it stands rather "
                "than guessed at")
        return sorted((a[i][i] for i in range(n)), reverse=True)


def effective_bets(matrix: CorrelationMatrix) -> Decimal:
    """The participation ratio of the eigenvalues, ``(Σλ)² / Σλ²``.

    One for a matrix of ones — every name the same bet — and ``n`` for the
    identity. Uses the whole dependence structure rather than summarising it
    with an average, which is why it is the one to quote when the two
    estimates disagree.

    Sampling noise can make a sample eigenvalue slightly negative. Those are
    used as they are rather than clamped to zero: a clamp would quietly repair
    a matrix that is telling you it was estimated on too little data.
    """
    lam = eigenvalues(matrix)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        total = sum(lam, _ZERO)
        square = sum((v * v for v in lam), _ZERO)
        if square == 0:
            raise ArithmeticError(
                "every eigenvalue is zero, so the matrix carries no structure "
                "to count bets in")
        return (total * total) / square


def effective_observations(matrix: CorrelationMatrix) -> Decimal:
    """``n / (1 + (n-1)ρ̄)``: what an average of ``n`` correlated series is
    worth as a sample size.

    This is the variance-of-the-mean sense of independence, and it is the
    number a cross-sectional t-statistic needs — hand it to
    :func:`mdnorm.deflate_t_stat` the way the time-axis count from
    :mod:`mdnorm.independence` is handed to it.

    It is not a second opinion on :func:`effective_bets` and the two are not
    interchangeable. It assumes one average correlation governs every pair,
    which no real cross-section obeys; the assumption is stated here rather
    than buried because the formula is the one most people already carry in
    their heads.

    A sufficiently negative average correlation drives the denominator to zero
    or below, at which point the formula stops meaning anything and this
    raises rather than returning a large or negative breadth.
    """
    n = matrix.size
    rho = average_correlation(matrix)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        denom = _ONE + (Decimal(n) - _ONE) * rho
        if denom <= 0:
            raise ArithmeticError(
                f"the average correlation of {rho} makes the equicorrelated "
                "denominator non-positive, so this formula has no meaning "
                "here; read effective_bets instead")
        return Decimal(n) / denom


def breadth_report(matrix: CorrelationMatrix) -> BreadthReport:
    """Both estimates, the average, and what the position count overstates."""
    return BreadthReport(
        names=matrix.size,
        observations=matrix.observations,
        average_correlation=average_correlation(matrix),
        effective_bets=effective_bets(matrix),
        effective_observations=effective_observations(matrix),
    )
