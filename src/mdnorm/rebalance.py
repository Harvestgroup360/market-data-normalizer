"""How often a backtest rebalances is an assumption, and it is never stated.

A weight vector is a decision made once. What happens to it afterwards is
arithmetic: winners grow, losers shrink, and by the end of a month an equal-
weight book is not equal-weight any more. Most backtests quietly snap the
weights back to target every period, which earns a return nobody could have
had without trading, and then report no turnover at all::

    from mdnorm import periodic_rebalance, buy_and_hold, compare_schedules

    daily   = periodic_rebalance(target, returns, every=1)
    monthly = periodic_rebalance(target, returns, every=21)
    never   = buy_and_hold(target, returns)

    compare_schedules({"daily": daily, "monthly": monthly, "never": never})

**The frequency is a free parameter that moves the answer.** Nothing in the
data changes between those three lines. The strategy, the instruments and the
returns are identical; only how often the book was assumed to be traded back
to target differs, and both the return and the trading it implies move with it.

**Turnover is the part that goes missing.** This module reports the turnover
each schedule requires at every rebalance and in total, because a schedule
without its turnover is half a description.
:class:`ScheduleComparison.breakeven_cost` turns that into the number worth
arguing about: the cost per unit of turnover at which the more active schedule
stops being the better one. Above it the ranking flips.

**No costs are applied here.** :mod:`mdnorm.costs` prices a trade and this
module says how much trading a rule implies; keeping them apart means the cost
model stays something the caller states rather than something a schedule
smuggled in.

**The residual is cash and it earns nothing.** Weights need not sum to one.
Whatever is left over — ``1 - sum(weights)`` — is carried at a zero return, so
a book that is not fully invested is not credited with a rate it was not paid.
:mod:`mdnorm.hurdle` is where that assumption gets priced.

There is no default schedule, no default band and no default turnover
convention. A library that picked one would be answering a question the caller
has not asked yet, and the frequency is exactly the parameter this module
exists to make visible.

Arithmetic runs at forty significant digits, matching the other evaluation
modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DriftReport",
    "ScheduleResult",
    "ScheduleComparison",
    "drift_once",
    "drift_path",
    "turnover_between",
    "periodic_rebalance",
    "band_rebalance",
    "buy_and_hold",
    "drift_report",
    "compare_schedules",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 40

Weights = Mapping[str, Decimal]
Returns = Sequence[Mapping[str, Decimal]]


def _check(weights: Weights, returns: Returns) -> None:
    if not weights:
        raise ValueError("a portfolio needs at least one instrument")
    if not returns:
        raise ValueError("a schedule needs at least one period of returns")
    for i, period in enumerate(returns):
        missing = [k for k in weights if k not in period]
        if missing:
            raise ValueError(
                f"period {i} has no return for {', '.join(sorted(missing))}. "
                "A missing return is not a zero return: a name that did not "
                "trade and a name that was flat produce the same weight here "
                "and mean different things. Fill the gap deliberately, or use "
                "mdnorm.coverage to see how many there are.")


def drift_once(weights: Weights,
               period: Mapping[str, Decimal]) -> Tuple[Dict[str, Decimal],
                                                       Decimal]:
    """Apply one period of returns and report the weights that come out.

    Returns the new weight mapping and the portfolio return for the period.
    The portfolio return is ``sum(w_i * r_i)``; the residual ``1 - sum(w)`` is
    cash and contributes nothing, which is an assumption rather than a fact
    and is stated in the module docstring.

    A period that takes the book to zero or below raises rather than returning
    weights, because there is no allocation after the equity is gone and a
    percentage of nothing is not one.
    """
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        pnl = sum((Decimal(w) * Decimal(period[k])
                   for k, w in weights.items()), _ZERO)
        equity = _ONE + pnl
        if equity <= 0:
            raise ArithmeticError(
                f"the period return of {pnl} leaves equity at {equity}; there "
                "are no weights after that. Check the sign convention on the "
                "returns, or whether these are simple returns rather than log "
                "ones — mdnorm.to_simple converts.")
        out = {k: Decimal(w) * (_ONE + Decimal(period[k])) / equity
               for k, w in weights.items()}
        return out, pnl


def drift_path(weights: Weights, returns: Returns) -> List[Dict[str, Decimal]]:
    """Start-of-period weights for every period, with no rebalancing at all.

    The first entry is the weights supplied. Each later entry is what the
    previous one became. This is the honest counterfactual to every schedule
    below: what the book held if nobody touched it.
    """
    _check(weights, returns)
    current = {k: Decimal(v) for k, v in weights.items()}
    path = [dict(current)]
    for period in returns[:-1]:
        current, _ = drift_once(current, period)
        path.append(dict(current))
    return path


def turnover_between(before: Weights, after: Weights, *,
                     one_sided: bool) -> Decimal:
    """Trading implied by moving from one allocation to another.

    ``one_sided=True`` is half the sum of absolute changes, which matches
    :func:`mdnorm.metrics.turnover` and is what most turnover figures mean.
    ``one_sided=False`` is the full sum, which is what a cost model wants,
    because both the sale and the purchase pay. Required rather than
    defaulted: the two differ by a factor of two and both appear in print
    under the same word.

    A name in one allocation and not the other counts at a zero weight in the
    other, so entering and leaving the book both register.
    """
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        names = set(before) | set(after)
        total = sum((abs(Decimal(after.get(k, _ZERO))
                         - Decimal(before.get(k, _ZERO)))
                     for k in names), _ZERO)
        return total / 2 if one_sided else total


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """What a rebalancing rule produced: returns, weights and trading.

    ``turnover`` has one entry per period, ``None`` where no rebalance
    happened. Period zero is ``None`` as well: the initial purchase is not a
    rebalance, and counting it as one puts a spike at the start of every
    turnover series.
    """

    label: str
    periods: int
    returns: Tuple[Decimal, ...]
    weights: Tuple[Dict[str, Decimal], ...]
    turnover: Tuple[Optional[Decimal], ...]
    rebalances: int

    @property
    def total_return(self) -> Decimal:
        """Compounded return over the whole path."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            acc = _ONE
            for r in self.returns:
                acc *= _ONE + r
            return acc - _ONE

    @property
    def total_turnover(self) -> Decimal:
        """Sum of the turnover at every rebalance."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return sum((t for t in self.turnover if t is not None), _ZERO)

    @property
    def turnover_per_period(self) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.total_turnover / Decimal(self.periods)


def _run(label: str, target: Weights, returns: Returns,
         should_rebalance, *, one_sided: bool) -> ScheduleResult:
    _check(target, returns)
    tgt = {k: Decimal(v) for k, v in target.items()}
    current = dict(tgt)
    rets: List[Decimal] = []
    path: List[Dict[str, Decimal]] = []
    turns: List[Optional[Decimal]] = []
    rebalances = 0
    for i, period in enumerate(returns):
        if i == 0:
            turns.append(None)
        elif should_rebalance(i, current, tgt):
            turns.append(turnover_between(current, tgt, one_sided=one_sided))
            current = dict(tgt)
            rebalances += 1
        else:
            turns.append(None)
        path.append(dict(current))
        current, pnl = drift_once(current, period)
        rets.append(pnl)
    return ScheduleResult(label=label, periods=len(returns),
                          returns=tuple(rets), weights=tuple(path),
                          turnover=tuple(turns), rebalances=rebalances)


def periodic_rebalance(target: Weights, returns: Returns, *, every: int,
                       one_sided: bool = True,
                       label: Optional[str] = None) -> ScheduleResult:
    """Snap back to ``target`` every ``every`` periods.

    The book starts at ``target``, drifts through each period, and is traded
    back at the start of every ``every``-th period after the first. ``every=1``
    is the schedule most backtests assume without saying so: rebalanced at
    every observation, at no cost, with the turnover never reported.

    ``every`` is required. There is no sensible default, and the frequency is
    the parameter this module exists to make visible.
    """
    if every < 1:
        raise ValueError(f"every must be at least one period, got {every}")
    return _run(label or f"every {every}", target, returns,
                lambda i, cur, tgt: i % every == 0, one_sided=one_sided)


def band_rebalance(target: Weights, returns: Returns, *, band: Decimal,
                   one_sided: bool = True,
                   label: Optional[str] = None) -> ScheduleResult:
    """Rebalance only when some weight has drifted more than ``band``.

    The band is in weight units, not relative ones: ``band=Decimal("0.02")``
    triggers when any name is two percentage points away from its target. This
    is what a desk with a risk limit actually does, and it trades far less
    than a calendar rule for a similar drift profile — which is the comparison
    :func:`compare_schedules` exists to price.
    """
    if band <= 0:
        raise ValueError(f"band must be positive, got {band}")
    b = Decimal(band)

    def trigger(i: int, cur: Dict[str, Decimal], tgt: Dict[str, Decimal]) -> bool:
        return any(abs(cur.get(k, _ZERO) - tgt.get(k, _ZERO)) > b
                   for k in set(cur) | set(tgt))

    return _run(label or f"band {b}", target, returns, trigger,
                one_sided=one_sided)


def buy_and_hold(target: Weights, returns: Returns, *,
                 label: str = "never") -> ScheduleResult:
    """Never rebalance. The weights are whatever the returns made them."""
    return _run(label, target, returns, lambda i, cur, tgt: False,
                one_sided=True)


@dataclass(frozen=True, slots=True)
class DriftReport:
    """How far the weights wandered from their target, and which name led."""

    periods: int
    band: Decimal
    max_drift: Decimal
    mean_abs_drift: Decimal
    worst_name: str
    periods_outside_band: int

    @property
    def share_outside_band(self) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(self.periods_outside_band) / Decimal(self.periods)


def drift_report(target: Weights, weights: Sequence[Mapping[str, Decimal]],
                 *, band: Decimal) -> DriftReport:
    """Summarise a weight path against the target it started from.

    ``band`` is required because "how far is too far" is a risk decision and
    not a property of the data. The report counts periods in which any single
    name was outside it, which is the condition :func:`band_rebalance` fires on.
    """
    if not weights:
        raise ValueError("a drift report needs at least one weight vector")
    if band <= 0:
        raise ValueError(f"band must be positive, got {band}")
    tgt = {k: Decimal(v) for k, v in target.items()}
    b = Decimal(band)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        worst = _ZERO
        worst_name = next(iter(tgt))
        total = _ZERO
        count = 0
        outside = 0
        for row in weights:
            row_outside = False
            for k in set(row) | set(tgt):
                d = abs(Decimal(row.get(k, _ZERO)) - tgt.get(k, _ZERO))
                total += d
                count += 1
                if d > worst:
                    worst, worst_name = d, k
                if d > b:
                    row_outside = True
            if row_outside:
                outside += 1
        return DriftReport(periods=len(weights), band=b, max_drift=worst,
                           mean_abs_drift=total / Decimal(count),
                           worst_name=worst_name,
                           periods_outside_band=outside)


@dataclass(frozen=True, slots=True)
class ScheduleComparison:
    """Several schedules on one set of returns, with the cost that separates them.

    ``results`` keeps insertion order. ``most_active`` and ``least_active`` are
    by total turnover, not by return.
    """

    results: Tuple[ScheduleResult, ...]

    @property
    def most_active(self) -> ScheduleResult:
        return max(self.results, key=lambda r: r.total_turnover)

    @property
    def least_active(self) -> ScheduleResult:
        return min(self.results, key=lambda r: r.total_turnover)

    @property
    def return_spread(self) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            totals = [r.total_return for r in self.results]
            return max(totals) - min(totals)

    @property
    def breakeven_cost(self) -> Optional[Decimal]:
        """Cost per unit of turnover at which the active schedule stops winning.

        Extra return divided by extra turnover, comparing the most and least
        active schedules. In the same units as the turnover convention the
        results were built with: a one-sided turnover of 0.10 costed at this
        number per unit gives back exactly the return difference.

        ``None`` when the two traded the same amount, and **negative when the
        more active schedule earned less**, which is not an error and not a
        cost level — it means the extra trading lost money before anybody
        charged for it. The sign is reported rather than hidden because that
        case is common and reads as a bargain if it is shown as a magnitude.
        """
        hi, lo = self.most_active, self.least_active
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            extra_turnover = hi.total_turnover - lo.total_turnover
            if extra_turnover == 0:
                return None
            return (hi.total_return - lo.total_return) / extra_turnover

    @property
    def breakeven_cost_bps(self) -> Optional[Decimal]:
        """:attr:`breakeven_cost` in basis points of traded notional."""
        c = self.breakeven_cost
        if c is None:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return c * Decimal(10_000)


def compare_schedules(results: Mapping[str, ScheduleResult]) -> ScheduleComparison:
    """Line several schedules up on the same returns.

    Every result must cover the same number of periods; comparing a five-year
    daily rule against a two-year monthly one is the mistake this refuses to
    make silently.
    """
    if not results:
        raise ValueError("a comparison needs at least one schedule")
    lengths = {r.periods for r in results.values()}
    if len(lengths) > 1:
        raise ValueError(
            f"the schedules cover different numbers of periods: "
            f"{sorted(lengths)}. They have to run on the same returns for the "
            "difference between them to be about the schedule.")
    return ScheduleComparison(results=tuple(results.values()))
