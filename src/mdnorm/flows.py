"""A strategy earns a return. An investor earns a rate on the money present.

Every return series elsewhere in this library describes a unit of capital held
from the first observation to the last. Nobody invests that way. Money arrives
after a good year and leaves after a bad one, and a rate of return that ignores
when it arrived is not the rate anybody earned::

    from mdnorm import flow_report

    rep = flow_report(returns, flows, when="start")

    rep.time_weighted        # what the strategy earned: the backtest figure
    rep.money_weighted_total # what the investor earned on this flow path
    rep.gap                  # the difference, with a direction

**The two figures answer different questions and only one of them is a
property of the strategy.** The time-weighted return chain-links the periods
and is deliberately blind to the size of the account, which is what makes it
the right number for judging a manager. The money-weighted return is the rate
that grows every contribution into the final value, so it weights each period
by the capital exposed to it. A manager is fairly judged on the first. An
investor is paid the second.

**Part of the gap is arithmetic and not behaviour at all.** The two rates are
different averages of the same periods: chain-linking compounds them, so the
time-weighted figure is a geometric mean, while the money-weighted rate weights
each period by the capital exposed to it. On a level schedule with no timing in
it whatever, the two still differ, and the difference has the sign of the
volatility drag described in :mod:`mdnorm.compounding`.
:attr:`FlowReport.timing_effect` holds the amount fixed and removes the timing,
so that what is left of the gap is the part a decision could have changed.

**The rest of the gap is not noise and it is not symmetric in practice.**
Capital tends to
arrive after good performance and leave after bad, so the periods carrying the
most money are disproportionately the ones that follow a strong run. Nothing
here assumes that; the module measures it on the path it is given.
:attr:`FlowReport.capital_at_worst_ratio` says how much capital was exposed to
the worst period relative to the average one, which is the mechanism rather
than the verdict.

**The timing convention is required.** A contribution at the start of a period
earns that period's return and the same contribution at the end does not. On a
volatile series the difference is not a rounding detail, so ``when`` has no
default: pass ``"start"`` or ``"end"`` and state which one the data means.

**The approximation used by most administrators is reported beside the exact
answer, with its error.** :func:`modified_dietz` weights each flow by the
fraction of the window remaining and needs no root-finding, which is why it is
in every performance report. It is a simple-interest figure: it divides the
gain by a weighted capital base and never compounds, so it agrees with the
internal rate of return only when there is one flow at the start of the
window. With flows spread through the window it understates even a perfectly
constant rate — twelve monthly contributions at one per cent a month come out
at 12.4512 per cent against a true 12.6825 — and the error grows with the
window, with the size of the flows and with the dispersion of the returns.
:attr:`FlowReport.dietz_error` is its size in this sample rather than a
correction to apply, because the sign is not guaranteed.

**No annualisation without a calendar.** Every rate here is per period or over
the whole window. :meth:`FlowReport.annualised` takes the periods in a year as
an argument, for the reason stated throughout this library: a default constant
rescales a report without changing its shape.

A money-weighted return is the internal rate of return of the flow path, and
an internal rate of return can have more than one root when the sign of the
cash flow changes more than once. :attr:`FlowReport.root_unique` says whether
that condition holds here, and :meth:`FlowReport.npv` lets a caller check any
rate directly rather than trusting a single number.

Arithmetic runs at forty significant digits, matching the other evaluation
modules. See :mod:`mdnorm.fundfees` for the fee between the gross return and
the investor, and :mod:`mdnorm.hurdle` for what a return is measured against.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FlowReport",
    "FlowComparison",
    "balances",
    "compare_flows",
    "flow_report",
    "internal_rate_of_return",
    "level_flows",
    "modified_dietz",
    "sign_changes",
    "time_weighted",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_PRECISION = 40
_WHEN = ("start", "end")
_FLOOR = Decimal("-0.999999999999999999")


def _check(returns: Sequence[Decimal], flows: Sequence[Decimal],
           when: str) -> Tuple[List[Decimal], List[Decimal]]:
    if when not in _WHEN:
        raise ValueError(
            f"when is one of {_WHEN}, got {when!r}. A flow at the start of a "
            "period earns that period's return and one at the end does not; "
            "there is no default because the difference is the answer.")
    if not returns:
        raise ValueError("a flow calculation needs at least one period")
    if len(flows) != len(returns):
        raise ValueError(
            f"{len(flows)} flows and {len(returns)} returns. They must "
            "describe the same periods; align them first rather than letting "
            "one be truncated.")
    rs = [Decimal(r) for r in returns]
    for i, r in enumerate(rs):
        if r <= -1:
            raise ValueError(
                f"return {i} is {r}; a simple return at or below -1 empties "
                "the account. Convert log returns with to_simple first.")
    return rs, [Decimal(f) for f in flows]


def balances(returns: Sequence[Decimal], flows: Sequence[Decimal], *,
             when: str) -> Tuple[Decimal, ...]:
    """End-of-period account values for one return path and one flow path.

    With ``when="start"`` the flow for a period is added before that period's
    return is applied; with ``when="end"`` it is added afterwards. Refuses a
    path on which a withdrawal takes the account below zero, because a rate of
    return for an account that was overdrawn describes nothing.
    """
    rs, fs = _check(returns, flows, when)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        v = _ZERO
        out: List[Decimal] = []
        for i, (r, f) in enumerate(zip(rs, fs)):
            if when == "start":
                v += f
                if v < 0:
                    raise ArithmeticError(
                        f"the flow at period {i} takes the account to {v}. "
                        "A withdrawal larger than the balance is not a "
                        "return path; check the sign convention — "
                        "contributions are positive.")
                v *= _ONE + r
            else:
                v = v * (_ONE + r) + f
                if v < 0:
                    raise ArithmeticError(
                        f"the flow at period {i} takes the account to {v}. "
                        "A withdrawal larger than the balance is not a "
                        "return path; check the sign convention — "
                        "contributions are positive.")
            out.append(v)
        return tuple(out)


def time_weighted(returns: Sequence[Decimal]) -> Decimal:
    """Chain-linked return over the whole window, blind to the flows.

    This is the figure a backtest reports and the one a manager is fairly
    judged on. It is the same for every flow path over the same returns, which
    is the property that makes it useful and the property that makes it the
    wrong answer to *what did I earn*.
    """
    if not returns:
        raise ValueError("a return needs at least one period")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        acc = _ONE
        for i, raw in enumerate(returns):
            r = Decimal(raw)
            if r <= -1:
                raise ValueError(
                    f"return {i} is {r}; a simple return at or below -1 "
                    "empties the account.")
            acc *= _ONE + r
        return acc - _ONE


def _per_period(total: Decimal, periods: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return ((_ONE + total).ln() / Decimal(periods)).exp() - _ONE


def sign_changes(cashflows: Sequence[Decimal]) -> int:
    """Number of times the sign of a cash-flow series changes, ignoring zeros.

    One change is the condition under which an internal rate of return has at
    most one positive root. More than one does not mean the answer is wrong; it
    means a single number is no longer a complete description of the solution.
    """
    changes = 0
    last = 0
    for c in cashflows:
        s = (Decimal(c) > 0) - (Decimal(c) < 0)
        if s == 0:
            continue
        if last and s != last:
            changes += 1
        last = s
    return changes


def _npv(cashflows: Sequence[Decimal], rate: Decimal) -> Decimal:
    acc = _ZERO
    df = _ONE
    step = _ONE / (_ONE + rate)
    for c in cashflows:
        acc += c * df
        df *= step
    return acc


def internal_rate_of_return(cashflows: Sequence[Decimal], *,
                            iterations: int = 400) -> Decimal:
    """The per-period rate at which the cash flows discount to zero.

    ``cashflows[t]`` is the amount the investor receives at time ``t``:
    contributions are negative, withdrawals and the final value positive. The
    root is found by bisection rather than by a Newton step from a guess, so
    the answer does not depend on where the search started.

    Raises when every flow has the same sign, because there is then no rate at
    which they cancel, and when no sign change can be bracketed below a rate of
    ten thousand per period.
    """
    cs = [Decimal(c) for c in cashflows]
    if len(cs) < 2:
        raise ValueError("an internal rate of return needs at least two dates")
    if not sign_changes(cs):
        raise ArithmeticError(
            "every cash flow has the same sign, so there is no rate at which "
            "they cancel. A path with no terminal value, or no contribution, "
            "has no internal rate of return.")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        lo = _FLOOR
        f_lo = _npv(cs, lo)
        hi = _ONE
        f_hi = _npv(cs, hi)
        tries = 0
        while (f_lo > 0) == (f_hi > 0):
            hi *= _TWO
            f_hi = _npv(cs, hi)
            tries += 1
            if tries > 40:
                raise ArithmeticError(
                    "no rate between -1 and 10000 per period discounts these "
                    "flows to zero; the path may have no root, or more than "
                    "one. sign_changes() says whether a single root was ever "
                    "guaranteed.")
        for _ in range(iterations):
            mid = (lo + hi) / _TWO
            f_mid = _npv(cs, mid)
            if f_mid == 0:
                return mid
            if (f_mid > 0) == (f_lo > 0):
                lo, f_lo = mid, f_mid
            else:
                hi, f_hi = mid, f_mid
        return (lo + hi) / _TWO


def _cashflows(rs: Sequence[Decimal], fs: Sequence[Decimal], when: str,
               terminal: Decimal) -> List[Decimal]:
    """Investor-side cash flows, one entry per date, contributions negative."""
    n = len(rs)
    if when == "start":
        cs = [-f for f in fs] + [_ZERO]        # dates 0..n, flows at 0..n-1
    else:
        cs = [_ZERO] + [-f for f in fs]        # flows at dates 1..n
    cs[n] += terminal
    return cs


def modified_dietz(returns: Sequence[Decimal], flows: Sequence[Decimal], *,
                   when: str) -> Decimal:
    """The approximation in every performance report, over the whole window.

    Gain divided by the average capital employed, where each flow is weighted
    by the fraction of the window that remained after it arrived. It needs no
    root-finding, which is why administrators use it, and it charges simple
    interest on that base: it matches the internal rate of return when a
    single flow opens the window and otherwise misses by the compounding it
    does not do. Compare it with the internal rate rather than substituting
    it.

    Raises when the weighted average capital is zero or negative, rather than
    dividing by it.
    """
    rs, fs = _check(returns, flows, when)
    n = len(rs)
    end = balances(rs, fs, when=when)[-1]
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        nn = Decimal(n)
        net = sum(fs, _ZERO)
        weighted = _ZERO
        for i, f in enumerate(fs):
            elapsed = Decimal(i if when == "start" else i + 1)
            weighted += f * (nn - elapsed) / nn
        if weighted <= 0:
            raise ArithmeticError(
                f"the weighted average capital is {weighted}; a return "
                "measured against it would divide by zero or by a negative "
                "base. This happens when withdrawals dominate the window.")
        return (end - net) / weighted


def level_flows(flows: Sequence[Decimal]) -> Tuple[Decimal, ...]:
    """The same net capital, contributed in equal parts at every date.

    The counterfactual that holds the amount fixed and removes the timing. It
    is a reference path, not a recommendation: nobody knows the returns in
    advance, which is exactly why the comparison is interesting.
    """
    fs = [Decimal(f) for f in flows]
    if not fs:
        raise ValueError("a level schedule needs at least one date")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        each = sum(fs, _ZERO) / Decimal(len(fs))
        return tuple(each for _ in fs)


@dataclass(frozen=True, slots=True)
class FlowReport:
    """One return path, one flow path, and the two rates they imply."""

    periods: int
    when: str
    returns: Tuple[Decimal, ...]
    flows: Tuple[Decimal, ...]
    balances: Tuple[Decimal, ...]
    cashflows: Tuple[Decimal, ...]
    time_weighted: Decimal
    money_weighted: Decimal
    modified_dietz: Decimal

    # ---------------------------------------------------------------- money
    @property
    def terminal_value(self) -> Decimal:
        """What the account was worth at the final observation."""
        return self.balances[-1]

    @property
    def contributed(self) -> Decimal:
        """Everything paid in, before any withdrawal."""
        return sum((f for f in self.flows if f > 0), _ZERO)

    @property
    def withdrawn(self) -> Decimal:
        """Everything taken out, as a positive amount."""
        return -sum((f for f in self.flows if f < 0), _ZERO)

    @property
    def profit(self) -> Decimal:
        """Terminal value plus withdrawals less contributions."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.terminal_value + self.withdrawn - self.contributed

    # ---------------------------------------------------------------- rates
    @property
    def time_weighted_per_period(self) -> Decimal:
        """The chain-linked return expressed per period, for comparison."""
        return _per_period(self.time_weighted, self.periods)

    @property
    def money_weighted_total(self) -> Decimal:
        """The internal rate compounded over the window, for comparison.

        Stated because the two headline figures are otherwise quoted on
        different bases: a total against a per-period rate.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return (_ONE + self.money_weighted) ** self.periods - _ONE

    @property
    def gap(self) -> Decimal:
        """Money-weighted total less time-weighted total, over one window.

        Negative means the flow path earned less than the strategy did: the
        capital was smaller in the periods that paid and larger in the ones
        that cost. Positive means the opposite. The sign is a property of the
        path, not of the manager.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.money_weighted_total - self.time_weighted

    @property
    def flows_helped(self) -> bool:
        """Whether the path earned more than the strategy's own return."""
        return self.gap > 0

    @property
    def timing_effect(self) -> Optional[Decimal]:
        """This path's money-weighted total less a level one's, same capital.

        The counterfactual contributes the same net amount in equal parts at
        every date, so the two paths differ only in when the money arrived.
        What is left in :attr:`gap` after this is subtracted is the arithmetic
        difference between a capital-weighted average and a geometric one,
        which no decision about timing could have changed.

        ``None`` when the net contribution is not positive, because a level
        schedule of withdrawals is not a comparison anybody could have made.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            if sum(self.flows, _ZERO) <= 0:
                return None
            level = flow_report(self.returns, level_flows(self.flows),
                                when=self.when)
            return self.money_weighted_total - level.money_weighted_total

    @property
    def dietz_error(self) -> Decimal:
        """Modified Dietz less the compounded internal rate, on this window.

        The cost of the approximation here. It is not a constant and it is
        not a bias to correct: it grows with the length of the window, with
        the size of the flows and with the dispersion of the returns, and on a
        level schedule at a constant positive rate it is negative because the
        approximation never compounds.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.modified_dietz - self.money_weighted_total

    def annualised(self, rate: Decimal, *, periods_per_year: int) -> Decimal:
        """Compound a per-period rate to a year. The calendar is given."""
        if periods_per_year < 1:
            raise ValueError(
                f"periods_per_year must be positive, got {periods_per_year}")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return (_ONE + Decimal(rate)) ** periods_per_year - _ONE

    def npv(self, rate: Decimal) -> Decimal:
        """Present value of the investor's cash flows at any rate.

        Provided so that a reported rate can be checked, and so that a path
        with more than one sign change can be examined rather than summarised.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return _npv(self.cashflows, Decimal(rate))

    @property
    def root_unique(self) -> bool:
        """Whether the cash flows change sign exactly once.

        True is the condition under which the internal rate of return is the
        only root above -100 per cent. False does not invalidate the figure; it
        means the figure is one root of several and should be read with
        :meth:`npv`.
        """
        return sign_changes(self.cashflows) == 1

    # ------------------------------------------------------------- exposure
    @property
    def capital(self) -> Tuple[Decimal, ...]:
        """Capital exposed to each period's return, in period order."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            out: List[Decimal] = []
            prev = _ZERO
            for i, r in enumerate(self.returns):
                base = prev + self.flows[i] if self.when == "start" else prev
                out.append(base)
                prev = self.balances[i]
            return tuple(out)

    @property
    def average_capital(self) -> Decimal:
        """Mean capital exposed across the periods."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return sum(self.capital, _ZERO) / Decimal(self.periods)

    @property
    def worst_period(self) -> int:
        """Index of the period with the lowest return."""
        return min(range(self.periods), key=lambda i: self.returns[i])

    @property
    def best_period(self) -> int:
        """Index of the period with the highest return."""
        return max(range(self.periods), key=lambda i: self.returns[i])

    @property
    def capital_at_worst_ratio(self) -> Optional[Decimal]:
        """Capital exposed to the worst period over the average, or ``None``.

        Above one means more money was in the account for the worst period than
        for a typical one. This is the mechanism behind a negative gap, stated
        as a measurement rather than as a lesson.
        """
        avg = self.average_capital
        if avg <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.capital[self.worst_period] / avg

    @property
    def capital_at_best_ratio(self) -> Optional[Decimal]:
        """Capital exposed to the best period over the average, or ``None``."""
        avg = self.average_capital
        if avg <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.capital[self.best_period] / avg


def flow_report(returns: Sequence[Decimal], flows: Sequence[Decimal], *,
                when: str) -> FlowReport:
    """Both rates, the path they came from, and the difference between them.

    ``returns`` is the strategy's per-period return series and ``flows`` the
    investor's external cash flows over the same periods, contributions
    positive. ``when`` states whether a flow lands before or after that
    period's return.

    The final value is treated as received at the last date, as if the account
    were redeemed there, so the internal rate describes a closed path.
    """
    rs, fs = _check(returns, flows, when)
    bs = balances(rs, fs, when=when)
    cs = _cashflows(rs, fs, when, bs[-1])
    irr = internal_rate_of_return(cs)
    return FlowReport(periods=len(rs), when=when, returns=tuple(rs),
                      flows=tuple(fs), balances=bs, cashflows=tuple(cs),
                      time_weighted=time_weighted(rs), money_weighted=irr,
                      modified_dietz=modified_dietz(rs, fs, when=when))


@dataclass(frozen=True, slots=True)
class FlowComparison:
    """Several flow paths over one unchanged return series.

    The time-weighted return is identical in every row by construction, which
    is the point: every difference in the money-weighted column belongs to the
    path the money took, and none of it to the strategy.
    """

    labels: Tuple[str, ...]
    reports: Tuple[FlowReport, ...]

    @property
    def time_weighted(self) -> Decimal:
        return self.reports[0].time_weighted

    @property
    def money_weighted_spread(self) -> Decimal:
        """Best money-weighted total less worst, on identical returns."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            totals = [r.money_weighted_total for r in self.reports]
            return max(totals) - min(totals)

    def row(self, label: str) -> FlowReport:
        return self.reports[self.labels.index(label)]


def compare_flows(returns: Sequence[Decimal],
                  paths: Mapping[str, Sequence[Decimal]], *,
                  when: str) -> FlowComparison:
    """Run every flow path over the same returns and line them up."""
    if not paths:
        raise ValueError("a comparison needs at least one flow path")
    labels = tuple(paths)
    reports = tuple(flow_report(returns, paths[k], when=when) for k in labels)
    return FlowComparison(labels=labels, reports=reports)
