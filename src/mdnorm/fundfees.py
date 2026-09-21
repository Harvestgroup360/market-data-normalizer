"""A backtest reports what the strategy earned. An investor keeps less.

Every figure elsewhere in this library is gross of the fees a fund charges its
investors. That is the right default for research and the wrong number to put
in front of anyone deciding whether to invest, because the fee is not a
constant subtracted from the return. A management fee compounds against the
investor. An incentive fee is a share of the upside with no share of the
downside, so it takes more of the profit than its headline rate, and how much
more depends on details that are rarely written next to the number::

    from mdnorm import FeeSchedule, apply_fees

    two_and_twenty = FeeSchedule(management=D("0.02"), incentive=D("0.20"),
                                 periods_per_year=12, crystallise_every=12,
                                 high_water_mark=True)
    res = apply_fees(gross_returns, two_and_twenty)

    res.gross_total             # what the strategy earned
    res.net_total               # what the investor kept
    res.fee_share_of_profit     # usually well above twenty per cent

**The headline rate is not the share of profit.** Two and twenty on a
strategy that earns eight per cent a year does not leave the investor eighty
per cent of the gain. The management fee is charged whether or not anything
was earned, and the incentive fee is charged on gains that later losses do not
return. :attr:`FeeResult.fee_share_of_profit` reports the share that actually
went, and it is the number worth comparing between funds.

**How often the fee crystallises is a free parameter that moves the answer.**
A fee crystallised monthly is taken on every good month and never returned on
a bad one; the same fee crystallised annually nets the year first. Nothing in
the gross returns changes between the two. This is the rebalancing-frequency
argument from :mod:`mdnorm.rebalance` again, one layer up: an unstated
schedule that decides how much is taken.

**A high-water mark is a promise with a memory, and without one the incentive
fee has none.** ``high_water_mark=False`` measures each crystallisation
against the previous one, so a fund that loses a third and then recovers it is
paid for the recovery. That is a real contract term in some vehicles, which is
why it is offered, and a costly one, which is why it is never assumed.

**The investor leaves at the end of the sample.** An incentive fee accrued
since the last crystallisation is charged at the final observation rather than
left off, because a net figure that ignores an accrued fee is a figure nobody
could have redeemed at.

There is no default fee schedule. Every field of :class:`FeeSchedule` is
required, including the ones everybody thinks they know, because "two and
twenty" is four decisions and the other two are the ones that matter.

Arithmetic runs at forty significant digits, matching the other evaluation
modules. The module is named ``fundfees`` rather than ``fees`` so that it is
not confused with :class:`mdnorm.costs.Fees`, which prices a trade.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FeeSchedule",
    "FeeResult",
    "FeeComparison",
    "apply_fees",
    "compare_fees",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_PRECISION = 40


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    """The four decisions inside "two and twenty", and an optional hurdle.

    ``management`` is an annual rate charged on start-of-period net asset
    value, accrued by dividing by ``periods_per_year`` — the convention fund
    administrators use, and stated because compounding would give a slightly
    different number. ``incentive`` is the share of qualifying gains taken at
    each crystallisation, every ``crystallise_every`` periods and at the final
    observation.

    ``hurdle``, when given, is a per-period rate series the high-water mark
    grows by, so the incentive fee is charged only on gains above it. It must
    be as long as the return series. See :mod:`mdnorm.hurdle` for turning a
    quoted rate into a per-period one.
    """

    management: Decimal
    incentive: Decimal
    periods_per_year: int
    crystallise_every: int
    high_water_mark: bool
    hurdle: Optional[Tuple[Decimal, ...]] = None

    def __post_init__(self) -> None:
        if self.management < 0:
            raise ValueError(
                f"management fee cannot be negative, got {self.management}")
        if not (0 <= self.incentive < 1):
            raise ValueError(
                f"incentive share lies in [0, 1), got {self.incentive}")
        if self.periods_per_year < 1:
            raise ValueError(
                f"periods_per_year must be positive, got {self.periods_per_year}")
        if self.crystallise_every < 1:
            raise ValueError(
                "crystallise_every must be at least one period, got "
                f"{self.crystallise_every}")
        if self.hurdle is not None and self.high_water_mark is False:
            raise ValueError(
                "a hurdle grows the high-water mark, and this schedule has "
                "none. Set high_water_mark=True or drop the hurdle.")


@dataclass(frozen=True, slots=True)
class FeeResult:
    """One gross return series taken through one fee schedule."""

    periods: int
    gross_returns: Tuple[Decimal, ...]
    net_returns: Tuple[Decimal, ...]
    management_paid: Decimal
    incentive_paid: Decimal
    crystallisations: int
    charged: int

    @property
    def gross_total(self) -> Decimal:
        """Compounded gross return over the whole sample."""
        return _compound(self.gross_returns)

    @property
    def net_total(self) -> Decimal:
        """Compounded return to the investor over the whole sample."""
        return _compound(self.net_returns)

    @property
    def fee_share_of_profit(self) -> Optional[Decimal]:
        """Share of the gross gain that did not reach the investor.

        ``1 - net_total / gross_total``. This is the figure to compare against
        the headline incentive rate, and it is usually well above it.
        ``None`` when the gross total is not positive, because a share of a
        loss is not a share of profit — and a fund that lost money and still
        charged a management fee has a fee problem that a ratio would hide.
        """
        g = self.gross_total
        if g <= 0:
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return _ONE - self.net_total / g

    @property
    def incentive_share_of_fees(self) -> Optional[Decimal]:
        """How much of what was paid was the incentive fee, by amount."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            total = self.management_paid + self.incentive_paid
            if total == 0:
                return None
            return self.incentive_paid / total

    def sharpe(self, *, net: bool, ddof: int) -> Optional[Decimal]:
        """Per-period Sharpe ratio of the gross or net series, no hurdle.

        The incentive fee trims the upside and leaves the downside, so net
        volatility is usually lower than gross and the Sharpe ratio falls by
        less than the return does. Both are available so that the smaller
        fall is not mistaken for a smaller fee.
        """
        xs = self.net_returns if net else self.gross_returns
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            n = len(xs)
            if n - ddof < 1:
                return None
            m = sum(xs, _ZERO) / Decimal(n)
            var = sum(((x - m) * (x - m) for x in xs), _ZERO) / Decimal(n - ddof)
            if var <= 0:
                return None
            return m / var.sqrt()


def _compound(xs: Sequence[Decimal]) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        acc = _ONE
        for x in xs:
            acc *= _ONE + x
        return acc - _ONE


def apply_fees(gross_returns: Sequence[Decimal],
               schedule: FeeSchedule) -> FeeResult:
    """Take a gross per-period return series through a fee schedule.

    Each period: the gross return is applied to start-of-period net asset
    value; the management fee for the period is charged on that
    start-of-period value; the high-water mark grows by the hurdle if there
    is one; and at a crystallisation the incentive fee is taken on the gain
    above the high-water mark, or above the previous crystallisation when
    there is no mark. The final observation always crystallises.

    Refuses a gross return at or below -100 per cent, and a period whose fees
    would take net asset value to zero or below, rather than reporting a net
    figure for an account that no longer exists.
    """
    if not gross_returns:
        raise ValueError("a fee calculation needs at least one period")
    n = len(gross_returns)
    if schedule.hurdle is not None and len(schedule.hurdle) != n:
        raise ValueError(
            f"the hurdle has {len(schedule.hurdle)} periods and the returns "
            f"have {n}. They must describe the same periods; align them "
            "first rather than letting one be truncated.")

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        mgmt_rate = Decimal(schedule.management) / Decimal(schedule.periods_per_year)
        inc = Decimal(schedule.incentive)
        nav = _ONE
        mark = _ONE
        mgmt_paid = _ZERO
        inc_paid = _ZERO
        crystallisations = 0
        charged = 0
        gross: List[Decimal] = []
        net: List[Decimal] = []
        for i, raw in enumerate(gross_returns):
            r = Decimal(raw)
            if r <= -1:
                raise ValueError(
                    f"gross return {i} is {r}; a simple return at or below "
                    "-1 empties the account. Convert log returns with "
                    "mdnorm.to_simple first.")
            start = nav
            fee_m = start * mgmt_rate
            nav = start * (_ONE + r) - fee_m
            mgmt_paid += fee_m
            if schedule.hurdle is not None:
                mark *= _ONE + Decimal(schedule.hurdle[i])
            last = i == n - 1
            if (i + 1) % schedule.crystallise_every == 0 or last:
                crystallisations += 1
                gain = nav - mark
                if gain > 0 and inc > 0:
                    fee_i = inc * gain
                    nav -= fee_i
                    inc_paid += fee_i
                    charged += 1
                if schedule.high_water_mark:
                    if nav > mark:
                        mark = nav
                else:
                    mark = nav
            if nav <= 0:
                raise ArithmeticError(
                    f"net asset value is {nav} after period {i}; the fees "
                    "have exceeded what was left. Check the management rate "
                    "and periods_per_year — an annual rate passed as a "
                    "per-period one does this.")
            gross.append(r)
            net.append(nav / start - _ONE)
        return FeeResult(periods=n, gross_returns=tuple(gross),
                         net_returns=tuple(net), management_paid=mgmt_paid,
                         incentive_paid=inc_paid,
                         crystallisations=crystallisations, charged=charged)


@dataclass(frozen=True, slots=True)
class FeeComparison:
    """Several fee schedules applied to one gross series.

    ``labels`` and ``results`` are in the order supplied. The gross total is
    the same for every entry by construction, which is the point: every
    difference in the net column is the contract.
    """

    labels: Tuple[str, ...]
    results: Tuple[FeeResult, ...]

    @property
    def gross_total(self) -> Decimal:
        return self.results[0].gross_total

    @property
    def net_spread(self) -> Decimal:
        """Best net total minus worst, on identical gross returns."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            nets = [r.net_total for r in self.results]
            return max(nets) - min(nets)

    def row(self, label: str) -> FeeResult:
        return self.results[self.labels.index(label)]


def compare_fees(gross_returns: Sequence[Decimal],
                 schedules: Mapping[str, FeeSchedule]) -> FeeComparison:
    """Apply every schedule to the same gross series and line them up."""
    if not schedules:
        raise ValueError("a comparison needs at least one fee schedule")
    labels = tuple(schedules)
    results = tuple(apply_fees(gross_returns, schedules[k]) for k in labels)
    return FeeComparison(labels=labels, results=results)
