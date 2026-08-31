"""What the trade costs, before you have made it.

:mod:`mdnorm.execution` measures what your fills actually cost against the
market that was there. This module answers the other question: what a backtest
should charge itself for a trade it never made.

That is the sixth way a backtest flatters you, and it is the crudest. The
first four read the future; the fifth reports the best of many attempts; this
one simply forgets to pay. It survives every check in the library because
nothing in the data is wrong — the strategy is being priced at a level nobody
trades at::

    from mdnorm import CostModel, Fees, ImpactModel, Liquidity, estimate

    model = CostModel(fees=Fees(taker_bps=D(1)),
                      impact=ImpactModel(coefficient=D("0.5")))
    liq = Liquidity(adv=D(2_000_000), volatility=D("0.02"), spread_bps=D(4))

    estimate(model, notional=D(500_000), quantity=D(20_000), liquidity=liq)

**Zero cost is not a default, it is a claim.** In the same way a delivery delay
of zero is a statement about your infrastructure rather than an absence of one,
a backtest that charges nothing has asserted that it trades at the midpoint, in
unlimited size, for free. Written down that way nobody would sign it.

**A cost that does not depend on size is not a cost model.** Charging a flat
five basis points says a strategy can trade a thousand dollars and a billion on
identical terms. Every capacity question then has the same answer, which is why
the answer is always wrong. :class:`ImpactModel` makes the charge grow with the
fraction of daily volume you take, so a strategy that only works at size is
visibly a strategy that only works at size.

**There is no default impact coefficient.** The square-root law — cost moves
with volatility times the square root of participation — is well supported, and
the constant in front of it is not universal. It depends on the venue, the
instrument, the horizon and how you execute. :class:`ImpactModel` requires you
to supply one, for the same reason :func:`mdnorm.features.periods_per_year`
requires a calendar: a plausible wrong constant rescales every cost in the
report and changes nothing about its shape.

**The useful output is not the cost.** It is
:func:`breakeven_participation` — the fraction of daily volume at which the
edge is exactly consumed — and :func:`capacity`, the same figure as a quantity.
A strategy with a two-basis-point edge and a breakeven at 0.3% of volume is a
different object from one with the same edge and a breakeven at 30%, and the
Sharpe ratio does not distinguish them.

Costs are quoted in basis points of traded notional throughout, and the sign
convention is that a cost is positive.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Fees",
    "Liquidity",
    "ImpactModel",
    "CostModel",
    "CostBreakdown",
    "CostReport",
    "estimate",
    "apply_costs",
    "cost_report",
    "breakeven_participation",
    "capacity",
]

_Series = Sequence[Optional[Decimal]]

#: Working precision, matching the rest of the library.
_PRECISION = 34

_ZERO = Decimal(0)
_ONE = Decimal(1)
_BPS = Decimal(10_000)

#: Participation above this is outside the range where a square-root impact
#: model is usually calibrated, and the estimate is reported with a warning.
_HIGH_PARTICIPATION = Decimal("0.10")


def _pow(base: Decimal, exponent: Decimal) -> Decimal:
    """``base ** exponent`` for positive ``base``, exactly for one half."""
    if base == 0:
        return _ZERO
    if exponent == _ONE:
        return base
    if exponent == Decimal("0.5"):
        return base.sqrt()
    return (exponent * base.ln()).exp()


# -- the pieces --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fees:
    """Explicit charges: what the venue and the broker take.

    ``maker_bps`` and ``taker_bps`` are basis points of notional. ``per_unit``
    is charged per share or contract, which is how most US equity commissions
    work and which behaves differently from a notional rate as the price
    moves. ``minimum`` is a floor per order — small orders are where it bites,
    and a backtest that rebalances a long tail of small positions pays it
    constantly.
    """

    maker_bps: Decimal = _ZERO
    taker_bps: Decimal = _ZERO
    per_unit: Decimal = _ZERO
    minimum: Decimal = _ZERO

    def __post_init__(self) -> None:
        for name in ("maker_bps", "taker_bps", "per_unit", "minimum"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")

    def commission(
        self, notional: Decimal, quantity: Decimal, *, maker: bool = False
    ) -> Decimal:
        """Currency charged on one order.

        A maker rebate is not expressible here: ``maker_bps`` is a cost like
        every other figure in this module. Model a rebate as a negative
        component of your own edge rather than as a negative cost, so the cost
        report cannot come out below zero and hide it.
        """
        if notional < 0 or quantity < 0:
            raise ValueError("notional and quantity must not be negative")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            rate = self.maker_bps if maker else self.taker_bps
            charged = notional * rate / _BPS + quantity * self.per_unit
            return max(charged, self.minimum) if quantity or notional else _ZERO


@dataclass(frozen=True, slots=True)
class Liquidity:
    """What is known about the instrument when the trade is sized.

    ``adv`` is average daily volume in the same units as the quantity traded.
    ``volatility`` is the daily standard deviation of returns as a fraction —
    0.02 for two percent, not 2. ``spread_bps`` is the quoted spread, not half
    of it; how much of it you pay is a property of how you execute and lives
    on :class:`CostModel`.

    All three are point-in-time quantities. Estimating a 2019 trade with an
    ADV computed over the whole sample is the survivorship problem in a
    different costume: the instruments that became liquid look liquid
    throughout, and the ones that dried up never do.
    """

    adv: Decimal
    volatility: Decimal
    spread_bps: Decimal = _ZERO

    def __post_init__(self) -> None:
        if self.adv <= 0:
            raise ValueError("adv must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must not be negative")
        if self.spread_bps < 0:
            raise ValueError("spread_bps must not be negative")

    def participation(self, quantity: Decimal) -> Decimal:
        """``quantity / adv`` — the fraction of a day's volume this trade is."""
        if quantity < 0:
            raise ValueError("quantity must not be negative")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(quantity) / self.adv


@dataclass(frozen=True, slots=True)
class ImpactModel:
    """Cost that grows with the fraction of daily volume taken.

    ``impact = coefficient * volatility * participation ** exponent``, as a
    fraction of price, converted to basis points. With the default exponent of
    one half this is the square-root law: doubling the size raises the cost per
    unit by about forty percent, so the total cost of a trade grows faster than
    its size.

    ``coefficient`` has no default. Published estimates for equities sit
    broadly in the region of a half to one for this specification, and the
    right value for your venue, instrument and execution style is something
    you calibrate against your own fills — which is what
    :func:`mdnorm.execution.evaluate` is for. Anything supplied here should be
    treated as a stated assumption and reported alongside the result.
    """

    coefficient: Decimal
    exponent: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        if self.coefficient < 0:
            raise ValueError("coefficient must not be negative")
        if self.exponent <= 0:
            raise ValueError("exponent must be positive")

    def cost_bps(self, participation: Decimal, volatility: Decimal) -> Decimal:
        """Impact in basis points at this participation rate."""
        if participation < 0:
            raise ValueError("participation must not be negative")
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            frac = self.coefficient * Decimal(volatility) * _pow(
                Decimal(participation), self.exponent
            )
            return frac * _BPS


@dataclass(frozen=True, slots=True)
class CostModel:
    """Fees, the part of the spread you pay, and impact.

    ``spread_fraction`` is how much of the quoted spread a trade gives up:
    one half for an order that crosses, zero for one that is always passive,
    and more than one half for a strategy whose orders are picked off. Half is
    the default because crossing is the assumption a backtest is implicitly
    making when it fills at the midpoint, and stating it is the point.

    ``impact`` may be ``None``, and then the model is size-independent. That
    is a legitimate choice for a strategy trading far inside the noise of daily
    volume, and :func:`estimate` says so in its warnings every time, because it
    is not a legitimate choice for anything else.
    """

    fees: Fees = Fees()
    impact: Optional[ImpactModel] = None
    spread_fraction: Decimal = Decimal("0.5")

    def __post_init__(self) -> None:
        if self.spread_fraction < 0:
            raise ValueError("spread_fraction must not be negative")


# -- one trade ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """One trade, priced, with each component kept separate.

    The components are separate because they behave differently. Fees are
    linear in size, the spread is linear, and impact is not — so the mix
    changes as the trade grows, and a single total hides which one will bite
    first when the strategy is scaled.
    """

    notional: Decimal
    quantity: Decimal
    participation: Optional[Decimal]
    commission: Decimal
    commission_bps: Decimal
    spread_bps: Decimal
    impact_bps: Decimal
    total_bps: Decimal
    total: Decimal
    warnings: Tuple[str, ...] = ()

    @property
    def fixed_bps(self) -> Decimal:
        """The part that does not grow with size: fees plus the spread."""
        return self.commission_bps + self.spread_bps


def estimate(
    model: CostModel,
    *,
    notional: Decimal,
    quantity: Decimal,
    liquidity: Optional[Liquidity] = None,
    maker: bool = False,
) -> CostBreakdown:
    """Price one trade under ``model``.

    ``liquidity`` is optional only so that a fees-and-spread model can be used
    without one. Omitting it with an impact model configured is an error rather
    than a silent zero, because the impact term is the one that decides whether
    the strategy scales.
    """
    if notional < 0 or quantity < 0:
        raise ValueError("notional and quantity must not be negative")
    if model.impact is not None and liquidity is None:
        raise ValueError(
            "an impact model needs liquidity: pass a Liquidity, or drop the "
            "impact model and accept that the cost is size-independent"
        )
    warns: List[str] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        notional = Decimal(notional)
        quantity = Decimal(quantity)

        commission = model.fees.commission(notional, quantity, maker=maker)
        commission_bps = (
            commission / notional * _BPS if notional else _ZERO
        )

        spread_bps = _ZERO
        if liquidity is not None:
            spread_bps = liquidity.spread_bps * model.spread_fraction
        elif model.spread_fraction:
            warns.append(
                "no liquidity supplied, so no spread was charged; the trade is "
                "priced as if it filled at the midpoint"
            )

        participation: Optional[Decimal] = None
        impact_bps = _ZERO
        if liquidity is not None:
            participation = liquidity.participation(quantity)
        if (model.impact is not None and liquidity is not None
                and participation is not None):
            impact_bps = model.impact.cost_bps(participation, liquidity.volatility)
            if participation > _HIGH_PARTICIPATION:
                warns.append(
                    f"participation is {participation * 100:.1f}% of daily "
                    f"volume, beyond the range where a square-root model is "
                    f"usually calibrated; treat the impact figure as a lower bound"
                )
        elif model.impact is None:
            warns.append(
                "no impact model configured, so this cost does not depend on "
                "trade size; every capacity question will answer the same way"
            )

        total_bps = commission_bps + spread_bps + impact_bps
        total = notional * total_bps / _BPS
        return CostBreakdown(
            notional, quantity, participation, commission, commission_bps,
            spread_bps, impact_bps, total_bps, total, tuple(warns),
        )


# -- a whole return series ---------------------------------------------------


def apply_costs(
    gross_returns: _Series, turnovers: _Series, *, cost_bps: Decimal
) -> List[Optional[Decimal]]:
    """Subtract trading costs from a series of gross returns.

    ``turnovers`` is one-sided, as produced by :func:`mdnorm.metrics.turnover`:
    a value of 1 means the whole book was replaced, which is two units of
    trading — one out and one in. The charge is therefore ``2 * turnover *
    cost_bps``, and getting that factor wrong halves every cost in the report.

    A ``None`` in either series yields ``None`` at that index rather than being
    treated as zero, so a period with unknown turnover does not silently become
    a free one.
    """
    if len(gross_returns) != len(turnovers):
        raise ValueError("gross_returns and turnovers must be the same length")
    if cost_bps < 0:
        raise ValueError("cost_bps must not be negative")
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        rate = Decimal(cost_bps) / _BPS
        for g, t in zip(gross_returns, turnovers):
            if g is None or t is None:
                out.append(None)
            else:
                out.append(Decimal(g) - 2 * Decimal(t) * rate)
    return out


@dataclass(frozen=True, slots=True)
class CostReport:
    """What the costs did to the result, rather than what they were.

    ``cost_fraction`` is the figure worth reporting: the share of the gross
    return that trading consumed. A strategy that keeps eighty percent of its
    gross is robust to a cost model being somewhat wrong. One that keeps ten
    percent is a bet on the cost model, not on the market.
    """

    periods: int
    skipped: int
    gross_return: Optional[Decimal]
    net_return: Optional[Decimal]
    cost: Optional[Decimal]
    cost_fraction: Optional[Decimal]
    total_turnover: Decimal
    average_turnover: Optional[Decimal]
    cost_bps: Decimal
    warnings: Tuple[str, ...] = ()


def cost_report(
    gross_returns: _Series, turnovers: _Series, *, cost_bps: Decimal
) -> CostReport:
    """Apply ``cost_bps`` to a series and say what it cost.

    Returns are compounded, so the cost is the difference between the two
    compounded totals rather than the sum of the per-period charges — those
    differ, and the difference grows with the length of the sample.
    """
    net = apply_costs(gross_returns, turnovers, cost_bps=cost_bps)
    warns: List[str] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        g_level = _ONE
        n_level = _ONE
        periods = 0
        skipped = 0
        turn = _ZERO
        for g, t, n in zip(gross_returns, turnovers, net):
            if g is None or t is None or n is None:
                skipped += 1
                continue
            periods += 1
            turn += Decimal(t)
            g_level *= _ONE + Decimal(g)
            n_level *= _ONE + Decimal(n)

        if skipped:
            warns.append(
                f"{skipped} period(s) had no return or no turnover and were "
                f"excluded"
            )
        if periods == 0:
            return CostReport(0, skipped, None, None, None, None, _ZERO, None,
                              Decimal(cost_bps), tuple(warns) + ("no periods",))
        if turn == 0:
            warns.append(
                "turnover was zero in every period, so the cost model was "
                "never exercised; this report says nothing about it"
            )

        gross = g_level - _ONE
        net_total = n_level - _ONE
        cost = gross - net_total
        fraction = cost / gross if gross != 0 else None

        if gross > 0 and net_total <= 0:
            warns.append(
                "the strategy is profitable before costs and not after them"
            )
        elif fraction is not None and gross > 0 and fraction > Decimal("0.5"):
            warns.append(
                f"costs consume {fraction * 100:.1f}% of the gross return; the "
                f"result depends more on the cost assumption than on the signal"
            )

        return CostReport(
            periods, skipped, gross, net_total, cost, fraction, turn,
            turn / periods, Decimal(cost_bps), tuple(warns),
        )


# -- the questions worth asking ----------------------------------------------


def breakeven_participation(
    edge_bps: Decimal, *, model: CostModel, liquidity: Liquidity
) -> Optional[Decimal]:
    """Fraction of daily volume at which ``edge_bps`` is exactly consumed.

    ``edge_bps`` is the gross edge of one round trip in basis points, before
    any cost. The answer is the participation rate at which fees, spread and
    impact add up to it — trade smaller and the strategy makes money, trade
    larger and it does not.

    ``None`` when the fixed costs alone already exceed the edge, which is a
    different failure and worth distinguishing: no trade size makes that
    strategy work, so there is nothing to solve for. ``None`` also when the
    model has no impact term, since then the cost never grows and the equation
    has no root.

    Only the notional part of the fee schedule enters here. ``per_unit`` and
    ``minimum`` cannot be turned into basis points without a price, so a
    breakeven computed against a per-share commission is optimistic; price the
    trade with :func:`estimate` when that part of the schedule matters.
    """
    if edge_bps < 0:
        raise ValueError("edge_bps must not be negative")
    if model.impact is None:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        fixed = liquidity.spread_bps * model.spread_fraction + model.fees.taker_bps
        remaining = Decimal(edge_bps) - fixed
        if remaining <= 0:
            return None
        scale = model.impact.coefficient * liquidity.volatility * _BPS
        if scale <= 0:
            return None
        try:
            ratio = remaining / scale
            return _pow(ratio, _ONE / model.impact.exponent)
        except (InvalidOperation, ValueError):
            return None


def capacity(
    edge_bps: Decimal, *, model: CostModel, liquidity: Liquidity
) -> Optional[Decimal]:
    """The quantity at which the edge is exactly consumed.

    :func:`breakeven_participation` multiplied by average daily volume, in the
    units the volume is quoted in. This is the number a strategy should be
    reported with: an annualised return means something different at a hundred
    thousand and at a hundred million, and only one of them is usually
    achievable.

    ``None`` for the same reasons :func:`breakeven_participation` returns it.
    """
    p = breakeven_participation(edge_bps, model=model, liquidity=liquidity)
    if p is None:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return p * liquidity.adv
