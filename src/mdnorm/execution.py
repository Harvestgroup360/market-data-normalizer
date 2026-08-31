"""Execution benchmarks: VWAP, TWAP, slippage and implementation shortfall.

Having normalised a tape, the next question is usually not about the market
but about yourself: did I execute well? The standard answers are a handful of
benchmarks, and each of them has a way of quietly flattering the person
running it::

    from mdnorm import Fill, Side, evaluate, exclude_fills

    market = exclude_fills(market_trades, my_fills)   # do this first
    report = evaluate(my_fills, market, decision_price=D("100"))
    print(report.slippage_bps, report.participation_rate)

**Your own trades are in the benchmark.** A VWAP computed over the public tape
includes the prints you just made. Benchmark yourself against it and you are
partly benchmarking yourself against yourself — and the larger your share of
volume, the more the benchmark bends toward your own average price, so the
worse you traded the better you score. :func:`exclude_fills` removes your
prints from the tape before the benchmark is computed. It is the first line of
any honest measurement and it is almost always skipped.

**Participation rate decides whether the number means anything.** Beating VWAP
by two basis points on 0.1% of volume is a result; the same number on 30% of
volume mostly measures how much you moved the price. :class:`ExecutionSummary`
always reports participation next to the score, so the two cannot be read
apart.

**Sign conventions are not obvious.** Here, positive basis points always mean
*better than the benchmark*: for a buy that is paying below it, for a sell it
is selling above it. Mixed-side fills are refused rather than netted, because
a single number covering both directions has no meaning.

Nothing here is a performance claim or a recommendation. These are arithmetic
definitions applied to data the caller supplies.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable, List, Optional, Sequence, cast

from .schema import EventType, MarketEvent, Side

__all__ = [
    "Fill",
    "ExecutionSummary",
    "vwap",
    "twap",
    "exclude_fills",
    "participation_rate",
    "average_fill_price",
    "slippage_bps",
    "implementation_shortfall_bps",
    "evaluate",
]

_BPS = Decimal(10_000)


@dataclass(frozen=True, slots=True)
class Fill:
    """One of your own executions."""

    ts_ns: int
    price: Decimal
    size: Decimal
    side: Side
    venue: str = ""

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.size <= 0:
            raise ValueError("size must be positive")


def _trades(events: Iterable[MarketEvent]) -> List[MarketEvent]:
    return sorted(
        (e for e in events
         if e.event_type is EventType.TRADE and e.price is not None),
        key=lambda e: e.ts_ns,
    )


def _in_window(ts: int, start_ns: Optional[int], end_ns: Optional[int]) -> bool:
    if start_ns is not None and ts < start_ns:
        return False
    if end_ns is not None and ts >= end_ns:
        return False
    return True


# -- benchmarks --------------------------------------------------------------


def vwap(
    events: Iterable[MarketEvent],
    *,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Optional[Decimal]:
    """Volume-weighted average price over ``[start_ns, end_ns)``.

    Trades without a size contribute nothing, since a weighted average with
    an unknown weight is not defined. Returns ``None`` when no traded volume
    falls in the window — distinct from a price of zero.
    """
    notional = Decimal(0)
    volume = Decimal(0)
    for e in _trades(events):
        if not _in_window(e.ts_ns, start_ns, end_ns) or e.size is None:
            continue
        # _trades() already dropped anything without a price.
        notional += cast(Decimal, e.price) * e.size
        volume += e.size
    if volume == 0:
        return None
    return notional / volume


def twap(
    events: Iterable[MarketEvent],
    *,
    interval_ns: int,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Optional[Decimal]:
    """Time-weighted average price, sampled every ``interval_ns``.

    The price of a bucket is its last trade; buckets with no trade are
    skipped rather than carried forward, because inventing a print to fill a
    silent interval is exactly the kind of plausible fiction this library
    avoids. That makes TWAP over an illiquid window an average of the
    intervals that traded, which is the honest reading of it.
    """
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")
    buckets: dict = {}
    for e in _trades(events):
        if not _in_window(e.ts_ns, start_ns, end_ns):
            continue
        buckets[e.ts_ns // interval_ns] = e.price
    if not buckets:
        return None
    prices = list(buckets.values())
    return sum(prices, Decimal(0)) / len(prices)


# -- your own prints ---------------------------------------------------------


def exclude_fills(
    market_trades: Iterable[MarketEvent],
    fills: Sequence[Fill],
    *,
    tolerance_ns: int = 0,
) -> List[MarketEvent]:
    """Remove your own executions from a public tape.

    Each fill removes at most one market trade with the same price and size
    whose timestamp is within ``tolerance_ns``. Matching is a heuristic — a
    public tape carries no identity — so a coincidental print of the same
    size at the same price and moment can be removed instead of yours. The
    error that introduces is far smaller than the one it prevents: leaving
    your prints in means benchmarking yourself partly against yourself.

    Order is preserved and the input is not modified.
    """
    if tolerance_ns < 0:
        raise ValueError("tolerance_ns must be non-negative")
    remaining = list(market_trades)
    for fill in fills:
        for i, e in enumerate(remaining):
            if (
                e.event_type is EventType.TRADE
                and e.price == fill.price
                and e.size == fill.size
                and abs(e.ts_ns - fill.ts_ns) <= tolerance_ns
            ):
                del remaining[i]
                break
    return remaining


def participation_rate(
    fills: Sequence[Fill],
    market_trades: Iterable[MarketEvent],
    *,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Optional[Decimal]:
    """Your filled size as a fraction of total market volume in the window.

    Pass the tape *including* your own prints: participation is your share of
    everything that traded. Returns ``None`` when the market volume is zero.
    """
    mine = sum(
        (f.size for f in fills if _in_window(f.ts_ns, start_ns, end_ns)),
        Decimal(0),
    )
    total = Decimal(0)
    for e in _trades(market_trades):
        if _in_window(e.ts_ns, start_ns, end_ns) and e.size is not None:
            total += e.size
    if total == 0:
        return None
    return mine / total


# -- scoring -----------------------------------------------------------------


def average_fill_price(fills: Sequence[Fill]) -> Optional[Decimal]:
    """Size-weighted average price of your fills, or ``None`` if there are none."""
    if not fills:
        return None
    volume = sum((f.size for f in fills), Decimal(0))
    if volume == 0:
        return None
    return sum((f.price * f.size for f in fills), Decimal(0)) / volume


def _single_side(fills: Sequence[Fill]) -> Side:
    sides = {f.side for f in fills}
    if len(sides) != 1:
        raise ValueError(
            "fills must all be on the same side; a single slippage number "
            "covering both buying and selling has no meaning. Split them and "
            "score each side separately."
        )
    return sides.pop()


def slippage_bps(fills: Sequence[Fill], benchmark: Decimal) -> Optional[Decimal]:
    """Performance against ``benchmark`` in basis points, positive is better.

    A buy filled below the benchmark scores positive; a sell filled above it
    scores positive. All fills must be on the same side.
    """
    if not fills:
        return None
    if benchmark <= 0:
        raise ValueError("benchmark must be positive")
    side = _single_side(fills)
    avg = average_fill_price(fills)
    if avg is None:
        return None
    edge = (benchmark - avg) if side is Side.BUY else (avg - benchmark)
    return edge / benchmark * _BPS


def implementation_shortfall_bps(
    fills: Sequence[Fill], decision_price: Decimal
) -> Optional[Decimal]:
    """Shortfall against the price when the decision was made, in basis points.

    Same sign convention as :func:`slippage_bps`: positive means the execution
    beat the decision price. This differs from slippage against VWAP in what
    it holds you responsible for — VWAP asks whether you traded well inside
    the window you chose, shortfall also charges you for the delay before you
    started.
    """
    return slippage_bps(fills, decision_price)


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """The numbers together, because reading them apart is how they mislead."""

    side: Side
    filled_size: Decimal
    average_price: Decimal
    vwap: Optional[Decimal]
    twap: Optional[Decimal]
    slippage_vs_vwap_bps: Optional[Decimal]
    shortfall_bps: Optional[Decimal]
    participation_rate: Optional[Decimal]
    own_prints_removed: int


def evaluate(
    fills: Sequence[Fill],
    market_trades: Iterable[MarketEvent],
    *,
    decision_price: Optional[Decimal] = None,
    twap_interval_ns: Optional[int] = None,
    exclude_own: bool = True,
    tolerance_ns: int = 0,
    start_ns: Optional[int] = None,
    end_ns: Optional[int] = None,
) -> Optional[ExecutionSummary]:
    """Score a set of fills against the market they traded in.

    By default the window runs from the first fill to the last, inclusive.
    That is the right frame for a worked order, and the wrong one for a
    single fill: the only print in the window is then your own, removing it
    leaves no market, and the summary correctly reports a null benchmark.
    Pass ``start_ns`` and ``end_ns`` to score against an interval you chose
    instead — the arrival window, the parent order, the whole session.

    Your own prints are removed from the tape before the benchmarks are
    computed unless ``exclude_own`` is turned off. Participation is
    deliberately measured against the *full* tape, since your share of
    volume includes your own trades.

    Returns ``None`` for an empty set of fills.
    """
    if not fills:
        return None
    side = _single_side(fills)
    ordered = sorted(fills, key=lambda f: f.ts_ns)
    start = ordered[0].ts_ns if start_ns is None else start_ns
    end = (ordered[-1].ts_ns + 1) if end_ns is None else end_ns
    if end <= start:
        raise ValueError("end_ns must be greater than start_ns")

    full_tape = list(market_trades)
    rate = participation_rate(fills, full_tape, start_ns=start, end_ns=end)

    tape = full_tape
    removed = 0
    if exclude_own:
        tape = exclude_fills(full_tape, ordered, tolerance_ns=tolerance_ns)
        removed = len(full_tape) - len(tape)

    bench = vwap(tape, start_ns=start, end_ns=end)
    t = (
        twap(tape, interval_ns=twap_interval_ns, start_ns=start, end_ns=end)
        if twap_interval_ns
        else None
    )
    avg = average_fill_price(ordered)
    assert avg is not None

    return ExecutionSummary(
        side=side,
        filled_size=sum((f.size for f in ordered), Decimal(0)),
        average_price=avg,
        vwap=bench,
        twap=t,
        slippage_vs_vwap_bps=slippage_bps(ordered, bench) if bench else None,
        shortfall_bps=(
            implementation_shortfall_bps(ordered, decision_price)
            if decision_price
            else None
        ),
        participation_rate=rate,
        own_prints_removed=removed,
    )
