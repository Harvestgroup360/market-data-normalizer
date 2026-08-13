"""Trade classification and microstructure metrics.

Most public trade tapes do not say who crossed the spread. The price and the
size are there; the aggressor is not. That single missing field is what
separates a price series from an order-flow series, and almost every
microstructure quantity — signed volume, order imbalance, effective spread,
Kyle's lambda, imbalance bars — is defined in terms of it.

This module fills the field in, using the three classification rules the
literature settled on, and then computes the quantities that become available
once it is filled::

    from mdnorm import SideRule, infer_sides, trade_imbalance

    classified = infer_sides(events, rule=SideRule.LEE_READY)
    print(trade_imbalance(classified))   # -1 (all selling) .. +1 (all buying)

The rules, in increasing order of how much they need:

``SideRule.TICK``
    Compare each trade price with the previous *different* trade price: an
    uptick is a buy, a downtick a sell, a repeat inherits the last direction.
    Needs trades only, which is why it is the fallback everywhere.

``SideRule.QUOTE``
    Compare the trade price with the prevailing mid. Above the mid is a buy,
    below is a sell, exactly at the mid is unclassifiable. Needs quotes.

``SideRule.LEE_READY``
    The quote rule, falling back to the tick rule for trades at the mid or
    with no quote yet. This is the default, and it is what Lee and Ready
    (1991) proposed.

None of the three is exact. Published accuracy on liquid US equities is
roughly 75-85%, and it degrades in fast markets, so treat an inferred side as
an estimate rather than a fact. Where a venue reports the aggressor, that
field wins: :func:`infer_sides` leaves existing sides alone unless asked not
to.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from .schema import EventType, MarketEvent, Side

__all__ = [
    "SideRule",
    "tick_rule",
    "quote_rule",
    "infer_sides",
    "signed_volume",
    "trade_imbalance",
    "effective_spreads",
    "mean_effective_spread",
    "roll_spread",
]


class SideRule(str, Enum):
    TICK = "tick"
    QUOTE = "quote"
    LEE_READY = "lee_ready"


def _is_trade(e: MarketEvent) -> bool:
    return e.event_type is EventType.TRADE and e.price is not None


def _is_quote(e: MarketEvent) -> bool:
    return (
        e.event_type is EventType.QUOTE
        and e.bid_price is not None
        and e.ask_price is not None
    )



class _QuoteBook:
    """As-of lookup: the bid/ask in force at, or before, a timestamp.

    Built once per call and queried by binary search, so classifying a file
    of trades against a file of quotes stays O(n log n) rather than
    degenerating into a scan per trade.
    """

    __slots__ = ("_ts", "_levels")

    def __init__(self, events: Sequence[MarketEvent]) -> None:
        self._ts: List[int] = []
        self._levels: List[Tuple[Decimal, Decimal]] = []
        for e in events:
            if _is_quote(e):
                self._ts.append(e.ts_ns)
                self._levels.append((e.bid_price, e.ask_price))  # type: ignore[arg-type]

    def as_of(self, ts_ns: int) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        i = bisect_right(self._ts, ts_ns)
        if i == 0:
            return (None, None)
        return self._levels[i - 1]


# -- the two primitive rules -------------------------------------------------


def quote_rule(
    price: Decimal, bid: Optional[Decimal], ask: Optional[Decimal]
) -> Optional[Side]:
    """Classify one trade against a quote. ``None`` at the mid or unquoted.

    A trade printed above the mid is assumed to have hit the ask, and is
    therefore buyer-initiated; below the mid, seller-initiated.
    """
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2
    if price > mid:
        return Side.BUY
    if price < mid:
        return Side.SELL
    return None


def tick_rule(prices: Sequence[Decimal]) -> List[Optional[Side]]:
    """Classify a price sequence by direction of change.

    An uptick is buyer-initiated, a downtick seller-initiated, and a trade at
    an unchanged price inherits the direction of the last change (a
    "zero-uptick" is a buy, a "zero-downtick" a sell). Leading trades before
    any price change are unclassifiable and come back as ``None``.
    """
    out: List[Optional[Side]] = []
    last_price: Optional[Decimal] = None
    last_dir: Optional[Side] = None
    for p in prices:
        if last_price is None or p == last_price:
            out.append(last_dir)
        else:
            last_dir = Side.BUY if p > last_price else Side.SELL
            out.append(last_dir)
        last_price = p
    return out


# -- applying a rule to a stream --------------------------------------------


def infer_sides(
    events: Iterable[MarketEvent],
    *,
    rule: SideRule = SideRule.LEE_READY,
    lag_ns: int = 0,
    overwrite: bool = False,
) -> List[MarketEvent]:
    """Fill in the aggressor side on trades that do not carry one.

    Quotes in the stream supply the prevailing bid and ask; they are passed
    through unchanged. Trades that already have a ``side`` are left alone
    unless ``overwrite`` is set — a venue-reported aggressor beats any
    inference.

    ``lag_ns`` matches a trade against the quote in force that many
    nanoseconds earlier. Lee and Ready used five seconds to compensate for
    trade reporting delays on 1980s tapes; on modern timestamped feeds 0 is
    usually right, which is the default.

    Input order is preserved. Trades that remain unclassifiable keep
    ``side=None`` rather than being guessed at.
    """
    if lag_ns < 0:
        raise ValueError("lag_ns must be non-negative")

    items = list(events)
    order = sorted(range(len(items)), key=lambda i: items[i].ts_ns)
    book = _QuoteBook([items[i] for i in order])

    resolved: dict = {}
    last_price: Optional[Decimal] = None
    last_dir: Optional[Side] = None

    for i in order:
        e = items[i]
        if not _is_trade(e):
            continue
        price: Decimal = e.price  # type: ignore[assignment]

        # Tick state advances on every trade, whichever rule is in force, so
        # the fallback is always ready.
        if last_price is not None and price != last_price:
            last_dir = Side.BUY if price > last_price else Side.SELL
        tick_side = last_dir
        last_price = price

        if e.side is not None and not overwrite:
            continue

        if rule is SideRule.TICK:
            side = tick_side
        else:
            bid, ask = book.as_of(e.ts_ns - lag_ns)
            side = quote_rule(price, bid, ask)
            if side is None and rule is SideRule.LEE_READY:
                side = tick_side

        if side is not None:
            resolved[i] = side

    return [
        replace(e, side=resolved[i]) if i in resolved else e
        for i, e in enumerate(items)
    ]


# -- quantities that need a side --------------------------------------------


def signed_volume(events: Iterable[MarketEvent]) -> Decimal:
    """Buy volume minus sell volume. Unclassified trades contribute nothing."""
    total = Decimal(0)
    for e in events:
        if not _is_trade(e) or e.side is None or e.size is None:
            continue
        total += e.size if e.side is Side.BUY else -e.size
    return total


def trade_imbalance(events: Iterable[MarketEvent]) -> Optional[Decimal]:
    """Signed volume as a fraction of classified volume, in ``[-1, 1]``.

    ``+1`` means every classified trade was buyer-initiated, ``-1`` every one
    seller-initiated, ``0`` a balanced tape. Returns ``None`` when no trade
    carries both a side and a size, since the ratio is undefined rather than
    zero — an important distinction when a feed simply lacks the field.
    """
    signed = Decimal(0)
    total = Decimal(0)
    for e in events:
        if not _is_trade(e) or e.side is None or e.size is None:
            continue
        signed += e.size if e.side is Side.BUY else -e.size
        total += e.size
    if total == 0:
        return None
    return signed / total


def effective_spreads(
    events: Iterable[MarketEvent], *, lag_ns: int = 0
) -> List[Decimal]:
    """Effective spread for every trade with a prevailing quote.

    The effective spread is ``2 * |price - mid|`` — what the trade actually
    paid to cross, which differs from the posted spread whenever a trade
    executes inside or outside the quote. Trades with no quote in force are
    skipped rather than counted as zero.
    """
    if lag_ns < 0:
        raise ValueError("lag_ns must be non-negative")
    out: List[Decimal] = []
    ordered = sorted(events, key=lambda e: e.ts_ns)
    book = _QuoteBook(ordered)
    for e in ordered:
        if not _is_trade(e):
            continue
        b, a = book.as_of(e.ts_ns - lag_ns)
        if b is None or a is None:
            continue
        mid = (b + a) / 2
        out.append(abs(e.price - mid) * 2)  # type: ignore[operator]
    return out


def mean_effective_spread(
    events: Iterable[MarketEvent], *, lag_ns: int = 0
) -> Optional[Decimal]:
    """Average effective spread, or ``None`` if no trade had a quote."""
    spreads = effective_spreads(events, lag_ns=lag_ns)
    if not spreads:
        return None
    return sum(spreads, Decimal(0)) / len(spreads)


def roll_spread(events: Iterable[MarketEvent]) -> Optional[Decimal]:
    """Roll's (1984) implied effective spread, from trade prices alone.

    Bid-ask bounce makes consecutive price changes negatively autocovariant,
    and Roll showed the spread can be recovered from that covariance as
    ``2 * sqrt(-cov(dP_t, dP_t-1))``. It needs no quotes and no side, which
    makes it the sanity check on everything else in this module: if the
    inferred sides and the posted spreads tell one story and Roll tells
    another, one of the inputs is wrong.

    The model assumes trade signs are serially uncorrelated. Where they are
    not — a strictly alternating tape being the extreme case — the estimate
    is biased upward, so read a Roll spread that badly exceeds the posted one
    as evidence about the sign process rather than about liquidity.

    Returns ``None`` when the covariance is non-negative — a real outcome on
    trending or thin data, where the estimator is simply undefined, and one
    that should be reported rather than clamped to zero.
    """
    prices = [e.price for e in sorted(events, key=lambda e: e.ts_ns) if _is_trade(e)]
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    if len(diffs) < 3:
        return None

    n = len(diffs) - 1
    mean_a = sum(diffs[1:], Decimal(0)) / n
    mean_b = sum(diffs[:-1], Decimal(0)) / n
    cov = sum(
        ((diffs[i] - mean_a) * (diffs[i - 1] - mean_b) for i in range(1, len(diffs))),
        Decimal(0),
    ) / n
    if cov >= 0:
        return None
    try:
        return (-cov).sqrt() * 2
    except InvalidOperation:  # pragma: no cover - guarded by the sign check
        return None
