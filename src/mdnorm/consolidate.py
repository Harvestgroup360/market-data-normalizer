"""Consolidating quotes from several venues into one best bid and offer.

An instrument trades in more than one place, and "the price" is then a
question rather than a fact. The consolidated top of book — the highest bid
and the lowest offer across venues — is the honest answer, and building it is
where three problems live that a naive `max()` over venues will not tell you
about::

    from mdnorm import consolidate

    top = consolidate(quotes, max_age_ns=2_000_000_000)   # 2s staleness cutoff
    print(top[-1].venue, top[-1].bid_price, top[-1].ask_price)

**A venue that goes quiet keeps voting.** If a feed stops updating — a
disconnect, a halt, a dropped subscription — its last quote sits in the
consolidation forever, and a stale price is very often the *best* price, so
the dead venue ends up setting the top of book. This is the failure mode that
produces a consolidated feed that looks excellent and is fiction. Pass
``max_age_ns`` and a venue that has not spoken within that window stops
counting. It defaults to ``None`` for the case where the caller has already
handled it, but on live multi-venue data leaving it unset is almost always a
mistake.

**A consolidated book can appear crossed.** A bid on one venue above the offer
on another looks like free money and is usually clock skew: the two feeds are
timestamped by different machines with different delays. :attr:`is_crossed`
reports it and :attr:`crossed_updates` counts how often it happened, because
the useful response is to investigate the clocks, not to trade the spread.

**Ties need a rule.** When two venues quote the same best price the tie is
broken by size, then by venue name, so the same input always produces the same
output. Documented rather than incidental, because a consolidation that
reshuffles under equal prices makes downstream diffs meaningless.

Which venue sets the price is itself a measurement, so the consolidator counts
it: :attr:`leadership` reports how many updates each venue spent at the top of
each side.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import EventType, MarketEvent

__all__ = [
    "CONSOLIDATED",
    "VenueTop",
    "Consolidator",
    "consolidate",
]

#: Venue label carried by consolidated events.
CONSOLIDATED = "consolidated"


@dataclass(frozen=True, slots=True)
class VenueTop:
    """The best price on one side, and which venue is showing it."""

    venue: str
    price: Decimal
    size: Optional[Decimal]
    ts_ns: int


@dataclass(frozen=True, slots=True)
class _Quote:
    ts_ns: int
    bid_price: Optional[Decimal]
    bid_size: Optional[Decimal]
    ask_price: Optional[Decimal]
    ask_size: Optional[Decimal]


class Consolidator:
    """Maintains the best bid and offer across venues for one symbol."""

    __slots__ = ("symbol", "max_age_ns", "_quotes", "_ts_ns",
                 "leadership", "crossed_updates")

    def __init__(self, symbol: str, *, max_age_ns: Optional[int] = None) -> None:
        if max_age_ns is not None and max_age_ns <= 0:
            raise ValueError("max_age_ns must be positive")
        self.symbol = symbol
        self.max_age_ns = max_age_ns
        self._quotes: Dict[str, _Quote] = {}
        self._ts_ns: int = 0
        #: venue -> {"bid": n, "ask": n}: how many updates it led each side.
        self.leadership: Dict[str, Dict[str, int]] = {}
        #: How many updates produced a crossed consolidated book.
        self.crossed_updates: int = 0

    # -- state -------------------------------------------------------------

    @property
    def ts_ns(self) -> int:
        return self._ts_ns

    @property
    def venues(self) -> List[str]:
        """Venues that have quoted at least once, in first-seen order."""
        return list(self._quotes)

    def fresh_venues(self, ts_ns: Optional[int] = None) -> List[str]:
        """Venues whose last quote is inside the staleness window."""
        now = self._ts_ns if ts_ns is None else ts_ns
        return [v for v, q in self._quotes.items() if self._is_fresh(q, now)]

    def stale_venues(self, ts_ns: Optional[int] = None) -> List[str]:
        """Venues that have gone quiet for longer than ``max_age_ns``.

        Always empty when no staleness window is configured — which is the
        point of configuring one.
        """
        now = self._ts_ns if ts_ns is None else ts_ns
        return [v for v, q in self._quotes.items() if not self._is_fresh(q, now)]

    def _is_fresh(self, q: _Quote, now: int) -> bool:
        if self.max_age_ns is None:
            return True
        return now - q.ts_ns <= self.max_age_ns

    @property
    def best_bid(self) -> Optional[VenueTop]:
        return self._best(bid=True)

    @property
    def best_ask(self) -> Optional[VenueTop]:
        return self._best(bid=False)

    def _best(self, *, bid: bool) -> Optional[VenueTop]:
        candidates: List[Tuple[Decimal, Optional[Decimal], str, int]] = []
        for venue, q in self._quotes.items():
            if not self._is_fresh(q, self._ts_ns):
                continue
            price = q.bid_price if bid else q.ask_price
            if price is None:
                continue
            size = q.bid_size if bid else q.ask_size
            candidates.append((price, size, venue, q.ts_ns))
        if not candidates:
            return None
        # Price first; then larger size; then venue name, so the result is
        # deterministic instead of dependent on dict ordering.
        def key(c):
            price, size, venue, _ = c
            s = size if size is not None else Decimal(0)
            return (-price, -s, venue) if bid else (price, -s, venue)

        price, size, venue, ts = min(candidates, key=key)
        return VenueTop(venue=venue, price=price, size=size, ts_ns=ts)

    @property
    def spread(self) -> Optional[Decimal]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a.price - b.price

    @property
    def is_crossed(self) -> bool:
        """Best bid at or above best offer — usually clock skew, not alpha."""
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b.price >= a.price

    # -- mutation ----------------------------------------------------------

    def update(self, quote: MarketEvent) -> Optional[MarketEvent]:
        """Take one venue quote in; return the consolidated top if it moved.

        Non-quote events and quotes for another symbol are ignored. Returns
        ``None`` when the consolidated best bid and offer are unchanged — a
        venue repeating itself produces no output — so the result is one event
        per actual change rather than one per input.
        """
        if quote.event_type is not EventType.QUOTE or quote.symbol != self.symbol:
            return None

        before = self._top_values()
        self._quotes[quote.venue] = _Quote(
            ts_ns=quote.ts_ns,
            bid_price=quote.bid_price, bid_size=quote.bid_size,
            ask_price=quote.ask_price, ask_size=quote.ask_size,
        )
        self._ts_ns = max(self._ts_ns, quote.ts_ns)

        bid, ask = self.best_bid, self.best_ask
        for side, top in (("bid", bid), ("ask", ask)):
            if top is not None:
                slot = self.leadership.setdefault(top.venue, {"bid": 0, "ask": 0})
                slot[side] += 1
        if self.is_crossed:
            self.crossed_updates += 1
        if self._top_values() == before:
            return None
        return self.to_quote()

    def _top_values(self):
        """The four numbers a consolidated event carries.

        Emission is decided on these rather than on the :class:`VenueTop`
        objects, because a venue re-sending an identical quote changes the
        source timestamp without changing anything a consumer would see.
        """
        b, a = self.best_bid, self.best_ask
        return (
            (b.price, b.size) if b else None,
            (a.price, a.size) if a else None,
        )

    def to_quote(self) -> Optional[MarketEvent]:
        """The consolidated top as a :class:`~mdnorm.schema.MarketEvent`.

        The venue is :data:`CONSOLIDATED` rather than any single venue, since
        the two sides may come from different places.
        """
        b, a = self.best_bid, self.best_ask
        if b is None and a is None:
            return None
        return MarketEvent(
            symbol=self.symbol,
            venue=CONSOLIDATED,
            event_type=EventType.QUOTE,
            ts_ns=self._ts_ns,
            bid_price=b.price if b else None,
            bid_size=b.size if b else None,
            ask_price=a.price if a else None,
            ask_size=a.size if a else None,
        )


def consolidate(
    quotes: Iterable[MarketEvent],
    *,
    symbol: Optional[str] = None,
    max_age_ns: Optional[int] = None,
) -> List[MarketEvent]:
    """Consolidate a multi-venue quote stream into best-bid-and-offer events.

    Quotes are processed in timestamp order and one event is emitted per
    change in the consolidated top. ``symbol`` defaults to the first quoted
    symbol; other symbols in the stream are ignored rather than merged, since
    consolidating two different instruments produces a number with no meaning.

    See :class:`Consolidator` for the staleness, crossing and tie-break rules
    that make this more than a maximum over venues.
    """
    ordered = sorted(
        (e for e in quotes if e.event_type is EventType.QUOTE),
        key=lambda e: e.ts_ns,
    )
    if not ordered:
        return []
    target = symbol if symbol is not None else ordered[0].symbol
    book = Consolidator(target, max_age_ns=max_age_ns)
    out: List[MarketEvent] = []
    for q in ordered:
        top = book.update(q)
        if top is not None:
            out.append(top)
    return out
