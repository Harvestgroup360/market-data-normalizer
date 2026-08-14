"""Limit order book reconstruction from incremental updates.

Exchanges do not send you a book. They send you a snapshot and then a stream of
deltas — "there are now 3.5 at 100.25", "there is nothing left at 100.20" — and
the book only exists if you apply them, in order, without missing one. Get that
wrong and every quantity downstream is wrong with it: the mid, the spread,
effective spread, book imbalance, and any classification rule that compares a
trade price against the prevailing quote.

This module maintains that book, and is deliberate about the two failure modes
that make a reconstructed book silently untrue::

    from mdnorm import BookDelta, OrderBook, Side

    book = OrderBook("BTC-USD", "binance")
    book.apply_snapshot(ts, bids=[(D("100"), D("2"))], asks=[(D("101"), D("3"))], seq=10)
    book.apply(BookDelta(ts + 1, Side.BUY, D("100.5"), D("1"), seq=11))
    print(book.best_bid, book.spread)

**Sequence gaps.** A missed message leaves the book permanently wrong in a way
no later update repairs, and nothing about the resulting numbers looks
suspicious. :class:`OrderBook` tracks the sequence number and raises
:class:`SequenceGapError` the moment one is skipped, because the honest
response to a gap is to stop and resynchronise from a fresh snapshot, not to
carry on with a book that is quietly missing a level. Pass
``strict_sequence=False`` for feeds that do not number their messages.

**Crossed books.** A bid at or above the ask is not a market state; it is a
symptom — a dropped delete, a stale snapshot, two venues merged by mistake.
The book exposes :attr:`is_crossed` rather than silently normalising it away.

A reconstructed book feeds the rest of the library through :meth:`to_quote`,
which emits a top-of-book :class:`~mdnorm.schema.MarketEvent` quote that
:mod:`mdnorm.micro` and everything else already understand.
"""
from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .schema import EventType, MarketEvent, Side

__all__ = [
    "BookDelta",
    "OrderBook",
    "SequenceGapError",
    "replay_book",
]

Level = Tuple[Decimal, Decimal]


class SequenceGapError(ValueError):
    """A delta arrived out of order, so the book can no longer be trusted."""


@dataclass(frozen=True, slots=True)
class BookDelta:
    """One incremental change to a price level.

    ``size`` is the new total resting quantity at ``price``, not a difference:
    this is how essentially every exchange delta feed is defined. A ``size`` of
    zero removes the level.
    """

    ts_ns: int
    side: Side
    price: Decimal
    size: Decimal
    seq: Optional[int] = None

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.size < 0:
            raise ValueError("size must not be negative (0 removes the level)")


class _Side:
    """One side of the book: price -> size, plus prices kept sorted ascending."""

    __slots__ = ("_sizes", "_prices")

    def __init__(self) -> None:
        self._sizes: Dict[Decimal, Decimal] = {}
        self._prices: List[Decimal] = []

    def set(self, price: Decimal, size: Decimal) -> None:
        if size == 0:
            if self._sizes.pop(price, None) is not None:
                i = bisect_left(self._prices, price)
                if i < len(self._prices) and self._prices[i] == price:
                    del self._prices[i]
            return
        if price not in self._sizes:
            insort(self._prices, price)
        self._sizes[price] = size

    def clear(self) -> None:
        self._sizes.clear()
        self._prices.clear()

    def __len__(self) -> int:
        return len(self._prices)

    def best(self, *, highest: bool) -> Optional[Level]:
        if not self._prices:
            return None
        p = self._prices[-1] if highest else self._prices[0]
        return (p, self._sizes[p])

    def top(self, n: int, *, highest: bool) -> List[Level]:
        prices = self._prices[::-1] if highest else self._prices
        return [(p, self._sizes[p]) for p in prices[:n]]


class OrderBook:
    """A single-symbol limit order book rebuilt from snapshots and deltas."""

    __slots__ = ("symbol", "venue", "max_depth", "strict_sequence",
                 "_bids", "_asks", "_ts_ns", "_seq")

    def __init__(
        self,
        symbol: str,
        venue: str,
        *,
        max_depth: Optional[int] = None,
        strict_sequence: bool = True,
    ) -> None:
        if max_depth is not None and max_depth <= 0:
            raise ValueError("max_depth must be positive")
        self.symbol = symbol
        self.venue = venue
        self.max_depth = max_depth
        self.strict_sequence = strict_sequence
        self._bids = _Side()
        self._asks = _Side()
        self._ts_ns: int = 0
        self._seq: Optional[int] = None

    # -- state ------------------------------------------------------------

    @property
    def ts_ns(self) -> int:
        """Timestamp of the last applied update."""
        return self._ts_ns

    @property
    def seq(self) -> Optional[int]:
        """Sequence number of the last applied update, if the feed numbers them."""
        return self._seq

    @property
    def best_bid(self) -> Optional[Level]:
        return self._bids.best(highest=True)

    @property
    def best_ask(self) -> Optional[Level]:
        return self._asks.best(highest=False)

    @property
    def mid(self) -> Optional[Decimal]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return (b[0] + a[0]) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        b, a = self.best_bid, self.best_ask
        if b is None or a is None:
            return None
        return a[0] - b[0]

    @property
    def is_crossed(self) -> bool:
        """True when the best bid is at or above the best ask.

        Not a market state — a symptom of a dropped update, a stale snapshot,
        or two venues merged by accident. Surfaced rather than smoothed over.
        """
        b, a = self.best_bid, self.best_ask
        return b is not None and a is not None and b[0] >= a[0]

    def depth(self, side: Side, levels: int = 5) -> List[Level]:
        """The best ``levels`` price levels on ``side``, best first."""
        if levels <= 0:
            raise ValueError("levels must be positive")
        if side is Side.BUY:
            return self._bids.top(levels, highest=True)
        return self._asks.top(levels, highest=False)

    def imbalance(self, levels: int = 1) -> Optional[Decimal]:
        """Resting-size imbalance over the top ``levels``, in ``[-1, 1]``.

        Positive means more size resting on the bid. This is *book* imbalance —
        what is waiting — and should not be confused with the trade imbalance
        in :mod:`mdnorm.micro`, which measures what actually executed. Returns
        ``None`` when either side is empty, since the ratio is undefined rather
        than extreme.
        """
        bids = self.depth(Side.BUY, levels)
        asks = self.depth(Side.SELL, levels)
        if not bids or not asks:
            return None
        bid_size = sum((s for _, s in bids), Decimal(0))
        ask_size = sum((s for _, s in asks), Decimal(0))
        total = bid_size + ask_size
        if total == 0:
            return None
        return (bid_size - ask_size) / total

    # -- mutation ---------------------------------------------------------

    def apply_snapshot(
        self,
        ts_ns: int,
        bids: Sequence[Level],
        asks: Sequence[Level],
        *,
        seq: Optional[int] = None,
    ) -> None:
        """Replace the whole book. This is also how you resynchronise."""
        self._bids.clear()
        self._asks.clear()
        for price, size in bids:
            self._bids.set(price, size)
        for price, size in asks:
            self._asks.set(price, size)
        self._ts_ns = ts_ns
        self._seq = seq
        self._trim()

    def apply(self, delta: BookDelta) -> None:
        """Apply one delta, checking the sequence first."""
        self._check_sequence(delta.seq)
        side = self._bids if delta.side is Side.BUY else self._asks
        side.set(delta.price, delta.size)
        self._ts_ns = delta.ts_ns
        if delta.seq is not None:
            self._seq = delta.seq
        self._trim()

    def apply_many(self, deltas: Iterable[BookDelta]) -> None:
        for d in deltas:
            self.apply(d)

    def _check_sequence(self, seq: Optional[int]) -> None:
        if not self.strict_sequence or seq is None or self._seq is None:
            return
        expected = self._seq + 1
        if seq == expected:
            return
        if seq <= self._seq:
            raise SequenceGapError(
                f"{self.symbol}@{self.venue}: delta seq {seq} is not newer than "
                f"the applied seq {self._seq} (duplicate or replayed message)"
            )
        raise SequenceGapError(
            f"{self.symbol}@{self.venue}: sequence gap, expected {expected} but "
            f"received {seq} — {seq - expected} update(s) missing. The book is "
            f"no longer reliable; resynchronise from a snapshot."
        )

    def _trim(self) -> None:
        """Drop levels beyond ``max_depth`` on each side.

        Only meaningful for feeds that publish a fixed depth; an unbounded book
        keeps everything.
        """
        if self.max_depth is None:
            return
        for side, highest in ((self._bids, True), (self._asks, False)):
            while len(side) > self.max_depth:
                worst = side.best(highest=not highest)
                if worst is None:
                    break
                side.set(worst[0], Decimal(0))

    # -- output -----------------------------------------------------------

    def to_quote(self) -> Optional[MarketEvent]:
        """Top of book as a :class:`~mdnorm.schema.MarketEvent` quote.

        This is the bridge to the rest of the library: once a book is a quote,
        session filtering, trade classification and effective spreads all work
        on it unchanged. Returns ``None`` while both sides are empty.
        """
        b, a = self.best_bid, self.best_ask
        if b is None and a is None:
            return None
        return MarketEvent(
            symbol=self.symbol,
            venue=self.venue,
            event_type=EventType.QUOTE,
            ts_ns=self._ts_ns,
            bid_price=b[0] if b else None,
            bid_size=b[1] if b else None,
            ask_price=a[0] if a else None,
            ask_size=a[1] if a else None,
        )


def replay_book(
    book: OrderBook,
    deltas: Iterable[BookDelta],
    *,
    top_of_book_only: bool = True,
) -> Iterator[MarketEvent]:
    """Apply ``deltas`` in order, yielding a quote whenever the top changes.

    With ``top_of_book_only`` (the default) a delta deep in the book updates
    the state but emits nothing, which is what you want when the output feeds
    quote-based analysis: one event per actual change in the best bid or offer,
    rather than one per message. Set it to ``False`` to emit after every delta.

    The book is mutated in place, so it can be inspected afterwards.
    """
    # Seed from the book's current top, not from nothing: replaying a deep
    # delta onto an already-populated book must emit nothing, and it would
    # emit a spurious first quote if the comparison started empty.
    last: Optional[Tuple[Optional[Level], Optional[Level]]] = (
        book.best_bid, book.best_ask
    )
    for delta in deltas:
        book.apply(delta)
        top = (book.best_bid, book.best_ask)
        if top_of_book_only and top == last:
            continue
        last = top
        quote = book.to_quote()
        if quote is not None:
            yield quote
