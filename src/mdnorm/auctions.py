"""A crossed auction is not a trade, and it is not a point on the tape.

Most of a day's shares change hands in the continuous session, and a
surprising share of them do not: the opening and closing crosses are single
prints, at a single price, aggregating orders that never met each other in a
book. Every statistic that treats them as ordinary trades is wrong in a
direction that flatters::

    from mdnorm import auction_windows, auction_report, vwap_gap

    windows = auction_windows(days, calendar)   # from the calendar, not a constant
    auction_report(trades, windows).volume_share
    vwap_gap(trades, windows).difference_bps

**An auction print has no aggressor.** Nobody crossed a spread; a clearing
price was computed. Handing it to the tick rule or the quote rule produces a
side, because those functions always produce a side, and that side is an
artefact of where the previous continuous print happened to sit.

**The close is a benchmark price and a microstructure outlier at the same
time.** It is the right number for marking a book and the wrong one for a
spread estimate, a realised-volatility figure built from tick returns, or an
order-flow imbalance. Which of those you are doing is not something a library
can infer, so this one splits the two populations and leaves the choice where
it belongs.

**Execution benchmarks are where it costs money.** A VWAP computed with the
closing cross in it is dominated by one print. A strategy that never trades
the auction, measured against that benchmark, is being scored against a price
it could not have obtained — and a strategy that only trades the auction beats
it by construction. :func:`vwap_gap` reports both numbers and the distance
between them.

**Auctions are not inferred here.** No condition-code guessing, no rule that a
print ten times the median size must be a cross. On a busy day that rule
reclassifies ordinary block trades, and the resulting statistic is a property
of the threshold rather than of the market. Either you supply the windows —
and :func:`auction_windows` derives them from a trading calendar, so a
half-day's cross lands where the venue actually closed — or the report comes
back saying it has no windows to work with.

Nothing here deletes anything. :func:`split_auctions` hands back both halves.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple

from .calendars import TradingCalendar
from .schema import EventType, MarketEvent

__all__ = [
    "AuctionKind",
    "AuctionWindow",
    "AuctionReport",
    "VwapGap",
    "auction_windows",
    "in_auction",
    "split_auctions",
    "exclude_auctions",
    "auction_report",
    "vwap_gap",
]

_ZERO = Decimal(0)


class AuctionKind(str, Enum):
    """Which cross a window covers."""

    OPEN = "open"
    CLOSE = "close"
    INTRADAY = "intraday"


@dataclass(frozen=True, slots=True)
class AuctionWindow:
    """A half-open span ``[start_ns, end_ns)`` covering one cross.

    Half-open like every other interval in this library, which matters more
    here than usual: the closing cross prints at or just after the bell, and a
    window that excluded its own end would drop the print it exists for.
    """

    kind: AuctionKind
    start_ns: int
    end_ns: int
    day: Optional[date] = None

    def __post_init__(self) -> None:
        if self.end_ns <= self.start_ns:
            raise ValueError("an auction window must cover a positive span")

    def __contains__(self, ts_ns: int) -> bool:
        return self.start_ns <= ts_ns < self.end_ns


@dataclass(frozen=True, slots=True)
class AuctionReport:
    """How much of a day went through the crosses rather than the book."""

    trades: int
    auction_trades: int
    volume: Decimal
    auction_volume: Decimal
    notional: Decimal
    auction_notional: Decimal
    largest_print: Decimal
    windows: int

    @property
    def volume_share(self) -> Optional[Decimal]:
        """Share of traded volume that crossed in an auction."""
        if self.volume == 0:
            return None
        return self.auction_volume / self.volume

    @property
    def notional_share(self) -> Optional[Decimal]:
        """Share of traded notional that crossed in an auction.

        Usually larger than the volume share, because the crosses print at
        the extremes of the day's range rather than at its average.
        """
        if self.notional == 0:
            return None
        return self.auction_notional / self.notional

    @property
    def largest_print_share(self) -> Optional[Decimal]:
        """The single biggest print as a share of the day's volume.

        Worth looking at even with no windows supplied: a feed where one
        print is a tenth of the day has a cross in it whether or not anything
        has been told where.
        """
        if self.volume == 0:
            return None
        return self.largest_print / self.volume


@dataclass(frozen=True, slots=True)
class VwapGap:
    """The same benchmark, computed with and without the crosses."""

    with_auctions: Optional[Decimal]
    without_auctions: Optional[Decimal]
    auction_only: Optional[Decimal]
    auction_volume_share: Optional[Decimal]

    @property
    def difference(self) -> Optional[Decimal]:
        """Benchmark including the crosses minus the benchmark without them."""
        if self.with_auctions is None or self.without_auctions is None:
            return None
        return self.with_auctions - self.without_auctions

    @property
    def difference_bps(self) -> Optional[Decimal]:
        """That difference in basis points of the continuous benchmark.

        This is the number an execution report is exposed to. It is small on
        a quiet name and not small on one whose close is a third of the day,
        and either way it is a property of the benchmark rather than of
        anybody's trading.
        """
        diff = self.difference
        if diff is None or not self.without_auctions:
            return None
        return diff / self.without_auctions * 10_000


def auction_windows(
    days: Iterable[date],
    calendar: TradingCalendar,
    *,
    open_ns: int = 0,
    close_ns: int = 0,
    lead_ns: int = 0,
) -> List[AuctionWindow]:
    """Build per-day auction windows from a calendar.

    ``open_ns`` and ``close_ns`` are how long after the open and before the
    close each cross may print in; both default to zero, which produces the
    smallest windows that can hold a print stamped exactly at the bell. The
    defaults are deliberately not "thirty seconds, everybody uses that" —
    that constant differs by venue and by decade, and a wrong one silently
    moves ordinary continuous prints into the auction bucket.

    ``lead_ns`` extends the closing window past the bell, for venues whose
    cross is published a moment after the session formally ends.

    The bounds come from the calendar, so a day that closed early has its
    closing cross where the venue actually closed rather than where the
    regular session would have put it. Days the calendar does not cover, or
    says did not trade, produce no windows.
    """
    if open_ns < 0 or close_ns < 0 or lead_ns < 0:
        raise ValueError("window extents must be non-negative")

    out: List[AuctionWindow] = []
    for day in days:
        try:
            span = calendar.session_on(day)
        except (KeyError, ValueError):
            continue
        if span is None:
            continue
        start, end = span
        out.append(AuctionWindow(AuctionKind.OPEN, start, start + open_ns + 1,
                                 day=day))
        out.append(AuctionWindow(AuctionKind.CLOSE, end - close_ns,
                                 end + lead_ns + 1, day=day))
    return out


def in_auction(ts_ns: int, windows: Sequence[AuctionWindow]) -> Optional[AuctionKind]:
    """Which cross a timestamp falls in, or ``None`` for the continuous session."""
    for w in windows:
        if ts_ns in w:
            return w.kind
    return None


def _trades(events: Iterable[MarketEvent]) -> List[MarketEvent]:
    return [e for e in events
            if e.event_type is EventType.TRADE and e.price is not None]


def split_auctions(
    events: Iterable[MarketEvent],
    windows: Sequence[AuctionWindow],
) -> Tuple[List[MarketEvent], List[MarketEvent]]:
    """Return ``(continuous, auction)``, preserving input order in both.

    Non-trade events stay with the continuous side: a quote inside a closing
    window is still a quote, and dropping it would leave the continuous
    stream without the state it needs at the end of the day.
    """
    continuous: List[MarketEvent] = []
    auction: List[MarketEvent] = []
    for e in events:
        if (e.event_type is EventType.TRADE
                and in_auction(e.ts_ns, windows) is not None):
            auction.append(e)
        else:
            continuous.append(e)
    return continuous, auction


def exclude_auctions(
    events: Iterable[MarketEvent],
    windows: Sequence[AuctionWindow],
) -> List[MarketEvent]:
    """The continuous half alone, for when the split is not needed."""
    return split_auctions(events, windows)[0]


def auction_report(
    events: Iterable[MarketEvent],
    windows: Sequence[AuctionWindow] = (),
) -> AuctionReport:
    """Count what went through the crosses against what went through the book.

    With no windows the counts come back at zero and the shares at zero, but
    ``largest_print`` is still measured — the one figure here that needs no
    external information, and often the one that tells you the file has a
    cross in it.
    """
    trades = _trades(events)
    volume = _ZERO
    notional = _ZERO
    a_volume = _ZERO
    a_notional = _ZERO
    a_count = 0
    largest = _ZERO

    for e in trades:
        if e.size is None:
            continue
        price: Decimal = e.price  # type: ignore[assignment]
        volume += e.size
        notional += price * e.size
        if e.size > largest:
            largest = e.size
        if windows and in_auction(e.ts_ns, windows) is not None:
            a_count += 1
            a_volume += e.size
            a_notional += price * e.size

    return AuctionReport(trades=len(trades), auction_trades=a_count,
                         volume=volume, auction_volume=a_volume,
                         notional=notional, auction_notional=a_notional,
                         largest_print=largest, windows=len(windows))


def _vwap(trades: Sequence[MarketEvent]) -> Optional[Decimal]:
    volume = _ZERO
    notional = _ZERO
    for e in trades:
        if e.size is None:
            continue
        price: Decimal = e.price  # type: ignore[assignment]
        notional += price * e.size
        volume += e.size
    return notional / volume if volume else None


def vwap_gap(
    events: Iterable[MarketEvent],
    windows: Sequence[AuctionWindow],
) -> VwapGap:
    """Compute the volume-weighted price with the crosses in and out.

    Both figures are real benchmarks and neither is the correct one in
    general: including the crosses is right for a fund marking against the
    official close, excluding them is right for scoring a strategy that only
    ever traded the book. The distance between them is what an execution
    report is silently exposed to when nobody states which was used.
    """
    continuous, auction = split_auctions(events, windows)
    trades_all = _trades(events)
    a_trades = _trades(auction)

    total_volume = sum((e.size for e in trades_all if e.size is not None),
                       _ZERO)
    a_volume = sum((e.size for e in a_trades if e.size is not None), _ZERO)

    return VwapGap(
        with_auctions=_vwap(trades_all),
        without_auctions=_vwap(_trades(continuous)),
        auction_only=_vwap(a_trades),
        auction_volume_share=(a_volume / total_volume if total_volume
                              else None),
    )
