"""Corporate actions and contract rolls: back-adjusted price series.

A raw price series is not continuous. A 4-for-1 split divides the printed
price by four overnight; a cash dividend drops it by the amount paid; rolling
from an expiring futures contract to the next one steps the price by the
spread between them. None of these are market moves, but every one of them
looks like a return to code that just takes ``close[t] / close[t-1]``.

The standard repair is *back-adjustment*: leave the most recent segment at the
prices that actually printed and restate everything before each event so the
joins are seamless. That is what this module does, for the three events that
cause the damage::

    from decimal import Decimal
    from mdnorm import adjust_bars, split, dividend

    actions = [
        split(iso_to_ns("2026-06-09T00:00:00Z"), Decimal("4")),
        dividend(iso_to_ns("2026-05-09T00:00:00Z"), Decimal("0.25")),
    ]
    clean = adjust_bars(bars, actions)

Two conventions are supported, selected with ``method``:

``AdjustMethod.RATIO``
    Multiply earlier prices by a factor. Returns are preserved exactly and
    prices stay positive, which is why it is the default and the right choice
    for equities.

``AdjustMethod.DIFFERENCE``
    Add a constant offset to earlier prices. Price *differences* are preserved
    exactly, which matters for futures spreads, but a long back-adjusted
    history can reach zero or go negative. That is an accepted artefact of the
    convention rather than a bug; :func:`mdnorm.find_issues` will flag the
    non-positive prices if you need to know.

Splits are always ratio-adjusted — a "difference-adjusted" split is not a
meaningful object. ``method`` therefore applies to dividends and rolls only.

Adjustment is applied to data *strictly before* each action's timestamp, so an
action stamped at the ex-date or the roll moment leaves that session's own
prints untouched.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Iterable, List, Optional, Sequence, Tuple, cast

from .bars import Bar
from .fileio import open_text
from .schema import MarketEvent
from .timeutil import epoch_to_ns, iso_to_ns

__all__ = [
    "ActionKind",
    "AdjustMethod",
    "Action",
    "split",
    "dividend",
    "roll",
    "adjustment_at",
    "adjust_events",
    "adjust_bars",
    "read_actions_csv",
]


class ActionKind(str, Enum):
    SPLIT = "split"
    DIVIDEND = "dividend"
    ROLL = "roll"


class AdjustMethod(str, Enum):
    RATIO = "ratio"
    DIFFERENCE = "difference"


@dataclass(frozen=True, slots=True)
class Action:
    """A single price-discontinuity event.

    Build these with :func:`split`, :func:`dividend` or :func:`roll` rather
    than by hand — the constructors validate the combination of fields that
    each kind requires.

    ``value`` and ``ref_price`` carry different meanings per kind:

    ==========  ======================  ================================
    kind        value                   ref_price
    ==========  ======================  ================================
    split       new shares per old      unused
    dividend    cash paid per share     close before the ex-date
    roll        price of the new leg    price of the expiring leg
    ==========  ======================  ================================
    """

    ts_ns: int
    kind: ActionKind
    value: Decimal
    ref_price: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        if self.kind is ActionKind.SPLIT and self.value <= 0:
            raise ValueError("split ratio must be positive")
        if self.kind is ActionKind.DIVIDEND and self.value < 0:
            raise ValueError("dividend amount must not be negative")
        if self.kind is ActionKind.ROLL:
            if self.value <= 0 or self.ref_price is None or self.ref_price <= 0:
                raise ValueError("roll requires positive from_price and to_price")


def split(ts_ns: int, ratio: Decimal) -> Action:
    """A stock split. ``ratio`` is new shares per old share (4-for-1 -> 4).

    A reverse split is the same call with a fraction: 1-for-10 is
    ``Decimal("0.1")``.
    """
    return Action(ts_ns=ts_ns, kind=ActionKind.SPLIT, value=Decimal(ratio))


def dividend(
    ts_ns: int, amount: Decimal, ref_price: Optional[Decimal] = None
) -> Action:
    """A cash dividend of ``amount`` per share, stamped at the ex-date.

    Ratio adjustment needs a reference price — the close before the ex-date.
    Leave ``ref_price`` as ``None`` to have it taken from the series being
    adjusted, which is what you want unless you are matching a vendor that
    published its own reference.
    """
    return Action(
        ts_ns=ts_ns,
        kind=ActionKind.DIVIDEND,
        value=Decimal(amount),
        ref_price=None if ref_price is None else Decimal(ref_price),
    )


def roll(ts_ns: int, from_price: Decimal, to_price: Decimal) -> Action:
    """A futures contract roll from the expiring leg to the new one.

    ``from_price`` and ``to_price`` are the two contracts' prices at the same
    moment — the spread between them is the step to remove.
    """
    return Action(
        ts_ns=ts_ns,
        kind=ActionKind.ROLL,
        value=Decimal(to_price),
        ref_price=Decimal(from_price),
    )


# -- factor arithmetic -------------------------------------------------------

# An adjustment is an affine map on price plus a scale on size:
#     adjusted_price = raw_price * price_factor + price_offset
#     adjusted_size  = raw_size  * size_factor
#
# The three components are held as exact rationals rather than Decimals.
# Composing several actions multiplies their factors, and a chain like
# "1/2 then 1/3" is not representable in decimal: carrying it as Decimal makes
# a 600 -> 100 adjustment land on 99.99999999999999999999999996. Fractions
# compose exactly and the single division happens once, at application time.
_Adjustment = Tuple[Fraction, Fraction, Fraction]

_IDENTITY: _Adjustment = (Fraction(1), Fraction(0), Fraction(1))


def _own_adjustment(
    action: Action, method: AdjustMethod, ref: Optional[Decimal]
) -> _Adjustment:
    """The transform a single action imposes on the data before it."""
    if action.kind is ActionKind.SPLIT:
        # Four shares where there was one: price divides, size multiplies.
        ratio = Fraction(action.value)
        return (1 / ratio, Fraction(0), ratio)

    if action.kind is ActionKind.DIVIDEND:
        if method is AdjustMethod.DIFFERENCE:
            return (Fraction(1), -Fraction(action.value), Fraction(1))
        if ref is None or ref <= 0:
            raise ValueError(
                "ratio-adjusting a dividend needs a positive reference price; "
                "none was supplied and none could be read from the series "
                f"before ts_ns={action.ts_ns}"
            )
        if action.value >= ref:
            raise ValueError(
                f"dividend {action.value} is not smaller than the reference "
                f"price {ref} at ts_ns={action.ts_ns}"
            )
        base = Fraction(ref)
        return ((base - Fraction(action.value)) / base, Fraction(0), Fraction(1))

    # ROLL
    assert action.ref_price is not None
    old_leg, new_leg = Fraction(action.ref_price), Fraction(action.value)
    if method is AdjustMethod.DIFFERENCE:
        return (Fraction(1), new_leg - old_leg, Fraction(1))
    return (new_leg / old_leg, Fraction(0), Fraction(1))


def _compose(earlier: _Adjustment, later: _Adjustment) -> _Adjustment:
    """Apply ``earlier`` first, then ``later``, as one transform."""
    ef, eo, es = earlier
    lf, lo, ls = later
    return (ef * lf, eo * lf + lo, es * ls)


def _to_decimal(value: Fraction) -> Decimal:
    if value.denominator == 1:
        return Decimal(value.numerator)
    return Decimal(value.numerator) / Decimal(value.denominator)


def _cumulative(
    actions: Sequence[Action],
    method: AdjustMethod,
    refs: Sequence[Optional[Decimal]],
) -> List[Tuple[int, _Adjustment]]:
    """Cumulative transforms, one per action, returned oldest first.

    ``actions`` must be sorted oldest to newest and ``refs`` aligned with it.
    Each entry says: data strictly before this timestamp gets this transform.
    Because the entries are built newest-first, the transform stored against
    the *oldest* action carries the product of every action after it.
    """
    running = _IDENTITY
    out: List[Tuple[int, _Adjustment]] = []
    for action, ref in zip(reversed(actions), reversed(refs)):
        running = _compose(_own_adjustment(action, method, ref), running)
        out.append((action.ts_ns, running))
    out.reverse()
    return out


def adjustment_at(
    ts_ns: int,
    actions: Sequence[Action],
    *,
    method: AdjustMethod = AdjustMethod.RATIO,
) -> Tuple[Decimal, Decimal, Decimal]:
    """The ``(price_factor, price_offset, size_factor)`` in force at ``ts_ns``.

    Every action stamped after ``ts_ns`` contributes; the most recent segment
    is left alone and returns ``(1, 0, 1)``. Dividends must carry an explicit
    ``ref_price`` here, since no series is available to read one from.

    The three values are converted to ``Decimal`` for reporting and so are
    subject to the current decimal context; :func:`adjust_events` and
    :func:`adjust_bars` apply the underlying rationals exactly.
    """
    ordered = sorted(actions, key=lambda a: a.ts_ns)
    refs = [a.ref_price for a in ordered]
    result = _IDENTITY
    for action, ref in zip(reversed(ordered), reversed(refs)):
        if action.ts_ns <= ts_ns:
            break
        result = _compose(_own_adjustment(action, method, ref), result)
    return tuple(_to_decimal(v) for v in result)  # type: ignore[return-value]


# -- reference-price resolution ---------------------------------------------


def _resolve_refs(
    actions: Sequence[Action],
    method: AdjustMethod,
    prices: Sequence[Tuple[int, Optional[Decimal]]],
) -> List[Optional[Decimal]]:
    """Fill in missing dividend reference prices from the series itself.

    The reference is the last *raw* price strictly before the ex-date. Where
    the caller supplied one it is kept as-is.
    """
    refs: List[Optional[Decimal]] = []
    ordered_prices = sorted(
        ((ts, p) for ts, p in prices if p is not None), key=lambda x: x[0]
    )
    for action in actions:
        if action.ref_price is not None:
            refs.append(action.ref_price)
            continue
        if action.kind is not ActionKind.DIVIDEND or method is AdjustMethod.DIFFERENCE:
            refs.append(None)
            continue
        found: Optional[Decimal] = None
        for ts, price in ordered_prices:
            if ts >= action.ts_ns:
                break
            found = price
        refs.append(found)
    return refs


def _scale(value: Decimal, factor: Fraction) -> Decimal:
    """``value * factor``, multiplying before dividing to stay exact."""
    if factor.denominator == 1:
        return value * factor.numerator
    return (value * factor.numerator) / factor.denominator


def _apply(value: Optional[Decimal], adj: _Adjustment) -> Optional[Decimal]:
    if value is None:
        return None
    factor, offset, _ = adj
    out = _scale(value, factor)
    return out if offset == 0 else out + _to_decimal(offset)


def _apply_size(value: Optional[Decimal], adj: _Adjustment) -> Optional[Decimal]:
    if value is None:
        return None
    return _scale(value, adj[2])


def _segments(
    actions: Sequence[Action],
    method: AdjustMethod,
    prices: Sequence[Tuple[int, Optional[Decimal]]],
) -> List[Tuple[int, _Adjustment]]:
    ordered = sorted(actions, key=lambda a: a.ts_ns)
    if not ordered:
        return []
    refs = _resolve_refs(ordered, method, prices)
    return _cumulative(ordered, method, refs)


def _adjustment_for(
    ts_ns: int, cumulative: Sequence[Tuple[int, _Adjustment]]
) -> _Adjustment:
    # ``cumulative`` is oldest-first, so the first action stamped after
    # ``ts_ns`` is the earliest one that still applies — and its entry already
    # carries the product of every action after it.
    for action_ts, adj in cumulative:
        if action_ts > ts_ns:
            return adj
    return _IDENTITY


def adjust_events(
    events: Iterable[MarketEvent],
    actions: Sequence[Action],
    *,
    method: AdjustMethod = AdjustMethod.RATIO,
) -> List[MarketEvent]:
    """Back-adjust trade and quote events for ``actions``.

    Prices, bid/ask and sizes are restated; timestamps, symbols and sides are
    untouched. Input order is preserved. An empty ``actions`` list returns the
    events unchanged.
    """
    items = list(events)
    if not actions:
        return items

    cumulative = _segments(
        actions,
        method,
        [(e.ts_ns, e.price if e.price is not None else e.mid_price) for e in items],
    )

    out: List[MarketEvent] = []
    for e in items:
        adj = _adjustment_for(e.ts_ns, cumulative)
        if adj == _IDENTITY:
            out.append(e)
            continue
        out.append(
            MarketEvent(
                symbol=e.symbol,
                venue=e.venue,
                event_type=e.event_type,
                ts_ns=e.ts_ns,
                price=_apply(e.price, adj),
                size=_apply_size(e.size, adj),
                side=e.side,
                bid_price=_apply(e.bid_price, adj),
                bid_size=_apply_size(e.bid_size, adj),
                ask_price=_apply(e.ask_price, adj),
                ask_size=_apply_size(e.ask_size, adj),
            )
        )
    return out


def adjust_bars(
    bars: Iterable[Bar],
    actions: Sequence[Action],
    *,
    method: AdjustMethod = AdjustMethod.RATIO,
) -> List[Bar]:
    """Back-adjust OHLCV bars for ``actions``.

    Open, high, low, close and VWAP are restated; volume is scaled by the
    split factor; trade counts and bar boundaries are untouched.
    """
    items = list(bars)
    if not actions:
        return items

    cumulative = _segments(
        actions, method, [(b.start_ns, b.close) for b in items]
    )

    out: List[Bar] = []
    for b in items:
        adj = _adjustment_for(b.start_ns, cumulative)
        if adj == _IDENTITY:
            out.append(b)
            continue
        out.append(
            Bar(
                start_ns=b.start_ns,
                interval_ns=b.interval_ns,
                # A Bar's OHLCV are never None, so neither are these.
                open=cast(Decimal, _apply(b.open, adj)),
                high=cast(Decimal, _apply(b.high, adj)),
                low=cast(Decimal, _apply(b.low, adj)),
                close=cast(Decimal, _apply(b.close, adj)),
                volume=cast(Decimal, _apply_size(b.volume, adj)),
                trades=b.trades,
                vwap=_apply(b.vwap, adj),
            )
        )
    return out


# -- file input --------------------------------------------------------------

_KIND_BY_NAME = {k.value: k for k in ActionKind}


def read_actions_csv(path: str, *, ts_unit: Optional[str] = None) -> List[Action]:
    """Read corporate actions from a CSV file.

    Expected columns are ``ts`` (``timestamp`` is accepted too, matching the
    trades CSV convention), ``kind``, ``value`` and an optional ``ref_price``;
    unknown columns are ignored. Timestamps are ISO-8601 by default, or epoch
    numbers when ``ts_unit`` is given. Transparently reads ``.csv.gz``::

        ts,kind,value,ref_price
        2026-06-09T00:00:00Z,split,4,
        2026-05-09T00:00:00Z,dividend,0.25,190.50
        2026-03-14T00:00:00Z,roll,5312.50,5290.25

    For a roll, ``value`` is the new contract's price and ``ref_price`` the
    expiring one.
    """
    actions: List[Action] = []
    with open_text(path) as f:
        for lineno, row in enumerate(csv.DictReader(f), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                actions.append(_action_from_row(row, ts_unit=ts_unit))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return actions


def _action_from_row(row: dict, *, ts_unit: Optional[str]) -> Action:
    raw_ts = ((row.get("ts") or row.get("timestamp")) or "").strip()
    if not raw_ts:
        raise ValueError("missing ts")
    ts_ns = epoch_to_ns(float(raw_ts), ts_unit) if ts_unit else iso_to_ns(raw_ts)

    name = (row.get("kind") or "").strip().lower()
    if name not in _KIND_BY_NAME:
        raise ValueError(
            f"unknown kind {name!r} (expected one of "
            f"{', '.join(sorted(_KIND_BY_NAME))})"
        )
    kind = _KIND_BY_NAME[name]

    raw_value = (row.get("value") or "").strip()
    if not raw_value:
        raise ValueError("missing value")
    value = Decimal(raw_value)

    raw_ref = (row.get("ref_price") or "").strip()
    ref = Decimal(raw_ref) if raw_ref else None

    if kind is ActionKind.SPLIT:
        return split(ts_ns, value)
    if kind is ActionKind.DIVIDEND:
        return dividend(ts_ns, value, ref)
    if ref is None:
        raise ValueError("roll requires ref_price (the expiring contract price)")
    return roll(ts_ns, ref, value)
