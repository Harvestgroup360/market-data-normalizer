"""Converting a price into another currency, at a moment you have to name.

A price is a number and a currency, and most pipelines carry only the number.
The moment a study spans two venues that quote in different currencies, or a
book is reported in one currency and traded in several, every figure in it
depends on a second series nobody was watching::

    from mdnorm import CurrencyPair, FxRates, convert_series

    rates = FxRates({CurrencyPair("EUR", "USD"): eurusd})
    usd, dropped = convert_series(prices_in_eur, rates,
                                  base="EUR", to="USD",
                                  max_age_ns=MINUTE)

**There is no default conversion time.** Converting at the observation's own
timestamp, at a daily fix, or at the end of the study are three different
questions, and only the first is available to someone standing at that moment.
The last is the one that gets used by accident, because a single rate is
easier to obtain than a series, and it is a look-ahead: the whole history is
restated using a number that did not exist until the end of it. This module
converts as-of and nothing else, which is why there is no function here that
takes a scalar rate.

**Staleness is the ordinary failure, not an exotic one.** FX quotes stop over
weekends and holidays while other venues keep trading, so an as-of join that
never checks the age of the rate will happily convert a Sunday crypto print
with Friday's close. :meth:`FxRates.quote` requires ``max_age_ns`` and returns
the age it used, so a conversion carries the evidence of how fresh it was.

**Direction is not guessable from a name.** ``USD/JPY`` and ``JPY/USD`` are
reciprocals, vendors disagree about which way round to publish a pair, and a
rate applied upside-down produces a number that is wrong by a factor of
thousands — or, for a pair near parity, wrong by a few per cent and entirely
plausible. Pairs here always carry an explicit base and quote, inversion is
recorded in the result, and :class:`FxRates` can be told to refuse it.

**A cross rate is not free.** Converting through a vehicle currency multiplies
two quotes and inherits both spreads and both staleness windows. That is often
the only route available, and it should be visible in the result rather than
implied. This module never searches for a path: state the vehicle with ``via``
or the conversion is refused, because a path found automatically is a
modelling decision made by a library.

**A converted return is not a converted price.** Over any interval the asset
return in the quote currency satisfies ``(1 + r_quote) = (1 + r_base)(1 +
r_fx)`` exactly. The familiar shorthand adds the two and drops the cross term,
which is small over a minute and not small over a year.
:func:`decompose_return` computes both and reports the difference rather than
choosing for you.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, cast

from .align import AsOfSeries
from .bars import Bar
from .features import _PRECISION

__all__ = [
    "CurrencyPair",
    "Quote",
    "Conversion",
    "FxRates",
    "ReturnDecomposition",
    "convert_series",
    "convert_bars",
    "decompose_return",
    "read_fx_csv",
]


def _norm(code: str) -> str:
    c = code.strip().upper()
    if not c:
        raise ValueError("a currency code cannot be empty")
    if not c.isalnum():
        raise ValueError(f"{code!r} is not a currency code")
    return c


@dataclass(frozen=True, slots=True)
class CurrencyPair:
    """One unit of ``base``, priced in ``quote``.

    ``CurrencyPair("EUR", "USD")`` at 1.09 means one euro costs 1.09 dollars.
    The direction is carried in the type rather than in a naming convention,
    because the conventions disagree with each other.
    """

    base: str
    quote: str

    def __init__(self, base: str, quote: str) -> None:
        b, q = _norm(base), _norm(quote)
        if b == q:
            raise ValueError(f"{b} against itself is not a pair")
        object.__setattr__(self, "base", b)
        object.__setattr__(self, "quote", q)

    @classmethod
    def parse(cls, text: str) -> "CurrencyPair":
        """Read ``EUR/USD`` or ``EUR-USD``.

        A separator is required. Six-letter forms like ``EURUSD`` are not
        accepted, because splitting them assumes every code is three letters
        and that assumption fails on the venues most likely to need this
        module.
        """
        for sep in ("/", "-", "_"):
            if sep in text:
                a, b = text.split(sep, 1)
                return cls(a, b)
        raise ValueError(
            f"{text!r} needs a separator, e.g. EUR/USD; a six-letter form "
            "cannot be split without assuming three-letter codes")

    @property
    def inverse(self) -> "CurrencyPair":
        return CurrencyPair(self.quote, self.base)

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.base}/{self.quote}"


@dataclass(frozen=True, slots=True)
class Quote:
    """One rate used in a conversion, and how it was obtained."""

    pair: CurrencyPair
    rate: Decimal
    as_of_ns: int
    age_ns: int
    inverted: bool = False

    @property
    def applied(self) -> Decimal:
        """The multiplier actually applied, after any inversion."""
        if not self.inverted:
            return self.rate
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(1) / self.rate


@dataclass(frozen=True, slots=True)
class Conversion:
    """A converted amount, with every rate that produced it."""

    amount: Decimal
    base: str
    quote: str
    legs: Tuple[Quote, ...]

    @property
    def rate(self) -> Decimal:
        """The effective rate from base to quote."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            r = Decimal(1)
            for leg in self.legs:
                r *= leg.applied
            return r

    @property
    def age_ns(self) -> int:
        """The age of the stalest rate used.

        The oldest leg governs, because a cross rate is no fresher than its
        worst half however recent the other one was.
        """
        return max((leg.age_ns for leg in self.legs), default=0)

    @property
    def crossed(self) -> bool:
        return len(self.legs) > 1


@dataclass(frozen=True, slots=True)
class ReturnDecomposition:
    """An asset return in the quote currency, split into its parts."""

    asset_return: Decimal
    fx_return: Decimal
    total_return: Decimal

    @property
    def cross_term(self) -> Decimal:
        """The product the additive shorthand drops."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return self.asset_return * self.fx_return

    @property
    def additive(self) -> Decimal:
        """``asset + fx``, the approximation."""
        return self.asset_return + self.fx_return

    @property
    def approximation_error(self) -> Decimal:
        """How much the shorthand misses by. Equal to the cross term."""
        return self.total_return - self.additive


class FxRates:
    """A set of rate series, queried as of a moment.

    Series are keyed by :class:`CurrencyPair`. A pair given one way round can
    answer the other way round by inversion unless ``allow_inverse`` is false;
    either way the result records that it happened.
    """

    __slots__ = ("_series", "allow_inverse")

    def __init__(
        self,
        series: Mapping[CurrencyPair, AsOfSeries],
        *,
        allow_inverse: bool = True,
    ) -> None:
        self._series: Dict[CurrencyPair, AsOfSeries] = dict(series)
        for pair, s in self._series.items():
            if not isinstance(pair, CurrencyPair):
                raise TypeError("keys must be CurrencyPair")
            if len(s) == 0:
                raise ValueError(f"{pair} has no observations")
        self.allow_inverse = allow_inverse

    @property
    def pairs(self) -> Tuple[CurrencyPair, ...]:
        return tuple(self._series)

    def currencies(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for p in self._series:
            for c in (p.base, p.quote):
                if c not in seen:
                    seen.append(c)
        return tuple(sorted(seen))

    # -- one leg -----------------------------------------------------------

    def quote(self, pair: CurrencyPair, ts_ns: int, *,
              max_age_ns: int) -> Optional[Quote]:
        """The rate for ``pair`` as of ``ts_ns``, or ``None``.

        ``max_age_ns`` is required. An as-of join with no age limit will
        convert against whatever it can reach, which over a weekend or a
        holiday is Friday — and the resulting number looks exactly like a
        fresh one.

        ``None`` means no rate was usable: either nothing had been observed
        yet, or the newest observation was staler than allowed. Both are
        refusals rather than approximations.
        """
        if max_age_ns < 0:
            raise ValueError("max_age_ns must be non-negative")
        direct = self._series.get(pair)
        if direct is not None:
            v, age = direct.at(ts_ns, max_age_ns=max_age_ns)
            if v is None:
                return None
            if v <= 0:
                raise ValueError(f"{pair} quoted a non-positive rate {v}")
            # A present value always carries an age; the pair type cannot say so.
            hit = cast(int, age)
            return Quote(pair, v, ts_ns - hit, hit, inverted=False)
        if not self.allow_inverse:
            return None
        flipped = self._series.get(pair.inverse)
        if flipped is None:
            return None
        v, age = flipped.at(ts_ns, max_age_ns=max_age_ns)
        if v is None:
            return None
        if v <= 0:
            raise ValueError(f"{pair.inverse} quoted a non-positive rate {v}")
        hit = cast(int, age)
        return Quote(pair.inverse, v, ts_ns - hit, hit, inverted=True)

    def has(self, pair: CurrencyPair) -> bool:
        """Whether this set can answer ``pair`` at all, inversion included."""
        if pair in self._series:
            return True
        return self.allow_inverse and pair.inverse in self._series

    # -- conversion --------------------------------------------------------

    def convert(
        self,
        amount: Decimal,
        base: str,
        quote: str,
        ts_ns: int,
        *,
        max_age_ns: int,
        via: Optional[str] = None,
    ) -> Optional[Conversion]:
        """Convert ``amount`` from ``base`` into ``quote`` as of ``ts_ns``.

        Returns ``None`` when a rate was missing or too stale — the caller
        decides what a missing conversion means, because dropping the
        observation and carrying it unconverted are both defensible and
        neither should happen silently.

        ``via`` names a vehicle currency for a cross rate. Nothing is searched
        for: if the direct pair is absent and no vehicle is stated, the
        conversion is refused. A library that finds its own path through the
        currency graph is choosing the counterparties whose spreads you pay,
        and doing it invisibly.
        """
        b, q = _norm(base), _norm(quote)
        if b == q:
            return Conversion(amount, b, q, ())

        direct = self.quote(CurrencyPair(b, q), ts_ns, max_age_ns=max_age_ns)
        if direct is not None:
            with localcontext() as ctx:
                ctx.prec = _PRECISION
                return Conversion(amount * direct.applied, b, q, (direct,))
        if via is None:
            if self.has(CurrencyPair(b, q)):
                return None          # the pair exists; the rate was unusable
            raise ValueError(
                f"no {b}/{q} rate; state a vehicle currency with via= rather "
                "than letting a path be chosen for you")

        v = _norm(via)
        if v in (b, q):
            raise ValueError(f"{v} cannot be the vehicle for {b}/{q}")
        first = self.quote(CurrencyPair(b, v), ts_ns, max_age_ns=max_age_ns)
        second = self.quote(CurrencyPair(v, q), ts_ns, max_age_ns=max_age_ns)
        if first is None or second is None:
            if not (self.has(CurrencyPair(b, v)) and self.has(CurrencyPair(v, q))):
                raise ValueError(
                    f"cannot reach {q} from {b} through {v}; one of the two "
                    "legs is not in this rate set")
            return None
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            out = amount * first.applied * second.applied
        return Conversion(out, b, q, (first, second))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FxRates(pairs={[str(p) for p in self._series]}, "
                f"allow_inverse={self.allow_inverse})")


def convert_series(
    series: AsOfSeries,
    rates: FxRates,
    *,
    base: str,
    to: str,
    max_age_ns: int,
    via: Optional[str] = None,
    name: str = "",
) -> Tuple[AsOfSeries, int]:
    """Convert every observation at its own timestamp.

    Returns the converted series and the number of observations dropped
    because no usable rate existed at that moment. The count is returned
    rather than logged because a converted series that is quietly shorter
    than its input is the kind of thing that gets noticed three steps later.

    Each point uses the rate as of that point. That is the only conversion a
    person standing at that moment could have made, and it is why there is no
    parameter here for a single rate.
    """
    out: List[Tuple[int, Decimal]] = []
    dropped = 0
    for ts in series._ts:  # noqa: SLF001 - sorted timestamps, read-only
        # ts came out of the series itself, so the lookup cannot miss.
        value = cast(Decimal, series.at(ts)[0])
        c = rates.convert(value, base, to, ts, max_age_ns=max_age_ns, via=via)
        if c is None:
            dropped += 1
            continue
        out.append((ts, c.amount))
    return AsOfSeries(out, name=name or series.name), dropped


def convert_bars(
    bars: Iterable[Bar],
    rates: FxRates,
    *,
    base: str,
    to: str,
    max_age_ns: int,
    via: Optional[str] = None,
) -> Tuple[List[Bar], int]:
    """Convert a bar's four prices at the bar's end, leaving volume alone.

    All four prices take the same rate — the one as of ``end_ns`` — so the
    OHLC relationships survive. Converting each price at the moment it
    occurred would be more precise about the high and the low and would let
    the high fall below the open, which is a worse trade than the precision
    is worth.

    Volume is a quantity of the instrument and is not a currency amount, so
    it is not touched. Notional turnover is the caller's to compute, and
    doing it here would silently invent a definition of it.
    """
    out: List[Bar] = []
    dropped = 0
    for bar in bars:
        c = rates.convert(Decimal(1), base, to, bar.end_ns,
                          max_age_ns=max_age_ns, via=via)
        if c is None:
            dropped += 1
            continue
        r = c.rate
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            out.append(Bar(
                start_ns=bar.start_ns, interval_ns=bar.interval_ns,
                open=bar.open * r, high=bar.high * r, low=bar.low * r,
                close=bar.close * r, volume=bar.volume, trades=bar.trades))
    return out, dropped


def decompose_return(
    price_start: Decimal,
    price_end: Decimal,
    rate_start: Decimal,
    rate_end: Decimal,
) -> ReturnDecomposition:
    """Split a converted return into asset, currency and the cross term.

    ``(1 + total) = (1 + asset)(1 + fx)`` holds exactly. The additive
    shorthand ``total ~= asset + fx`` drops the product of the two, which is
    negligible over a minute and reaches the tenths of a percent over a year
    on ordinary moves — always in the direction that makes a gain in both
    look smaller than it was, and a loss in both look smaller too.
    """
    for name, v in (("price_start", price_start), ("price_end", price_end),
                    ("rate_start", rate_start), ("rate_end", rate_end)):
        if v <= 0:
            raise ValueError(f"{name} must be positive")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        r_asset = price_end / price_start - 1
        r_fx = rate_end / rate_start - 1
        total = (price_end * rate_end) / (price_start * rate_start) - 1
    return ReturnDecomposition(r_asset, r_fx, total)


def read_fx_csv(
    path: str,
    *,
    pair_column: str = "pair",
    ts_column: str = "ts_ns",
    rate_column: str = "rate",
    allow_inverse: bool = True,
) -> FxRates:
    """Read a long-format CSV of ``pair,ts_ns,rate`` into a rate set.

    Pairs must carry a separator, so a file of ``EURUSD`` is rejected rather
    than split on an assumption. A non-positive rate is an error and not a
    missing value: zero is not a price, and a negative one is a sign error
    somewhere upstream that should surface here.
    """
    import csv

    from .fileio import open_text

    buckets: Dict[CurrencyPair, List[Tuple[int, Decimal]]] = {}
    with open_text(path) as fh:
        for row in csv.DictReader(fh):
            pair = CurrencyPair.parse(row[pair_column].strip())
            rate = Decimal(row[rate_column].strip())
            if rate <= 0:
                raise ValueError(f"{pair} has a non-positive rate {rate}")
            buckets.setdefault(pair, []).append(
                (int(row[ts_column]), rate))
    if not buckets:
        raise ValueError("no rates in file")
    return FxRates({p: AsOfSeries(obs, name=str(p))
                    for p, obs in buckets.items()},
                   allow_inverse=allow_inverse)
