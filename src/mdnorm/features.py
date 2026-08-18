"""Returns and rolling statistics that only ever look backwards.

Once several instruments sit on one time grid (see :mod:`mdnorm.align`), the
next step is turning prices into features: returns, volatility, z-scores,
correlations. This is the second place where the future gets into a research
pipeline, and again it gets in quietly::

    from mdnorm import ReturnMethod, column, returns, rolling_zscore

    px = column(rows, "BTC")
    r = returns(px, method=ReturnMethod.LOG)
    z = rolling_zscore(px, window=60)

**A z-score standardised over the whole sample is look-ahead.** Subtracting the
full-sample mean and dividing by the full-sample standard deviation gives every
point in the series knowledge of every other point, including the ones that had
not happened yet. It is a single line of code, it is in a great deal of
published work, and it inflates results substantially, because the model learns
where each observation sat inside a distribution nobody could have known at the
time. Everything here is rolling: the value at index ``i`` is computed from
``values[i - window + 1 : i + 1]`` and nothing else.

**A partial window is not a result.** Until a window has filled, these
functions emit ``None`` rather than a value computed from three observations
where twenty were asked for. The first rows of a feature matrix are supposed to
be empty; a library that quietly shortens the window at the start hands you a
series whose early values are noisier than its late ones for no stated reason.

**A gap propagates rather than being stepped over.** A window containing a
``None`` yields ``None``. Silently dropping the hole would compute a
twenty-period statistic from nineteen periods and label it as twenty.

**There is no default annualisation factor.** Multiplying by the square root of
252 is correct for daily bars on a market that trades 252 days a year and wrong
for essentially everything else, including every minute bar and every
instrument that trades around the clock. :func:`periods_per_year` makes you
state the session length and the number of sessions, and
:func:`realized_volatility` returns per-period volatility unless you supply one.

Nothing here forecasts, ranks, or scores. These are descriptive statistics with
their window written down.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import List, Optional, Sequence

from .align import AlignedRow

__all__ = [
    "ReturnMethod",
    "column",
    "timestamps",
    "returns",
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "rolling_correlation",
    "realized_volatility",
    "periods_per_year",
]

_Series = Sequence[Optional[Decimal]]

#: Working precision for the square roots and logarithms below. Wide enough
#: that rounding never reaches a figure anyone reports.
_PRECISION = 34


# -- getting a series out of an aligned matrix -------------------------------


def column(rows: Sequence[AlignedRow], name: str) -> List[Optional[Decimal]]:
    """One column of an aligned matrix, holes included.

    Raises :class:`KeyError` if the column is absent from the first row, so a
    typo fails immediately instead of producing a series of ``None``.
    """
    if rows and name not in rows[0].values:
        raise KeyError(
            f"no column {name!r} in the aligned rows; "
            f"available: {sorted(rows[0].values)}"
        )
    return [r.values.get(name) for r in rows]


def timestamps(rows: Sequence[AlignedRow]) -> List[int]:
    """The grid timestamps of an aligned matrix, for writing features back out."""
    return [r.ts_ns for r in rows]


# -- returns -----------------------------------------------------------------


class ReturnMethod(str, Enum):
    """Simple (arithmetic) or log returns."""

    SIMPLE = "simple"
    LOG = "log"


def returns(
    values: _Series, *, method: ReturnMethod = ReturnMethod.SIMPLE
) -> List[Optional[Decimal]]:
    """Period-over-period returns, aligned to the *later* observation.

    ``out[i]`` is the return from ``values[i - 1]`` to ``values[i]``, so it is
    knowable at ``i`` and at no earlier point. ``out[0]`` is always ``None``:
    there is no return into the first observation, and putting a zero there
    would add a fabricated flat period to every series. The output is always
    the same length as the input, so it can be written back alongside the grid
    it came from.

    A missing or non-positive price yields ``None`` rather than an exception —
    a non-positive price has no return in either convention, and in normalized
    data it means something upstream is wrong.
    """
    if not values:
        return []
    out: List[Optional[Decimal]] = [None]
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for i in range(1, len(values)):
            prev, cur = values[i - 1], values[i]
            if prev is None or cur is None or prev <= 0 or cur <= 0:
                out.append(None)
                continue
            try:
                if method is ReturnMethod.LOG:
                    out.append((cur / prev).ln())
                else:
                    out.append(cur / prev - 1)
            except (InvalidOperation, ZeroDivisionError):  # pragma: no cover
                out.append(None)
    return out


# -- rolling statistics ------------------------------------------------------


def _check_window(window: int, minimum: int = 1) -> None:
    if window < minimum:
        raise ValueError(f"window must be at least {minimum}")


def _windows(values: _Series, window: int):
    """Yield ``(index, window_values_or_None)`` for every index.

    ``None`` for an index whose window is incomplete — either because the
    series has not produced ``window`` observations yet, or because one of them
    is missing.
    """
    for i in range(len(values)):
        if i + 1 < window:
            yield i, None
            continue
        chunk = values[i - window + 1: i + 1]
        yield i, (None if any(v is None for v in chunk) else list(chunk))


def rolling_mean(values: _Series, window: int) -> List[Optional[Decimal]]:
    """Trailing arithmetic mean over ``window`` observations."""
    _check_window(window)
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for _, chunk in _windows(values, window):
            out.append(None if chunk is None
                       else sum(chunk, Decimal(0)) / window)
    return out


def rolling_std(
    values: _Series, window: int, *, ddof: int = 1
) -> List[Optional[Decimal]]:
    """Trailing standard deviation, sample by default (``ddof=1``).

    The sample convention is the default because a rolling window is a sample
    of a longer process, not the population. Pass ``ddof=0`` if you mean the
    population form; the difference matters at the window sizes people
    actually use.
    """
    _check_window(window, minimum=2)
    if ddof < 0 or ddof >= window:
        raise ValueError("ddof must be non-negative and smaller than window")
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for _, chunk in _windows(values, window):
            if chunk is None:
                out.append(None)
                continue
            mean = sum(chunk, Decimal(0)) / window
            var = sum(((v - mean) ** 2 for v in chunk), Decimal(0)) / (window - ddof)
            out.append(var.sqrt())
    return out


def rolling_zscore(
    values: _Series, window: int, *, ddof: int = 1
) -> List[Optional[Decimal]]:
    """Trailing z-score: how unusual the latest value is against its own past.

    This is the honest counterpart of the full-sample standardisation that
    appears throughout published research. Standardising against the mean and
    standard deviation of the entire series gives every observation knowledge
    of the whole distribution, including the part that had not happened yet.

    A window whose values are all identical has zero dispersion, and the
    z-score is then undefined rather than zero or infinite, so ``None`` is
    returned. That case is common and meaningful: it is usually a forward-fill
    that has not expired.
    """
    _check_window(window, minimum=2)
    means = rolling_mean(values, window)
    stds = rolling_std(values, window, ddof=ddof)
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for i, v in enumerate(values):
            m, s = means[i], stds[i]
            if v is None or m is None or s is None or s == 0:
                out.append(None)
            else:
                out.append((v - m) / s)
    return out


def rolling_correlation(
    a: _Series, b: _Series, window: int
) -> List[Optional[Decimal]]:
    """Trailing Pearson correlation between two aligned series.

    Both inputs must already be on the same grid — that is what
    :func:`mdnorm.align` is for — and must be the same length.

    A series with no variation inside the window has an undefined correlation
    with anything, so the result is ``None`` rather than zero. Reading that as
    zero is how a dead feed becomes an apparent diversifier: a frozen column
    correlates with nothing precisely because it is not moving.
    """
    _check_window(window, minimum=2)
    if len(a) != len(b):
        raise ValueError("series must be the same length; align them first")
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for i in range(len(a)):
            if i + 1 < window:
                out.append(None)
                continue
            xa = a[i - window + 1: i + 1]
            xb = b[i - window + 1: i + 1]
            if any(v is None for v in xa) or any(v is None for v in xb):
                out.append(None)
                continue
            ma = sum(xa, Decimal(0)) / window
            mb = sum(xb, Decimal(0)) / window
            cov = sum(((x - ma) * (y - mb) for x, y in zip(xa, xb)), Decimal(0))
            va = sum(((x - ma) ** 2 for x in xa), Decimal(0))
            vb = sum(((y - mb) ** 2 for y in xb), Decimal(0))
            if va == 0 or vb == 0:
                out.append(None)
                continue
            out.append(cov / (va.sqrt() * vb.sqrt()))
    return out


# -- volatility --------------------------------------------------------------


def periods_per_year(
    interval_ns: int, *, sessions_per_year: int, session_length_ns: int
) -> Decimal:
    """How many bars of ``interval_ns`` a year contains, stated explicitly.

    Both arguments are required because there is no answer that is right for
    every market. US cash equities are roughly 252 sessions of six and a half
    hours; a perpetual futures venue is 365 sessions of twenty-four. Getting
    this wrong rescales every volatility number in a report by a constant that
    nobody notices, because the shape of the series is unchanged.
    """
    if interval_ns <= 0:
        raise ValueError("interval_ns must be positive")
    if sessions_per_year <= 0 or session_length_ns <= 0:
        raise ValueError("sessions_per_year and session_length_ns must be positive")
    return Decimal(sessions_per_year * session_length_ns) / Decimal(interval_ns)


def realized_volatility(
    period_returns: _Series,
    *,
    window: int,
    periods_per_year: Optional[Decimal] = None,
    ddof: int = 1,
) -> List[Optional[Decimal]]:
    """Trailing standard deviation of returns, per period unless annualised.

    Feed this the output of :func:`returns`, not prices. Without
    ``periods_per_year`` the result is the volatility of one period, which is
    the only figure derivable from the data alone; supply the factor — see
    :func:`periods_per_year` — and it is scaled by its square root.

    The annualisation is deliberately not defaulted. A square root of 252
    applied to minute bars, or to an instrument that trades continuously,
    produces a number that is wrong by an order of magnitude and looks
    entirely plausible.
    """
    vol = rolling_std(period_returns, window, ddof=ddof)
    if periods_per_year is None:
        return vol
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        factor = Decimal(periods_per_year).sqrt()
        return [None if v is None else v * factor for v in vol]
