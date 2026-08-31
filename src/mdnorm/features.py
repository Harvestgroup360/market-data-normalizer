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

**The rolling sum is slid, and only where sliding is exact.** Recomputing a
window at every index is what made these functions cost O(n x window); adding
the arriving value and subtracting the departing one is O(n). The usual reason
not to do it is drift, so each update runs with the ``Inexact`` flag cleared
and is discarded the moment it would round, falling back to a full sum of the
window. On ordinary data the results are identical; where they differ it is
because the forward recomputation rounded an intermediate partial and the slid
total did not, and the slid total is the exact one.

Nothing here forecasts, ranks, or scores. These are descriptive statistics with
their window written down.
"""
from __future__ import annotations

from decimal import (Decimal, Inexact, InvalidOperation, getcontext,
                     localcontext)
from enum import Enum
from typing import List, Optional, Sequence

from .align import AlignedRow

__all__ = [
    "ReturnMethod",
    "column",
    "timestamps",
    "returns",
    "rolling_sum",
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

_ZERO = Decimal(0)

#: Placeholder yielded in place of the window values when the caller does not
#: need them; slicing the window is most of the cost of a rolling sum.
_PRESENT = ()


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


def _windows_with_sum(values: _Series, window: int, *,
                      need_values: bool = True):
    """Yield ``(index, window_values, trailing_sum)`` for every index.

    ``(None, None)`` for an index whose window is incomplete — either because
    the series has not produced ``window`` observations yet, or because one of
    them is missing. The gap check is carried rather than rescanned:
    remembering where the most recent hole was makes it one comparison per
    index instead of a sweep of the whole window.

    A rolling sum recomputed from scratch at every index is the whole reason
    these functions cost O(n x window). Sliding it — add the arriving value,
    subtract the departing one — is O(n), and the usual objection is that it
    drifts: a running total accumulates rounding a fresh sum does not, so the
    same window stops giving the same answer twice.

    That objection is about rounding, and rounding is observable. Each update
    runs with the ``Inexact`` flag cleared; if the flag is raised the update is
    discarded and the window is summed from scratch. The slid total is
    therefore used only on the steps where it is provably the exact sum.

    On ordinary price data the two agree value for value. They can disagree,
    and it is worth being exact about which way: summing a window forwards can
    round an intermediate partial that the slid total never holds — a window
    containing both 1e25 and 3.14159 is enough — and in that case the slid
    total is the correct one and the recomputed one has lost a digit. So this
    is a correctness improvement that shows up as a changed value on series
    mixing very large and very small magnitudes, and as no change at all
    everywhere else.
    """
    last_gap = -1
    running: Optional[Decimal] = None
    trust = True
    flags = getcontext().flags
    for i in range(len(values)):
        if values[i] is None:
            last_gap = i
        if i + 1 < window:
            yield i, None, None
            continue
        start = i - window + 1
        if last_gap >= start:
            running = None
            yield i, None, None
            continue
        chunk = list(values[start:i + 1]) if need_values else _PRESENT
        if running is None or not trust:
            flags[Inexact] = False
            running = sum(values[start:i + 1], _ZERO)
            trust = not flags[Inexact]
        else:
            flags[Inexact] = False
            candidate = running + values[i] - values[start - 1]
            if flags[Inexact]:
                flags[Inexact] = False
                running = sum(values[start:i + 1], _ZERO)
                trust = not flags[Inexact]
            else:
                running = candidate
        yield i, chunk, running


def _mean_and_std(chunk: List[Decimal], total: Decimal, window: int,
                  ddof: int):
    """Both statistics, given the window and its already-computed sum.

    The variance is still a second pass over the window in the original order.
    It cannot be slid the way the sum can: the identity that would let it —
    subtracting the square of the mean from the mean of the squares — is exact
    in algebra and a different sequence of roundings in arithmetic, so it would
    change published numbers to save time. The sum is slid because sliding it
    changes nothing.
    """
    mean = total / window
    var = sum(((v - mean) ** 2 for v in chunk), _ZERO) / (window - ddof)
    return mean, var.sqrt()


def rolling_sum(values: _Series, window: int) -> List[Optional[Decimal]]:
    """Trailing sum over ``window`` observations.

    The primitive under the mean, and useful on its own for trailing volume,
    turnover or trade counts. Linear in the length of the series: the total is
    slid rather than recomputed, on every step where sliding it is provably
    exact, and recomputed on the steps where it is not.
    """
    _check_window(window)
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for _, _chunk, total in _windows_with_sum(values, window,
                                                  need_values=False):
            out.append(total)
    return out


def rolling_mean(values: _Series, window: int) -> List[Optional[Decimal]]:
    """Trailing arithmetic mean over ``window`` observations."""
    _check_window(window)
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for _, _chunk, total in _windows_with_sum(values, window,
                                                  need_values=False):
            out.append(None if total is None else total / window)
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
        for _, chunk, total in _windows_with_sum(values, window):
            out.append(None if chunk is None
                       else _mean_and_std(chunk, total, window, ddof)[1])
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
    if ddof < 0 or ddof >= window:
        raise ValueError("ddof must be non-negative and smaller than window")
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for i, chunk, total in _windows_with_sum(values, window):
            v = values[i]
            if chunk is None or v is None:
                out.append(None)
                continue
            mean, std = _mean_and_std(chunk, total, window, ddof)
            out.append(None if std == 0 else (v - mean) / std)
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
