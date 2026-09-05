"""How many independent observations you actually have.

A five-day forward return sampled every day gives you a thousand rows and
about two hundred pieces of information. Every statistic computed on the
thousand — a t-statistic, a Sharpe, a confidence interval, a p-value — is
overstated by roughly the square root of five, and nothing in the arithmetic
complains::

    from mdnorm import label_spans, effective_sample_size, deflate_t_stat

    spans = label_spans(1_000, horizon=5)
    sample = effective_sample_size(spans)
    sample.nominal, sample.effective      # 1000, 200.8
    sample.inflation                      # 2.23x on every t-statistic
    deflate_t_stat(t, sample)             # the honest one

:func:`~mdnorm.labels.forward_returns` in this library produces exactly those
overlapping labels, and :func:`~mdnorm.labels.purged_splits` already removes
the training rows whose label windows reach into a test block. This module is
the other half of the same problem: purging stops the overlap from leaking
across a split, and nothing stops it from inflating the sample within one.

**Overlap is arithmetic, not an estimate.** When you know each label's window
you can count, at every point in time, how many labels are live. A label that
shares its window with four others contributes a fifth of an observation.
Summing that over the labels gives the effective count exactly, with no model
and no assumption, and it is the number the honest t-statistic divides by.

**Autocorrelation is an estimate, and it is labelled as one.** For a return
series with no explicit label windows there is no exact answer, only the
sample autocorrelation, which is itself noisy. :func:`effective_sample_size_series`
gives a figure and marks it ``estimated``. The sum is truncated at the first
non-positive autocorrelation — the initial positive sequence — because
continuing into the noise adds terms whose sign is arbitrary and can produce
an effective sample larger than the real one, which is the one direction this
whole module exists to prevent.

**No default lag, and no default horizon.** How far the dependence reaches is
a property of the data. A truncation lag chosen for you would be a constant
that rescales the answer while leaving its shape intact, which is the error
this library declines to make anywhere else either.

**Nothing is corrected silently.** :func:`deflate_t_stat` returns the
adjusted figure and the report keeps both counts, because a statistic that
has quietly been divided by something is harder to argue with than one that
shows its working.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

__all__ = [
    "Span",
    "EffectiveSample",
    "label_spans",
    "concurrency",
    "uniqueness",
    "effective_sample_size",
    "autocorrelation",
    "effective_sample_size_series",
    "deflate_t_stat",
    "read_spans_csv",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)


@dataclass(frozen=True, slots=True)
class Span:
    """The half-open window ``[start, end)`` a label depends on.

    Units are whatever the caller counts in — bar indices, nanoseconds, days.
    The arithmetic only cares that the numbers are comparable.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("a label span must cover a positive range")

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class EffectiveSample:
    """The nominal count, the effective count, and what separates them."""

    nominal: int
    effective: Decimal
    estimated: bool = False

    def __post_init__(self) -> None:
        if self.effective < 0:
            raise ValueError("an effective sample size cannot be negative")

    @property
    def ratio(self) -> Optional[Decimal]:
        """Effective over nominal. One means the observations do not overlap."""
        if self.nominal == 0:
            return None
        return self.effective / self.nominal

    @property
    def inflation(self) -> Optional[Decimal]:
        """How far a t-statistic computed on the nominal count is overstated.

        The square root of nominal over effective. A five-day label sampled
        daily lands near 2.2, which turns a t of 2.0 — publishable — into 0.9.
        """
        if self.effective <= 0:
            return None
        return (Decimal(self.nominal) / self.effective).sqrt()


def label_spans(count: int, *, horizon: int, step: int = 1,
                start: int = 0) -> List[Span]:
    """Windows for ``count`` labels of length ``horizon``, sampled every ``step``.

    This is the shape :func:`~mdnorm.labels.forward_returns` produces: label
    ``i`` looks forward from observation ``i`` to ``i + horizon``, so with the
    default step of one every label overlaps the ``horizon - 1`` labels on
    either side of it.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")
    return [Span(start + i * step, start + i * step + horizon)
            for i in range(count)]


def concurrency(spans: Sequence[Span]) -> Dict[int, int]:
    """How many labels are live at each point, as ``{point: count}``.

    Only the points where the count changes are keys; the value holds until
    the next key. Points outside every span are absent rather than zero,
    since a stretch nothing depends on is not part of the sample at all.
    """
    if not spans:
        return {}
    edges: Dict[int, int] = {}
    for s in spans:
        edges[s.start] = edges.get(s.start, 0) + 1
        edges[s.end] = edges.get(s.end, 0) - 1

    out: Dict[int, int] = {}
    live = 0
    for point in sorted(edges):
        live += edges[point]
        out[point] = live
    return out


def uniqueness(spans: Sequence[Span]) -> List[Decimal]:
    """For each label, the average of ``1 / concurrency`` over its own window.

    One when a label shares its window with nothing. A half when it is
    everywhere paired with one other. This is the fraction of an independent
    observation the label is worth, and summing it gives the effective count.
    """
    if not spans:
        return []
    changes = concurrency(spans)
    points = sorted(changes)
    counts = [changes[p] for p in points]

    out: List[Decimal] = []
    for s in spans:
        total = _ZERO
        covered = 0
        # Walk only the segments the span touches, not every unit in it.
        i = max(0, bisect_right(points, s.start) - 1)
        while i < len(points) and points[i] < s.end:
            seg_start = max(points[i], s.start)
            seg_end = min(points[i + 1] if i + 1 < len(points) else s.end,
                          s.end)
            width = seg_end - seg_start
            live = counts[i]
            if width > 0 and live > 0:
                total += Decimal(width) / live
                covered += width
            i += 1
        out.append(total / covered if covered else _ZERO)
    return out


def effective_sample_size(spans: Sequence[Span]) -> EffectiveSample:
    """The exact number of independent observations a set of labels carries.

    No model and no assumption: the labels state their own windows, and the
    count follows from them. ``estimated`` is False for exactly that reason.
    """
    return EffectiveSample(nominal=len(spans),
                           effective=sum(uniqueness(spans), _ZERO),
                           estimated=False)


def autocorrelation(values: Sequence[Decimal], *, max_lag: int) -> List[Decimal]:
    """Sample autocorrelation at lags ``1..max_lag``.

    Uses the whole-series mean and the same denominator at every lag, which
    is the biased-but-stable convention: the alternative rescales each lag by
    a shrinking sample and produces tail estimates that swing wildly on the
    handful of pairs behind them.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    n = len(values)
    if n < 2:
        raise ValueError("autocorrelation needs at least two observations")
    if max_lag >= n:
        raise ValueError(f"max_lag must be below the sample size ({n})")

    mean = sum(values, _ZERO) / n
    centred = [v - mean for v in values]
    denominator = sum((c * c for c in centred), _ZERO)
    if denominator == 0:
        raise ValueError("a constant series has no autocorrelation")

    out: List[Decimal] = []
    for lag in range(1, max_lag + 1):
        numerator = sum((centred[i] * centred[i - lag]
                         for i in range(lag, n)), _ZERO)
        out.append(numerator / denominator)
    return out


def effective_sample_size_series(
    values: Sequence[Decimal], *, max_lag: int
) -> EffectiveSample:
    """Estimate the effective count of an autocorrelated series.

    ``n / (1 + 2 * sum(rho))``, with the sum taken over the **initial positive
    sequence** — it stops at the first non-positive autocorrelation. Past that
    point the estimates are mostly noise whose signs cancel arbitrarily, and
    including them can produce an effective sample larger than the nominal
    one, which is the direction this module exists to rule out.

    The result carries ``estimated=True``. It is a sample statistic computed
    from a sample statistic, and it should be read as an order of magnitude
    rather than a figure to quote.
    """
    rho = autocorrelation(values, max_lag=max_lag)
    total = _ZERO
    for r in rho:
        if r <= 0:
            break
        total += r

    n = len(values)
    factor = _ONE + 2 * total
    return EffectiveSample(nominal=n, effective=Decimal(n) / factor,
                           estimated=True)


def deflate_t_stat(t_stat: Decimal, sample: EffectiveSample) -> Optional[Decimal]:
    """Rescale a t-statistic computed on the nominal count.

    Returns ``None`` when the effective count is not positive, rather than a
    figure with nothing behind it. The adjusted statistic is the original
    divided by :attr:`EffectiveSample.inflation`, which is the same as
    recomputing it against the effective count.
    """
    factor = sample.inflation
    if factor is None or factor == 0:
        return None
    return t_stat / factor


def read_spans_csv(
    path: str,
    *,
    start_column: str = "start",
    end_column: str = "end",
) -> List[Span]:
    """Read explicit label windows from ``start,end`` rows.

    For the case where the windows are irregular — event-driven labels, a
    holding period that ends on a signal rather than on a clock — and cannot
    be described by a horizon and a step.
    """
    import csv

    from .fileio import open_text

    out: List[Span] = []
    with open_text(path) as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            try:
                out.append(Span(int(row[start_column]), int(row[end_column])))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"line {i}: {exc}")
    if not out:
        raise ValueError("no spans in file")
    return out
