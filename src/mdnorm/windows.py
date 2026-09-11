"""Choosing where the sample starts is a trial, and nobody counts it.

A backtest is reported as one number over one window. The window was also
chosen — by when the data happened to begin, by which vendor file was to
hand, by somebody sliding the start forward until the equity curve looked
right. Each of those is a decision, and the last one is a search::

    from mdnorm import trimmed_starts, sweep, sharpe_ratio

    rep = sweep(returns, trimmed_starts(len(returns), step=21, count=24),
                sharpe_ratio)
    rep.full        # 0.79 over everything
    rep.highest     # 1.34 starting eleven months in
    rep.lowest      # 0.31 starting four months in
    rep.trials      # 25 — the number to hand a deflated Sharpe

**The spread is the finding, not the best value in it.** Twenty-five start
dates that produce Sharpe ratios between 0.31 and 1.34 have not found a
strategy with a Sharpe of 1.34. They have found one whose headline number is
mostly a function of where the window opens, which is a fact about the
sample and not about the market.

**A window count is a trial count.** :attr:`SensitivityReport.trials` exists
so it can be handed to :func:`mdnorm.deflated_sharpe_ratio` rather than left
in somebody's memory of how the window was picked. Quoting the best of
twenty-five windows without deflating for twenty-five trials is the same
error as quoting the best of twenty-five strategies, and it is harder to see
because only one strategy was ever written down.

**Shorter windows are noisier, and that is part of what you are seeing.** A
metric over a third of the data has roughly √3 times the standard error, so
some of the spread is sampling noise rather than instability. Nothing here
separates the two — that would need a model of the return process, and this
library does not have one. Every sample carries its observation count so the
shrinkage is visible, and the honest reading is that a wide spread is
consistent with instability rather than proof of it.

**The sensitivity is usually to a handful of observations.** If removing the
first four months changes the answer, look at what was in those four months
before concluding anything about regimes — :mod:`mdnorm.extremes` counts how
few observations a result rests on, and the answer is often small enough
that a moved start date is really a dropped outlier.

Nothing here picks a window for you, and there is no default length or step.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "WindowKind",
    "Window",
    "WindowResult",
    "SensitivityReport",
    "rolling_windows",
    "expanding_windows",
    "trimmed_starts",
    "trimmed_ends",
    "sweep",
]

_ZERO = Decimal(0)

Metric = Callable[[Sequence[Decimal]], Optional[Decimal]]


class WindowKind(str, Enum):
    """How a set of windows was generated."""

    ROLLING = "rolling"
    EXPANDING = "expanding"
    TRIMMED_START = "trimmed_start"
    TRIMMED_END = "trimmed_end"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class Window:
    """A half-open slice ``[start, end)`` of a series, by index."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("a window starts at or after the beginning")
        if self.end <= self.start:
            raise ValueError("a window ends after it starts")

    @property
    def observations(self) -> int:
        return self.end - self.start

    def slice(self, values: Sequence[Decimal]) -> Sequence[Decimal]:
        return values[self.start:self.end]


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One window and what the metric said about it.

    Named ``WindowResult`` rather than ``Sample`` because
    :mod:`mdnorm.seasonality` already exports a ``Sample``, and the sweep is
    called ``sweep`` rather than ``evaluate`` because
    :mod:`mdnorm.execution` already exports that. Two things sharing a name
    in one namespace is a bug waiting for a reader in a hurry.
    """

    window: Window
    value: Optional[Decimal]

    @property
    def observations(self) -> int:
        return self.window.observations


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """What one metric did across a set of windows."""

    kind: WindowKind
    samples: Tuple[WindowResult, ...]
    full: Optional[Decimal]

    @property
    def values(self) -> Tuple[Decimal, ...]:
        """The metric where it could be computed at all.

        A window too short for the metric contributes ``None`` and is left
        out here rather than counted as zero, which would drag every summary
        toward a value the metric never produced.
        """
        return tuple(s.value for s in self.samples if s.value is not None)

    @property
    def trials(self) -> int:
        """Windows that produced a value, plus the full sample.

        This is the number to hand :func:`mdnorm.deflated_sharpe_ratio`. A
        window was chosen; choosing it was a search over this many
        alternatives, whether or not anybody experienced it that way.
        """
        return len(self.values) + (0 if self.full is None else 1)

    @property
    def lowest(self) -> Optional[Decimal]:
        return min(self.values) if self.values else None

    @property
    def highest(self) -> Optional[Decimal]:
        return max(self.values) if self.values else None

    @property
    def median(self) -> Optional[Decimal]:
        if not self.values:
            return None
        ordered = sorted(self.values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return (ordered[mid - 1] + ordered[mid]) / 2

    @property
    def spread(self) -> Optional[Decimal]:
        """Highest minus lowest, in the metric's own units."""
        if not self.values:
            return None
        return max(self.values) - min(self.values)

    @property
    def positive(self) -> int:
        return sum(1 for v in self.values if v > 0)

    @property
    def share_positive(self) -> Optional[Decimal]:
        if not self.values:
            return None
        return Decimal(self.positive) / len(self.values)

    @property
    def changes_sign(self) -> bool:
        """Whether the metric is positive on some windows and negative on others.

        The sharpest form of the finding. A result that is profitable over
        one window and loss-making over another differs from its neighbours
        by more than a matter of degree.
        """
        vals = self.values
        return any(v > 0 for v in vals) and any(v < 0 for v in vals)

    @property
    def shortest(self) -> Optional[int]:
        return min((s.observations for s in self.samples), default=None)

    @property
    def longest(self) -> Optional[int]:
        return max((s.observations for s in self.samples), default=None)


def _check(n: int) -> None:
    if n <= 0:
        raise ValueError("a series needs at least one observation")


def rolling_windows(n: int, *, length: int, step: int) -> List[Window]:
    """Fixed-length windows advancing by ``step`` until the data runs out.

    No default length and no default step: how long a window has to be before
    it says anything is a property of the series and of the metric, not of
    this library.
    """
    _check(n)
    if length <= 0 or step <= 0:
        raise ValueError("length and step must be positive")
    if length > n:
        return []
    return [Window(s, s + length) for s in range(0, n - length + 1, step)]


def expanding_windows(n: int, *, minimum: int, step: int) -> List[Window]:
    """Windows that all begin at zero and end later and later.

    This is what a walk-forward report looks like, and its last window is the
    whole sample, so the series it produces is not independent of the
    headline figure.
    """
    _check(n)
    if minimum <= 0 or step <= 0:
        raise ValueError("minimum and step must be positive")
    if minimum > n:
        return []
    ends = list(range(minimum, n + 1, step))
    if ends[-1] != n:
        ends.append(n)
    return [Window(0, e) for e in ends]


def trimmed_starts(
    n: int, *, step: int, count: Optional[int] = None
) -> List[Window]:
    """Windows that drop more and more of the beginning and keep the end.

    The question a reader of a backtest actually wants answered: how much of
    this depends on the data happening to begin when it did. ``count``
    limits how many are produced; left out, it goes as far as the series
    allows.
    """
    _check(n)
    if step <= 0:
        raise ValueError("step must be positive")
    if count is not None and count < 0:
        raise ValueError("count must not be negative")
    starts = list(range(step, n, step))
    if count is not None:
        starts = starts[:count]
    return [Window(s, n) for s in starts]


def trimmed_ends(
    n: int, *, step: int, count: Optional[int] = None
) -> List[Window]:
    """Windows that keep the beginning and stop earlier and earlier.

    The mirror question, and the one that catches a result which depends on
    the last few weeks — including the ones that arrived after somebody
    decided the strategy was working.
    """
    _check(n)
    if step <= 0:
        raise ValueError("step must be positive")
    if count is not None and count < 0:
        raise ValueError("count must not be negative")
    ends = list(range(n - step, 0, -step))
    if count is not None:
        ends = ends[:count]
    return [Window(0, e) for e in ends]


def sweep(
    values: Sequence[Decimal],
    windows: Sequence[Window],
    metric: Metric,
    *,
    kind: WindowKind = WindowKind.EXPLICIT,
) -> SensitivityReport:
    """Run one metric over a set of windows and over the whole series.

    ``metric`` is any callable taking a sequence and returning a
    :class:`~decimal.Decimal` or ``None`` — :func:`mdnorm.sharpe_ratio` and
    the rest of :mod:`mdnorm.metrics` fit without adaptation. A metric that
    raises on a short window is caught and recorded as ``None``, because a
    window the metric cannot answer for is a fact about the window rather
    than an error in the sweep.

    Windows reaching past the end of the series raise, rather than being
    silently shortened. A window that does not describe the data it was
    built for is a bug in the caller's arithmetic, and quietly repairing it
    would hide the one thing worth knowing.
    """
    _check(len(values))
    for w in windows:
        if w.end > len(values):
            raise ValueError(
                f"window [{w.start}, {w.end}) reaches past {len(values)} "
                "observations")

    samples: List[WindowResult] = []
    for w in windows:
        try:
            value = metric(w.slice(values))
        except (ValueError, ZeroDivisionError, ArithmeticError):
            value = None
        samples.append(WindowResult(window=w, value=value))

    try:
        full = metric(values)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        full = None

    return SensitivityReport(kind=kind, samples=tuple(samples), full=full)
