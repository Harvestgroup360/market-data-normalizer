"""Labels, and splitting a time series without letting the answer leak.

Everything in :mod:`mdnorm.features` refuses to look forward. A label has to,
because a label *is* the future: the thing you are trying to predict. That
reversal is the whole difficulty of this module — the one series in a research
dataset that is allowed to see ahead is also the one that quietly contaminates
every split it touches::

    from mdnorm import forward_returns, purged_splits

    y = forward_returns(prices, horizon=5)
    for split in purged_splits(len(prices), n_splits=5, horizon=5, embargo=10):
        train, test = split.train, split.test

**A label is not a feature.** :func:`forward_returns` at index ``i`` looks at
``values[i + horizon]``. That is correct and intended, and it means the series
must never be fed back in as an input. Nothing in a type system stops you; the
only defence is that the function is in this module rather than in the feature
one, and that its name says what it does.

**A label with a horizon makes neighbouring rows overlap.** If the label at
``i`` spans the next five bars, then rows ``i`` through ``i + 5`` all describe
the same stretch of future. Put row ``i`` in the training set and row ``i + 3``
in the test set and the model has already been shown most of the answer. This
is the most common leak in financial machine learning and it survives every
shuffle-based defence, because the rows genuinely are different rows — they
merely share an outcome. :func:`purged_splits` removes training samples whose
label window reaches into the test block.

**Features have memory, so a gap after the test block is not enough.** A
rolling statistic computed just after a test period is built partly from
observations inside it. An embargo drops the training samples immediately
following each test block, and it should be at least as long as the longest
feature window in the dataset. It defaults to zero because the right value is a
property of your features, not of this function.

The purging and embargo scheme here follows the treatment in Marcos López de
Prado's *Advances in Financial Machine Learning* (2018), chapter 7.

Nothing here trains, tunes, or scores a model. It produces label series and
index sets.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from typing import List, Optional, Tuple

from .features import ReturnMethod, _PRECISION, _Series

__all__ = [
    "forward_returns",
    "Split",
    "purged_splits",
    "purged_train_test",
]


def forward_returns(
    values: _Series,
    *,
    horizon: int,
    method: ReturnMethod = ReturnMethod.SIMPLE,
) -> List[Optional[Decimal]]:
    """The return over the next ``horizon`` observations, as a label.

    ``out[i]`` is the return from ``values[i]`` to ``values[i + horizon]``, so
    it is knowable only at ``i + horizon``. The final ``horizon`` entries are
    ``None`` because their outcome has not happened yet — a real property of
    the data, and rows carrying it should be dropped before training rather
    than filled.

    This function looks forward on purpose. It is the only one in the library
    that does, and its output belongs on the left-hand side of a model. Using
    it as an input is not a subtle mistake; it is training on the answer.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    n = len(values)
    out: List[Optional[Decimal]] = []
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        for i in range(n):
            j = i + horizon
            if j >= n:
                out.append(None)
                continue
            a, b = values[i], values[j]
            if a is None or b is None or a <= 0 or b <= 0:
                out.append(None)
                continue
            try:
                out.append((b / a).ln() if method is ReturnMethod.LOG else b / a - 1)
            except (InvalidOperation, ZeroDivisionError):  # pragma: no cover
                out.append(None)
    return out


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test division, with what it had to throw away to be honest."""

    train: Tuple[int, ...]
    test: Tuple[int, ...]
    purged: int = 0
    embargoed: int = 0

    @property
    def discarded(self) -> int:
        """Training samples removed to keep the test block clean."""
        return self.purged + self.embargoed


def _one_split(
    n_samples: int, start: int, stop: int, horizon: int, embargo: int
) -> Split:
    """Build a split whose test block is ``[start, stop)``."""
    test = tuple(range(start, stop))
    train: List[int] = []
    purged = 0
    embargoed = 0
    for i in range(n_samples):
        if start <= i < stop:
            continue
        # A training sample before the block whose label window reaches into
        # it describes the same future the test block does.
        if i < start and i + horizon >= start:
            purged += 1
            continue
        # A training sample just after the block carries feature windows built
        # partly from observations inside it.
        if i >= stop and i < stop + embargo:
            embargoed += 1
            continue
        train.append(i)
    return Split(train=tuple(train), test=test, purged=purged, embargoed=embargoed)


def _validate(n_samples: int, horizon: int, embargo: int) -> None:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if embargo < 0:
        raise ValueError("embargo must be non-negative")


def purged_train_test(
    n_samples: int, *, test_fraction: float, horizon: int, embargo: int = 0
) -> Split:
    """A single chronological split: train first, test last.

    The test block is the final ``test_fraction`` of the samples. Training
    samples whose label window reaches into it are purged, and the embargo
    is irrelevant here because nothing follows the test block — it is included
    only so the same arguments work for both functions.
    """
    _validate(n_samples, horizon, embargo)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    n_test = max(1, int(round(n_samples * test_fraction)))
    if n_test >= n_samples:
        raise ValueError("test_fraction leaves no training samples")
    return _one_split(n_samples, n_samples - n_test, n_samples, horizon, embargo)


def purged_splits(
    n_samples: int, *, n_splits: int, horizon: int, embargo: int = 0
) -> List[Split]:
    """K contiguous test blocks in time order, each purged and embargoed.

    The blocks partition the sample range, so every observation is tested
    exactly once, and the test blocks stay contiguous rather than being drawn
    at random — a shuffled split of a series with overlapping labels leaks in
    both directions at once and is the standard way this goes wrong.

    Note that a fold's training set includes samples from *after* its test
    block. That is intentional for cross-validation, and it is also the reason
    the embargo exists; if you need strictly walk-forward evaluation, take the
    training indices below the test block yourself.
    """
    _validate(n_samples, horizon, embargo)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > n_samples:
        raise ValueError("n_splits cannot exceed n_samples")
    bounds = [(n_samples * k) // n_splits for k in range(n_splits + 1)]
    return [
        _one_split(n_samples, bounds[k], bounds[k + 1], horizon, embargo)
        for k in range(n_splits)
    ]
