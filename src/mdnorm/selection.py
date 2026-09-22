"""Picking the best backtest is a procedure, and procedures can be tested.

Every research process ends with a choice: of the variants tried, keep the
one that looked best. :func:`mdnorm.metrics.deflated_sharpe_ratio` asks how
good that winner would have looked by luck alone. This module asks a
different question, and one that needs no distributional assumption: **does
the procedure of choosing the in-sample winner pick something that does well
out of sample?**

The method is combinatorially symmetric cross-validation (Bailey, Borwein,
López de Prado and Zhu, 2017). Cut the sample into an even number of equal
blocks. For every way of using half of them as the in-sample period and the
other half as the out-of-sample one, find the variant that ranked first
in-sample and record where it ranked out of sample::

    from mdnorm import cscv

    rep = cscv(variants, blocks=10, metric="sharpe", ddof=1)
    rep.pbo                    # share of splits where the winner fell below
                               # the out-of-sample median
    rep.mean_is_best           # how good the winner looked
    rep.mean_oos_of_is_best    # what it went on to do

**The probability of backtest overfitting is a property of the search, not of
a strategy.** On variants that share the same true edge it is at least one
half, and usually above it: the two halves of every split are complements, so
a variant that looked best in one has, for a given whole-sample result, done
relatively worse in the other. On the worked example — twenty variants with
identical edge — it is 0.75. Add one genuinely better variant and it falls, to
0.39 on the same data. A reader shown one Sharpe ratio is being shown the
output of a procedure, and this is the measurement of the procedure itself.

**The in-sample figure of the winner is always flattering, by construction.**
It is the maximum of many noisy estimates. What matters is the out-of-sample
figure of the same variant, and the degradation slope — out-of-sample against
in-sample across splits — says whether a better-looking winner goes on to do
better or worse. On noise the slope is typically negative.

**Blocks keep time together.** The sample is split into contiguous blocks
rather than shuffled observations, so serial structure inside a block is
preserved. The block count must divide the sample exactly; a remainder is
refused rather than dropped, because dropping it silently changes the sample
the answer describes.

There is no default block count and no default metric. ``blocks`` decides how
many splits exist and how long each half is; ``metric`` decides what "best"
means. Both are the caller's decision, and both change the answer.

Arithmetic runs at forty significant digits. Each variant's per-block sums are
computed once, so the cost grows with the number of splits rather than with
the number of splits times the length of the sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import combinations
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "SelectionSplit",
    "SelectionReport",
    "block_splits",
    "cscv",
]

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_PRECISION = 40
_METRICS = ("mean", "sharpe")


def block_splits(blocks: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Every way of choosing half of ``blocks`` as in-sample.

    Returns ``(in_sample, out_of_sample)`` pairs of block indices, in
    lexicographic order of the in-sample half. There are
    ``C(blocks, blocks / 2)`` of them: 252 for ten blocks, 12,870 for sixteen.
    """
    if blocks < 2 or blocks % 2:
        raise ValueError(f"blocks must be an even number of at least 2, got {blocks}")
    everything = set(range(blocks))
    return [(is_, tuple(sorted(everything - set(is_))))
            for is_ in combinations(range(blocks), blocks // 2)]


@dataclass(frozen=True, slots=True)
class SelectionSplit:
    """One in-sample / out-of-sample split and what the winner did in it."""

    in_sample: Tuple[int, ...]
    winner: str
    is_metric: Decimal
    oos_metric: Decimal
    oos_rank: int          # 1 = worst out of sample, n = best
    variants: int

    @property
    def relative_rank(self) -> Decimal:
        """``oos_rank / (variants + 1)``, strictly between zero and one."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return Decimal(self.oos_rank) / Decimal(self.variants + 1)

    @property
    def logit(self) -> Decimal:
        """Log-odds of the relative rank. At or below zero: median or worse."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            w = self.relative_rank
            return (w / (_ONE - w)).ln()


@dataclass(frozen=True, slots=True)
class SelectionReport:
    """What choosing the in-sample winner did, across every split."""

    variants: int
    observations: int
    blocks: int
    metric: str
    splits: Tuple[SelectionSplit, ...]
    mean_oos_all: Decimal

    @property
    def pbo(self) -> Decimal:
        """Share of splits in which the winner finished at or below the median.

        The probability of backtest overfitting. Near one half means choosing
        the in-sample winner is no better than choosing at random; near zero
        means the procedure generalises. It describes the search, not any one
        strategy in it.
        """
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            bad = sum(1 for s in self.splits if s.logit <= 0)
            return Decimal(bad) / Decimal(len(self.splits))

    @property
    def median_logit(self) -> Decimal:
        xs = sorted(s.logit for s in self.splits)
        n = len(xs)
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / _TWO

    @property
    def mean_is_best(self) -> Decimal:
        """Average in-sample metric of the winner. Flattering by construction."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return sum((s.is_metric for s in self.splits), _ZERO) / len(self.splits)

    @property
    def mean_oos_of_is_best(self) -> Decimal:
        """Average out-of-sample metric of the same winner."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            return sum((s.oos_metric for s in self.splits), _ZERO) / len(self.splits)

    @property
    def oos_loss_share(self) -> Decimal:
        """Share of splits in which the winner's out-of-sample metric was below zero."""
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            bad = sum(1 for s in self.splits if s.oos_metric < 0)
            return Decimal(bad) / Decimal(len(self.splits))

    @property
    def degradation_slope(self) -> Optional[Decimal]:
        """OLS slope of the winner's out-of-sample metric on its in-sample one.

        Negative means a better-looking winner went on to do worse. ``None``
        when the winners' in-sample metrics do not vary.
        """
        xs = [s.is_metric for s in self.splits]
        ys = [s.oos_metric for s in self.splits]
        with localcontext() as ctx:
            ctx.prec = _PRECISION
            n = Decimal(len(xs))
            mx, my = sum(xs, _ZERO) / n, sum(ys, _ZERO) / n
            sxx = sum(((x - mx) ** 2 for x in xs), _ZERO)
            if sxx == 0:
                return None
            return sum(((x - mx) * (y - my) for x, y in zip(xs, ys)), _ZERO) / sxx

    @property
    def chosen(self) -> Dict[str, int]:
        """How many splits each variant won in-sample. Unchosen variants omitted."""
        out: Dict[str, int] = {}
        for s in self.splits:
            out[s.winner] = out.get(s.winner, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


def _metric(total: Decimal, squares: Decimal, count: int, *, metric: str,
            ddof: Optional[int]) -> Optional[Decimal]:
    n = Decimal(count)
    mean = total / n
    if metric == "mean":
        return mean
    assert ddof is not None
    if count - ddof < 1:
        return None
    var = (squares - n * mean * mean) / Decimal(count - ddof)
    if var <= 0:
        return None
    return mean / var.sqrt()


def cscv(variants: Mapping[str, Sequence[Decimal]], *, blocks: int, metric: str,
         ddof: Optional[int] = None) -> SelectionReport:
    """Combinatorially symmetric cross-validation over a set of variants.

    ``variants`` maps a name to a per-period return series; every series must
    cover the same periods. ``metric`` is ``"mean"`` or ``"sharpe"``
    (per-period, no hurdle), and ``ddof`` is required for ``"sharpe"``.

    Ties in-sample go to the first name in sorted order, and the out-of-sample
    rank counts only variants strictly below the winner, so a tie never
    flatters the winner. Both rules are stated because both affect the answer
    on short samples.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric is one of {_METRICS}, got {metric!r}")
    if metric == "sharpe" and ddof is None:
        raise ValueError("a Sharpe ratio needs ddof; there is no default")
    if len(variants) < 2:
        raise ValueError("choosing a winner needs at least two variants")
    names = sorted(variants)
    lengths = {len(variants[k]) for k in names}
    if len(lengths) != 1:
        raise ValueError(
            f"the variants cover different numbers of periods: {sorted(lengths)}. "
            "They must be the same periods for a split to mean the same thing "
            "for each of them.")
    n = lengths.pop()
    splits_idx = block_splits(blocks)
    if n % blocks:
        raise ValueError(
            f"{n} observations do not divide into {blocks} equal blocks; "
            f"{n % blocks} would be left over. Trim the sample deliberately "
            "rather than letting a remainder be dropped.")
    size = n // blocks

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        sums: Dict[str, List[Decimal]] = {}
        sqs: Dict[str, List[Decimal]] = {}
        for k in names:
            xs = [Decimal(x) for x in variants[k]]
            sums[k] = [sum(xs[b * size:(b + 1) * size], _ZERO) for b in range(blocks)]
            sqs[k] = [sum((x * x for x in xs[b * size:(b + 1) * size]), _ZERO)
                      for b in range(blocks)]

        def value(k: str, idx: Tuple[int, ...], label: str) -> Decimal:
            v = _metric(sum((sums[k][b] for b in idx), _ZERO),
                        sum((sqs[k][b] for b in idx), _ZERO),
                        size * len(idx), metric=metric, ddof=ddof)
            if v is None:
                raise ArithmeticError(
                    f"variant {k!r} has no {metric} on {label} blocks {idx}: "
                    "no dispersion, or too few observations for ddof.")
            return v

        out: List[SelectionSplit] = []
        oos_all = _ZERO
        for is_idx, oos_idx in splits_idx:
            is_vals = {k: value(k, is_idx, "in-sample") for k in names}
            winner = max(names, key=lambda k: (is_vals[k], -names.index(k)))
            oos_vals = {k: value(k, oos_idx, "out-of-sample") for k in names}
            w_oos = oos_vals[winner]
            rank = 1 + sum(1 for k in names if oos_vals[k] < w_oos)
            oos_all += sum(oos_vals.values(), _ZERO) / Decimal(len(names))
            out.append(SelectionSplit(in_sample=is_idx, winner=winner,
                             is_metric=is_vals[winner], oos_metric=w_oos,
                             oos_rank=rank, variants=len(names)))
        return SelectionReport(variants=len(names), observations=n,
                               blocks=blocks, metric=metric, splits=tuple(out),
                               mean_oos_all=oos_all / Decimal(len(out)))
