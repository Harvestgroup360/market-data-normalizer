"""Every cleaning decision is a fork, and nobody counts the forks.

A result arrives as one number produced by one pipeline. That pipeline
contains a dozen decisions that were made once and never revisited: which
staleness threshold counted as stale, whether extremes were clipped at four
sigma or five, whether the halts file was applied, which side of a session
boundary an overnight bar belonged to. Each was defensible. Together they
define a grid, and the published number is one cell of it::

    from mdnorm import Choice, specifications, explore, sharpe_ratio

    choices = [
        Choice("clip", [("none", None), ("5 sigma", Decimal(5))]),
        Choice("stale", [("keep", False), ("drop", True)]),
        Choice("halts", [("ignore", False), ("exclude", True)]),
    ]
    curve = explore(specifications(choices), run_pipeline)
    curve.highest          # 1.21 — clip at 5, drop stale, exclude halts
    curve.lowest           # 0.44 — keep everything
    curve.changes_sign     # False
    curve.trials           # 8

**The spread across the grid is the finding.** Eight defensible pipelines that
produce Sharpe ratios between 0.44 and 1.21 have not measured a strategy with
a Sharpe of 1.21. They have measured one whose headline figure is substantially
a function of decisions taken during cleaning, which is a fact about the
pipeline rather than about the market.

**A specification count is a trial count, with a caveat we will not bury.**
:attr:`SpecCurve.trials` can be handed to :func:`mdnorm.deflated_sharpe_ratio`
the way :attr:`mdnorm.SensitivityReport.trials` can. But specifications built
from a shared grid are *not independent* — two pipelines differing in one
choice out of six see almost the same data — so the raw count overstates how
much searching actually happened. Deflating by it is conservative, which is
the direction to be wrong in, and it is still wrong. Nothing here estimates
the effective number, because doing so needs a model of how the choices
correlate and this library does not have one.

**Which decision is doing the work is usually answerable.**
:func:`choice_effect` reports the median result under each option of a single
choice, so a grid whose entire spread comes from one threshold says so. That
is the useful output most of the time: not *the number is unstable* but *the
number is a function of the clipping threshold, which we picked in a meeting*.

Nothing here picks a specification, ranks one above another, or suggests which
is correct. The grid is supplied by the caller, and so is the function that
turns a specification into a number.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Choice",
    "Specification",
    "SpecResult",
    "SpecCurve",
    "ChoiceEffect",
    "specifications",
    "explore",
    "choice_effect",
    "dominant_choice",
]

Evaluate = Callable[["Specification"], Optional[Decimal]]


@dataclass(frozen=True, slots=True)
class Choice:
    """One decision point in a pipeline, and the options it could take.

    Options are ``(label, value)`` pairs. The label is what appears in a
    report and the value is whatever the caller's pipeline needs — a
    threshold, a flag, a callable. Nothing here inspects the value.
    """

    name: str
    options: Tuple[Tuple[str, Any], ...]

    def __init__(self, name: str, options: Sequence[Tuple[str, Any]]) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", tuple(options))
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a choice needs a name")
        if not self.options:
            raise ValueError(
                f"choice {self.name!r} has no options; a decision with one "
                "outcome was not a decision")
        labels = [label for label, _ in self.options]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"choice {self.name!r} lists an option label twice, so a "
                "result could not be traced back to the option that produced "
                "it")

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(label for label, _ in self.options)


@dataclass(frozen=True, slots=True)
class Specification:
    """One cell of the grid: a label and a value for every choice."""

    selections: Tuple[Tuple[str, str], ...]
    assignments: Tuple[Tuple[str, Any], ...]

    @property
    def labels(self) -> Dict[str, str]:
        """Choice name to the label chosen for it."""
        return dict(self.selections)

    @property
    def values(self) -> Dict[str, Any]:
        """Choice name to the value chosen for it — what a pipeline reads."""
        return dict(self.assignments)

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def __str__(self) -> str:
        return ", ".join(f"{n}={label}" for n, label in self.selections)


@dataclass(frozen=True, slots=True)
class SpecResult:
    """One specification and what the pipeline said about it."""

    specification: Specification
    value: Optional[Decimal]


@dataclass(frozen=True, slots=True)
class SpecCurve:
    """What one metric did across a grid of defensible pipelines."""

    results: Tuple[SpecResult, ...]

    @property
    def values(self) -> Tuple[Decimal, ...]:
        """The metric where the pipeline produced one at all.

        A specification the pipeline could not answer for contributes ``None``
        and is left out rather than counted as zero, which would drag every
        summary toward a value nothing produced.
        """
        return tuple(r.value for r in self.results if r.value is not None)

    @property
    def answered(self) -> int:
        return len(self.values)

    @property
    def unanswered(self) -> int:
        return len(self.results) - self.answered

    @property
    def trials(self) -> int:
        """Specifications that produced a value.

        Hand this to :func:`mdnorm.deflated_sharpe_ratio` in the knowledge
        that it is an **upper bound** on the effective number of independent
        trials: neighbouring cells of a shared grid differ in one choice and
        see nearly the same data. Deflating by it errs toward caution. It does
        not err toward being right, and nothing here estimates the effective
        count, which would need a model of how the choices correlate.
        """
        return self.answered

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
        return Decimal(self.positive) / self.answered

    @property
    def changes_sign(self) -> bool:
        """Whether the metric is positive under some pipelines and negative
        under others.

        The sharpest form of the finding: a result that is profitable after
        one defensible cleaning and loss-making after another differs from its
        neighbours by more than a matter of degree.
        """
        vals = self.values
        return any(v > 0 for v in vals) and any(v < 0 for v in vals)

    @property
    def best(self) -> Optional[SpecResult]:
        """The specification that produced the highest value.

        Named ``best`` for the number and not for the pipeline. Which cleaning
        is correct is not a question this library can answer, and the cell that
        flatters the result is the one most in need of a reason.
        """
        answered = [r for r in self.results if r.value is not None]
        if not answered:
            return None
        return max(answered, key=lambda r: r.value)  # type: ignore[arg-type,return-value]

    @property
    def worst(self) -> Optional[SpecResult]:
        answered = [r for r in self.results if r.value is not None]
        if not answered:
            return None
        return min(answered, key=lambda r: r.value)  # type: ignore[arg-type,return-value]

    @property
    def names(self) -> Tuple[str, ...]:
        """The choice names present in the grid, in their original order."""
        if not self.results:
            return ()
        return tuple(n for n, _ in self.results[0].specification.selections)


@dataclass(frozen=True, slots=True)
class ChoiceEffect:
    """How much one decision moved the answer, holding nothing else fixed."""

    name: str
    medians: Tuple[Tuple[str, Optional[Decimal]], ...]

    @property
    def median_map(self) -> Dict[str, Optional[Decimal]]:
        return dict(self.medians)

    @property
    def spread(self) -> Optional[Decimal]:
        """Highest option median minus lowest.

        A crude attribution and deliberately so: it marginalises over every
        other choice rather than isolating an effect, which is the right
        summary when the question is *how much does this decision move the
        published figure* and the wrong one for anything causal.
        """
        vals = [v for _, v in self.medians if v is not None]
        if len(vals) < 2:
            return None
        return max(vals) - min(vals)


def specifications(choices: Sequence[Choice]) -> List[Specification]:
    """Every combination of every choice, in a stable order.

    The count multiplies: six binary decisions are sixty-four pipelines, which
    is the point. No sampling and no default grid — which decisions belong in
    it is a property of the pipeline, and a library that guessed would be
    reporting on a pipeline it invented.
    """
    if not choices:
        raise ValueError(
            "a multiverse needs at least one choice; a pipeline with no "
            "decisions in it does not need this module")
    names = [c.name for c in choices]
    if len(set(names)) != len(names):
        raise ValueError("a choice name is listed twice")

    out: List[Specification] = []
    for combo in product(*(c.options for c in choices)):
        out.append(Specification(
            selections=tuple((c.name, label)
                             for c, (label, _) in zip(choices, combo)),
            assignments=tuple((c.name, value)
                              for c, (_, value) in zip(choices, combo)),
        ))
    return out


def explore(
    specs: Sequence[Specification], evaluate: Evaluate
) -> SpecCurve:
    """Run the caller's pipeline once per specification.

    ``evaluate`` takes a :class:`Specification` and returns a
    :class:`~decimal.Decimal` or ``None``. A specification the pipeline raises
    on is recorded as ``None`` rather than aborting the sweep: a combination of
    settings that cannot be run is a fact about the grid, and losing the other
    sixty-three results to it would be a poor trade.

    Nothing is run in parallel and nothing is cached. A grid that is expensive
    to evaluate is expensive because the pipeline is, and hiding that behind a
    pool here would only move the surprise.
    """
    if not specs:
        raise ValueError("a curve needs at least one specification")
    results: List[SpecResult] = []
    for spec in specs:
        try:
            value = evaluate(spec)
        except (ValueError, ZeroDivisionError, ArithmeticError, LookupError):
            value = None
        results.append(SpecResult(specification=spec, value=value))
    return SpecCurve(results=tuple(results))


def choice_effect(curve: SpecCurve, name: str) -> ChoiceEffect:
    """The median result under each option of one choice.

    Answers the question a reader of a wide spread actually has: *which
    decision is this sensitive to*. An option that no specification answered
    for carries ``None`` rather than a zero.
    """
    if name not in curve.names:
        raise ValueError(
            f"no choice named {name!r} in this curve; it has "
            f"{', '.join(repr(n) for n in curve.names) or 'none'}")

    buckets: Dict[str, List[Decimal]] = {}
    order: List[str] = []
    for r in curve.results:
        label = r.specification.labels[name]
        if label not in buckets:
            buckets[label] = []
            order.append(label)
        if r.value is not None:
            buckets[label].append(r.value)

    medians: List[Tuple[str, Optional[Decimal]]] = []
    for label in order:
        vals = sorted(buckets[label])
        if not vals:
            medians.append((label, None))
            continue
        mid = len(vals) // 2
        med = (vals[mid] if len(vals) % 2
               else (vals[mid - 1] + vals[mid]) / 2)
        medians.append((label, med))
    return ChoiceEffect(name=name, medians=tuple(medians))


def dominant_choice(curve: SpecCurve) -> Optional[ChoiceEffect]:
    """The choice whose options pull the median furthest apart.

    ``None`` when no choice separates the results at all, which happens when
    nothing was answered or when every option lands on the same median. That
    is a real outcome and worth reporting as one: a grid whose spread cannot
    be attributed to any single decision is telling you the decisions interact.
    """
    effects = [choice_effect(curve, n) for n in curve.names]
    ranked = [e for e in effects if e.spread is not None]
    if not ranked:
        return None
    best = max(ranked, key=lambda e: e.spread)  # type: ignore[arg-type,return-value]
    if best.spread == 0:
        return None
    return best
