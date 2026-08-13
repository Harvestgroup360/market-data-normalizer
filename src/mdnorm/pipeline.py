"""Composable processing pipelines.

Every real ingestion job wires the same steps together: drop duplicates,
clean bad ticks, aggregate to bars, resample, fill gaps. :class:`Pipeline`
lets you declare that chain once and reuse it across venues and files,
instead of hand-wiring the calls at every call site.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Callable, List, Sequence, Tuple

from .adjust import Action, AdjustMethod
from .adjust import adjust_bars as _adjust_bars
from .adjust import adjust_events as _adjust_events
from .bars import count_bars as _count_bars
from .bars import dollar_bars as _dollar_bars
from .bars import fill_gaps as _fill_gaps
from .bars import imbalance_bars as _imbalance_bars
from .bars import resample_bars as _resample_bars
from .bars import time_bars as _time_bars
from .bars import volume_bars as _volume_bars
from .micro import SideRule
from .micro import infer_sides as _infer_sides
from .quality import QualityIssue
from .quality import clean as _clean
from .schema import MarketEvent
from .sessions import Session
from .sessions import filter_session as _filter_session
from .streams import dedupe as _dedupe

Step = Callable[[list], list]


class Pipeline:
    """A declarative, reusable chain of processing steps.

    Build once, run on any event sequence::

        from decimal import Decimal
        from mdnorm import Pipeline

        pipe = (
            Pipeline()
            .dedupe()
            .clean(max_return=Decimal("0.1"))
            .time_bars(60_000_000_000)   # 1-minute bars
            .fill_gaps()
        )
        bars = pipe.run(events)
        print(pipe.last_issues)          # QualityIssue report from clean()

    Steps run in the order they were added. Event-level steps
    (``dedupe``, ``clean``) must come before bar-level steps
    (``time_bars``, then optionally ``resample`` / ``fill_gaps``).
    """

    def __init__(self) -> None:
        self._steps: List[Tuple[str, Step]] = []
        self.last_issues: List[QualityIssue] = []

    # -- event-level steps -------------------------------------------------

    def dedupe(self) -> "Pipeline":
        """Drop exact duplicate events (reconnects/replays)."""
        self._steps.append(("dedupe", lambda data: list(_dedupe(data))))
        return self

    def clean(self, *, max_return: Decimal = Decimal("0.1")) -> "Pipeline":
        """Drop outlier/non-positive ticks; report goes to ``last_issues``."""

        def step(data: list) -> list:
            cleaned, issues = _clean(data, max_return=max_return)
            self.last_issues = issues
            return cleaned

        self._steps.append(("clean", step))
        return self

    def session(self, session: Session) -> "Pipeline":
        """Keep only items inside a trading session (see :mod:`mdnorm.sessions`).

        Works on events before aggregation and on bars after it.
        """
        self._steps.append(
            ("session", lambda data: _filter_session(data, session))
        )
        return self

    def adjust(
        self,
        actions: Sequence[Action],
        *,
        method: AdjustMethod = AdjustMethod.RATIO,
    ) -> "Pipeline":
        """Back-adjust for splits, dividends and rolls (:mod:`mdnorm.adjust`).

        Works on events before aggregation and on bars after it; the step
        dispatches on what it is handed.
        """

        def step(data: list) -> list:
            if data and isinstance(data[0], MarketEvent):
                return _adjust_events(data, actions, method=method)
            return _adjust_bars(data, actions, method=method)

        self._steps.append(("adjust", step))
        return self

    def infer_sides(
        self,
        *,
        rule: SideRule = SideRule.LEE_READY,
        lag_ns: int = 0,
        overwrite: bool = False,
    ) -> "Pipeline":
        """Classify the aggressor side of trades (:mod:`mdnorm.micro`).

        Runs before aggregation, so imbalance bars and signed-volume features
        downstream have a side to work with.
        """
        self._steps.append(
            ("infer_sides",
             lambda data: _infer_sides(
                 data, rule=rule, lag_ns=lag_ns, overwrite=overwrite))
        )
        return self

    # -- bar-level steps ---------------------------------------------------

    def time_bars(self, interval_ns: int) -> "Pipeline":
        """Aggregate trade events into fixed-interval OHLCV bars."""
        self._steps.append(
            ("time_bars", lambda data: _time_bars(data, interval_ns))
        )
        return self

    def count_bars(self, every: int) -> "Pipeline":
        """Aggregate trades into tick bars of ``every`` trades each."""
        self._steps.append(
            ("count_bars", lambda data: _count_bars(data, every))
        )
        return self

    def volume_bars(self, min_volume: Decimal) -> "Pipeline":
        """Aggregate trades into volume bars of >= ``min_volume``."""
        self._steps.append(
            ("volume_bars", lambda data: _volume_bars(data, min_volume))
        )
        return self

    def dollar_bars(self, min_notional: Decimal) -> "Pipeline":
        """Aggregate trades into dollar bars of >= ``min_notional``."""
        self._steps.append(
            ("dollar_bars", lambda data: _dollar_bars(data, min_notional))
        )
        return self

    def imbalance_bars(self, threshold: Decimal, *, by: str = "volume") -> "Pipeline":
        """Aggregate trades into order-flow imbalance bars."""
        self._steps.append(
            ("imbalance_bars", lambda data: _imbalance_bars(data, threshold, by=by))
        )
        return self

    def resample(self, interval_ns: int) -> "Pipeline":
        """Downsample bars to a coarser interval (after ``time_bars``)."""
        self._steps.append(
            ("resample", lambda data: _resample_bars(data, interval_ns))
        )
        return self

    def fill_gaps(self) -> "Pipeline":
        """Insert flat zero-volume bars for missing intervals."""
        self._steps.append(("fill_gaps", lambda data: _fill_gaps(data)))
        return self

    # -- extensibility -----------------------------------------------------

    def apply(self, name: str, fn: Step) -> "Pipeline":
        """Append a custom step: any callable ``list -> list``."""
        self._steps.append((name, fn))
        return self

    # -- execution ---------------------------------------------------------

    @property
    def steps(self) -> List[str]:
        """Names of the configured steps, in execution order."""
        return [name for name, _ in self._steps]

    def run(self, events: Sequence[MarketEvent]) -> list:
        """Execute the chain on ``events`` and return the final output."""
        data: list = list(events)
        for _, fn in self._steps:
            data = fn(data)
        return data
