"""Composable processing pipelines.

Every real ingestion job wires the same steps together: drop duplicates,
clean bad ticks, aggregate to bars, resample, fill gaps. :class:`Pipeline`
lets you declare that chain once and reuse it across venues and files,
instead of hand-wiring the calls at every call site.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Callable, List, Sequence, Tuple

from .bars import count_bars as _count_bars
from .bars import dollar_bars as _dollar_bars
from .bars import fill_gaps as _fill_gaps
from .bars import resample_bars as _resample_bars
from .bars import time_bars as _time_bars
from .bars import volume_bars as _volume_bars
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
