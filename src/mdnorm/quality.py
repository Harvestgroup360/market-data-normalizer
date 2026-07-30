"""Data-quality checks for normalized market-data streams.

Real feeds arrive with bad ticks, gaps, out-of-order records and the odd
zero/negative field. These helpers flag (and optionally drop) those so the
rest of your pipeline can assume clean input.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import List, Optional, Sequence, Tuple

from .schema import MarketEvent


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One detected problem, referencing the event's position in the input."""

    kind: str          # "outlier" | "gap" | "out_of_order" | "non_positive"
    index: int
    detail: str


def find_issues(
    events: Sequence[MarketEvent],
    *,
    max_return: Decimal = Decimal("0.1"),
    max_gap_ns: Optional[int] = None,
) -> List[QualityIssue]:
    """Scan events and return a list of quality issues.

    - ``out_of_order``: timestamp goes backwards vs the previous event.
    - ``gap``: forward time gap exceeds ``max_gap_ns`` (if given).
    - ``non_positive``: price <= 0 or size < 0.
    - ``outlier``: absolute tick-to-tick return exceeds ``max_return``
      (compared against the last valid price).
    """
    issues: List[QualityIssue] = []
    prev_ts: Optional[int] = None
    prev_price: Optional[Decimal] = None

    for i, e in enumerate(events):
        if prev_ts is not None:
            delta = e.ts_ns - prev_ts
            if delta < 0:
                issues.append(QualityIssue("out_of_order", i,
                                           f"ts {e.ts_ns} < previous {prev_ts}"))
            elif max_gap_ns is not None and delta > max_gap_ns:
                issues.append(QualityIssue("gap", i,
                                           f"gap {delta} ns > {max_gap_ns}"))

        if e.price is not None:
            if e.price <= 0:
                issues.append(QualityIssue("non_positive", i, f"price {e.price}"))
            elif prev_price is None:
                prev_price = e.price
            else:
                ret = abs(e.price / prev_price - 1)
                if ret > max_return:
                    # keep the last *good* price as the reference
                    issues.append(QualityIssue("outlier", i,
                                               f"return {ret:.4f} > {max_return}"))
                else:
                    prev_price = e.price

        if e.size is not None and e.size < 0:
            issues.append(QualityIssue("non_positive", i, f"size {e.size}"))

        prev_ts = e.ts_ns

    return issues


def clean(
    events: Sequence[MarketEvent],
    *,
    max_return: Decimal = Decimal("0.1"),
) -> Tuple[List[MarketEvent], List[QualityIssue]]:
    """Return ``(cleaned_events, issues)``.

    Events flagged as ``outlier`` or ``non_positive`` are dropped; timing
    issues (``gap``, ``out_of_order``) are reported but not removed.
    """
    events = list(events)
    issues = find_issues(events, max_return=max_return)
    drop = {iss.index for iss in issues if iss.kind in ("outlier", "non_positive")}
    cleaned = [e for i, e in enumerate(events) if i not in drop]
    return cleaned, issues
