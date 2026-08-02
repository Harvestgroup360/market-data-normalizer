"""Consolidate and clean up multiple event streams.

Real setups pull from several venues and reconnect often, so you end up with
interleaved feeds and replayed duplicates. These helpers merge streams into one
chronological timeline and drop exact duplicate events.
"""
from __future__ import annotations

from typing import Iterable, List

from .schema import MarketEvent


def merge_streams(*streams: Iterable[MarketEvent]) -> List[MarketEvent]:
    """Merge several event streams into one list ordered by timestamp.

    The sort is stable, so events sharing a timestamp keep the relative order
    in which the streams were passed.
    """
    merged: List[MarketEvent] = []
    for s in streams:
        merged.extend(s)
    merged.sort(key=lambda e: e.ts_ns)
    return merged


def dedupe(events: Iterable[MarketEvent]) -> List[MarketEvent]:
    """Drop exact duplicate events, preserving first-seen order.

    ``MarketEvent`` is a frozen dataclass, so two events are duplicates when all
    their fields match — exactly what a reconnect/replay produces.
    """
    seen: set = set()
    out: List[MarketEvent] = []
    for e in events:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out
