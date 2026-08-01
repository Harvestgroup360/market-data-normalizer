"""market-data-normalizer (mdnorm).

Normalize heterogeneous market-data feeds (CSV, exchange WebSocket JSON, FIX)
into a single, exchange-agnostic :class:`MarketEvent` schema.
"""
from __future__ import annotations

from .bars import Bar, fill_gaps, resample_bars, time_bars
from .normalizers import (
    from_csv_quote,
    from_csv_row,
    from_fix,
    from_ws_json,
    from_ws_quote,
)
from .quality import QualityIssue, clean, find_issues
from .records import bar_to_dict, event_to_dict, to_records
from .schema import EventType, MarketEvent, Side
from .symbols import canonical_symbol
from .timeutil import epoch_to_ns, fix_utc_to_ns, iso_to_ns

__version__ = "0.7.0"

__all__ = [
    "MarketEvent",
    "EventType",
    "Side",
    "from_csv_row",
    "from_ws_json",
    "from_fix",
    "from_ws_quote",
    "from_csv_quote",
    "Bar",
    "time_bars",
    "resample_bars",
    "fill_gaps",
    "QualityIssue",
    "find_issues",
    "clean",
    "event_to_dict",
    "bar_to_dict",
    "to_records",
    "canonical_symbol",
    "epoch_to_ns",
    "iso_to_ns",
    "fix_utc_to_ns",
]
