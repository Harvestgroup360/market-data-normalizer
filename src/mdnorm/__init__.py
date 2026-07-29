"""market-data-normalizer (mdnorm).

Normalize heterogeneous market-data feeds (CSV, exchange WebSocket JSON, FIX)
into a single, exchange-agnostic :class:`MarketEvent` schema.
"""
from __future__ import annotations

from .bars import Bar, time_bars
from .normalizers import (
    from_csv_quote,
    from_csv_row,
    from_fix,
    from_ws_json,
    from_ws_quote,
)
from .schema import EventType, MarketEvent, Side
from .symbols import canonical_symbol
from .timeutil import epoch_to_ns, fix_utc_to_ns, iso_to_ns

__version__ = "0.3.0"

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
    "canonical_symbol",
    "epoch_to_ns",
    "iso_to_ns",
    "fix_utc_to_ns",
]
