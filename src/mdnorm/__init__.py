"""market-data-normalizer (mdnorm).

Normalize heterogeneous market-data feeds (CSV, exchange WebSocket JSON, FIX)
into a single, exchange-agnostic :class:`MarketEvent` schema.
"""
from __future__ import annotations

from .bars import (
    Bar,
    count_bars,
    dollar_bars,
    fill_gaps,
    resample_bars,
    time_bars,
    volume_bars,
)
from .csvio import iter_csv_trades, read_csv_trades, write_records_csv
from .jsonl import event_from_dict, iter_jsonl_events, read_jsonl_events, write_jsonl
from .normalizers import (
    from_csv_quote,
    from_csv_row,
    from_fix,
    from_ws_json,
    from_ws_quote,
)
from .pipeline import Pipeline
from .quality import QualityIssue, clean, find_issues
from .records import bar_to_dict, event_to_dict, to_records
from .schema import EventType, MarketEvent, Side
from .sessions import (
    US_EQUITY_RTH,
    US_FUTURES_OVERNIGHT,
    WEEKDAYS,
    Session,
    filter_session,
    group_by_session_date,
    in_session,
    parse_session,
    session_date,
)
from .streams import dedupe, merge_streams
from .symbols import canonical_symbol
from .timeutil import epoch_to_ns, fix_utc_to_ns, iso_to_ns

__version__ = "1.3.1"

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
    "count_bars",
    "volume_bars",
    "dollar_bars",
    "Session",
    "in_session",
    "filter_session",
    "session_date",
    "group_by_session_date",
    "parse_session",
    "WEEKDAYS",
    "US_EQUITY_RTH",
    "US_FUTURES_OVERNIGHT",
    "QualityIssue",
    "find_issues",
    "clean",
    "event_to_dict",
    "bar_to_dict",
    "to_records",
    "merge_streams",
    "dedupe",
    "read_csv_trades",
    "iter_csv_trades",
    "write_records_csv",
    "read_jsonl_events",
    "iter_jsonl_events",
    "write_jsonl",
    "event_from_dict",
    "Pipeline",
    "canonical_symbol",
    "epoch_to_ns",
    "iso_to_ns",
    "fix_utc_to_ns",
]
