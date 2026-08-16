"""market-data-normalizer (mdnorm).

Normalize heterogeneous market-data feeds (CSV, exchange WebSocket JSON, FIX)
into a single, exchange-agnostic :class:`MarketEvent` schema.
"""
from __future__ import annotations

from .adjust import (
    Action,
    ActionKind,
    AdjustMethod,
    adjust_bars,
    adjust_events,
    adjustment_at,
    dividend,
    read_actions_csv,
    roll,
    split,
)
from .book import BookDelta, OrderBook, SequenceGapError, replay_book
from .consolidate import CONSOLIDATED, Consolidator, VenueTop, consolidate
from .execution import (
    ExecutionSummary,
    Fill,
    average_fill_price,
    evaluate,
    exclude_fills,
    implementation_shortfall_bps,
    participation_rate,
    slippage_bps,
    twap,
    vwap,
)
from .bars import (
    Bar,
    count_bars,
    dollar_bars,
    fill_gaps,
    imbalance_bars,
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
from .micro import (
    SideRule,
    effective_spreads,
    infer_sides,
    mean_effective_spread,
    quote_rule,
    roll_spread,
    signed_volume,
    tick_rule,
    trade_imbalance,
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

__version__ = "1.8.0"

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
    "imbalance_bars",
    "BookDelta",
    "OrderBook",
    "SequenceGapError",
    "replay_book",
    "Consolidator",
    "consolidate",
    "VenueTop",
    "CONSOLIDATED",
    "Fill",
    "ExecutionSummary",
    "vwap",
    "twap",
    "exclude_fills",
    "participation_rate",
    "average_fill_price",
    "slippage_bps",
    "implementation_shortfall_bps",
    "evaluate",
    "SideRule",
    "infer_sides",
    "tick_rule",
    "quote_rule",
    "signed_volume",
    "trade_imbalance",
    "effective_spreads",
    "mean_effective_spread",
    "roll_spread",
    "Action",
    "ActionKind",
    "AdjustMethod",
    "split",
    "dividend",
    "roll",
    "adjust_events",
    "adjust_bars",
    "adjustment_at",
    "read_actions_csv",
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
