"""Command-line interface.

The common conversions, as a zero-dependency CLI::

    mdnorm bars trades.csv --venue binance --interval 1m -o bars.csv
    mdnorm quality trades.csv --venue binance --max-gap 5m
    mdnorm convert trades.csv --venue binance -o trades.jsonl

Input format is inferred from the extension: ``.jsonl`` / ``.ndjson`` files
are read as NDJSON (already-normalized events), anything else as a trades
CSV. Output format likewise: ``.jsonl`` / ``.ndjson`` writes NDJSON,
anything else CSV.
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from typing import List, Optional

from . import __version__
from .bars import fill_gaps, time_bars
from .csvio import read_csv_trades, write_records_csv
from .jsonl import read_jsonl_events, write_jsonl
from .quality import clean, find_issues
from .schema import MarketEvent
from .streams import dedupe

_UNIT_NS = {
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 3_600 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
}


def parse_interval(text: str) -> int:
    """Parse ``30s`` / ``1m`` / ``4h`` / ``1d`` into nanoseconds."""
    t = text.strip().lower()
    if len(t) < 2 or t[-1] not in _UNIT_NS or not t[:-1].isdigit():
        raise argparse.ArgumentTypeError(
            f"invalid interval {text!r} (use e.g. 30s, 1m, 4h, 1d)"
        )
    n = int(t[:-1])
    if n <= 0:
        raise argparse.ArgumentTypeError("interval must be positive")
    return n * _UNIT_NS[t[-1]]


def _is_jsonl(path: str) -> bool:
    return path.lower().endswith(
        (".jsonl", ".ndjson", ".jsonl.gz", ".ndjson.gz")
    )


def _read_events(args: argparse.Namespace) -> List[MarketEvent]:
    if _is_jsonl(args.input):
        return read_jsonl_events(args.input)
    return read_csv_trades(args.input, venue=args.venue, ts_unit=args.ts_unit)


def _write(items: list, path: str, *, as_float: bool) -> int:
    if _is_jsonl(path):
        return write_jsonl(items, path, as_float=as_float)
    return write_records_csv(items, path, as_float=as_float)


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="input file (.csv, .jsonl, .ndjson; .gz accepted)")
    p.add_argument(
        "--venue", default="csv",
        help="venue label stamped on events read from CSV (default: csv)",
    )
    p.add_argument(
        "--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
        help="parse CSV timestamps as epoch in this unit (default: ISO-8601)",
    )


def _cmd_bars(args: argparse.Namespace) -> int:
    events = _read_events(args)
    if args.dedupe:
        events = list(dedupe(events))
    if args.clean:
        events, issues = clean(events, max_return=Decimal(args.max_return))
        if issues:
            print(f"clean: dropped/flagged {len(issues)} issue(s)",
                  file=sys.stderr)
    bars = time_bars(events, args.interval)
    if args.fill_gaps:
        bars = fill_gaps(bars)
    n = _write(bars, args.output, as_float=args.as_float)
    print(f"wrote {n} bar(s) -> {args.output}")
    return 0


def _cmd_quality(args: argparse.Namespace) -> int:
    events = _read_events(args)
    issues = find_issues(
        events,
        max_return=Decimal(args.max_return),
        max_gap_ns=args.max_gap,
    )
    if not issues:
        print(f"{len(events)} event(s), no issues found")
        return 0
    by_kind: dict = {}
    for iss in issues:
        by_kind[iss.kind] = by_kind.get(iss.kind, 0) + 1
    print(f"{len(events)} event(s), {len(issues)} issue(s):")
    for kind in sorted(by_kind):
        print(f"  {kind}: {by_kind[kind]}")
    if args.verbose:
        for iss in issues:
            print(f"  [{iss.kind}] #{iss.index}: {iss.detail}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    events = _read_events(args)
    if args.dedupe:
        events = list(dedupe(events))
    n = _write(events, args.output, as_float=args.as_float)
    print(f"wrote {n} event(s) -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mdnorm",
        description="Normalize market-data files and aggregate OHLCV bars.",
    )
    parser.add_argument(
        "--version", action="version", version=f"mdnorm {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bars = sub.add_parser("bars", help="aggregate trades into OHLCV bars")
    _add_input_args(p_bars)
    p_bars.add_argument("-o", "--output", required=True,
                        help="output file (.csv, .jsonl, .ndjson; .gz accepted)")
    p_bars.add_argument("--interval", type=parse_interval, required=True,
                        help="bar interval, e.g. 30s, 1m, 4h, 1d")
    p_bars.add_argument("--dedupe", action="store_true",
                        help="drop exact duplicate events first")
    p_bars.add_argument("--clean", action="store_true",
                        help="drop outlier/non-positive ticks first")
    p_bars.add_argument("--max-return", default="0.1",
                        help="outlier threshold for --clean (default: 0.1)")
    p_bars.add_argument("--fill-gaps", action="store_true",
                        help="insert flat bars for empty intervals")
    p_bars.add_argument("--as-float", action="store_true",
                        help="write numeric values instead of strings")
    p_bars.set_defaults(func=_cmd_bars)

    p_q = sub.add_parser("quality", help="scan a file for data-quality issues")
    _add_input_args(p_q)
    p_q.add_argument("--max-return", default="0.1",
                     help="outlier threshold (default: 0.1)")
    p_q.add_argument("--max-gap", type=parse_interval, default=None,
                     help="flag forward gaps larger than this, e.g. 5m")
    p_q.add_argument("-v", "--verbose", action="store_true",
                     help="print every issue, not just the summary")
    p_q.set_defaults(func=_cmd_quality)

    p_c = sub.add_parser("convert", help="convert between CSV and NDJSON")
    _add_input_args(p_c)
    p_c.add_argument("-o", "--output", required=True,
                     help="output file (.csv, .jsonl, .ndjson; .gz accepted)")
    p_c.add_argument("--dedupe", action="store_true",
                     help="drop exact duplicate events")
    p_c.add_argument("--as-float", action="store_true",
                     help="write numeric values instead of strings")
    p_c.set_defaults(func=_cmd_convert)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
