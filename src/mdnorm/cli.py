"""Command-line interface.

The common conversions, as a zero-dependency CLI::

    mdnorm bars trades.csv --venue binance --interval 1m -o bars.csv
    mdnorm quality trades.csv --venue binance --max-gap 5m
    mdnorm convert trades.csv --venue binance -o trades.jsonl
    mdnorm bars trades.csv --interval 5m --session 09:30-16:00 --tz America/New_York -o rth.csv
    mdnorm bars trades.csv --interval 1d --actions splits.csv -o adjusted.csv
    mdnorm bars quotes_and_trades.jsonl --infer-sides --every-imbalance 500 -o imb.csv
    mdnorm book deltas.csv --symbol BTC-USD --venue binance -o quotes.jsonl
    mdnorm nbbo quotes.jsonl --symbol BTC-USD --max-age 2s -o top.jsonl
    mdnorm tca fills.csv --market tape.jsonl --decision-price 100
    mdnorm align BTC=btc.csv ETH=eth.csv --interval 1m --max-age 5m -o matrix.csv
    mdnorm features matrix.csv --returns log --zscore 60 --vol 60 -o feats.csv

Input format is inferred from the extension: ``.jsonl`` / ``.ndjson`` files
are read as NDJSON (already-normalized events), anything else as a trades
CSV. Output format likewise: ``.jsonl`` / ``.ndjson`` writes NDJSON,
anything else CSV.
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal, localcontext
from typing import List, Optional

from . import __version__
from .adjust import AdjustMethod, adjust_events, read_actions_csv
from .align import Field, align
from .features import periods_per_year
from .features import (ReturnMethod, realized_volatility, returns,
                       rolling_correlation, rolling_zscore)
from .bars import (count_bars, dollar_bars, fill_gaps, imbalance_bars,
                   time_bars, volume_bars)
from .book import BookDelta, OrderBook, replay_book
from .consolidate import Consolidator
from .execution import Fill, evaluate
from .csvio import read_csv_trades, write_records_csv
from .jsonl import read_jsonl_events, write_jsonl
from .micro import SideRule, infer_sides
from .quality import clean, find_issues
from .fileio import open_text
from .schema import MarketEvent, Side
from .timeutil import epoch_to_ns, iso_to_ns
from .sessions import filter_session, parse_session
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
        events = read_jsonl_events(args.input)
    else:
        events = read_csv_trades(
            args.input, venue=args.venue, ts_unit=args.ts_unit
        )
    if getattr(args, "session", None):
        session = parse_session(args.session, args.tz)
        kept = filter_session(events, session)
        dropped = len(events) - len(kept)
        if dropped:
            print(f"session: dropped {dropped} event(s) outside "
                  f"{args.session} {args.tz}", file=sys.stderr)
        events = kept
    if getattr(args, "infer_sides", False):
        before = sum(1 for e in events if e.side is not None)
        events = infer_sides(events, rule=SideRule(args.side_rule))
        after = sum(1 for e in events if e.side is not None)
        print(f"infer-sides: classified {after - before} trade(s) "
              f"using the {args.side_rule} rule", file=sys.stderr)
    actions = _load_actions(args)
    if actions:
        events = adjust_events(
            events, actions, method=AdjustMethod(args.adjust)
        )
        print(f"adjust: applied {len(actions)} corporate action(s) to events",
              file=sys.stderr)
    return events


def _load_actions(args: argparse.Namespace) -> list:
    """Read --actions, if given. Returns an empty list when absent."""
    path = getattr(args, "actions", None)
    if not path:
        return []
    return read_actions_csv(path, ts_unit=args.ts_unit)


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
    p.add_argument(
        "--session", metavar="HH:MM-HH:MM", default=None,
        help="keep only trades inside this local trading window "
             "(e.g. 09:30-16:00); a window that ends before it starts "
             "is treated as overnight",
    )
    p.add_argument(
        "--tz", default="UTC", metavar="ZONE",
        help="timezone for --session, e.g. America/New_York (default: UTC)",
    )
    p.add_argument(
        "--actions", metavar="FILE", default=None,
        help="CSV of splits, dividends and contract rolls to back-adjust for "
             "(columns: timestamp,kind,value[,ref_price])",
    )
    p.add_argument(
        "--adjust", choices=["ratio", "difference"], default="ratio",
        help="back-adjustment convention for --actions (default: ratio)",
    )
    p.add_argument(
        "--infer-sides", action="store_true",
        help="classify the aggressor side of trades that do not report one",
    )
    p.add_argument(
        "--side-rule", choices=["tick", "quote", "lee_ready"], default="lee_ready",
        help="classification rule for --infer-sides (default: lee_ready)",
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
    if args.interval is not None:
        bars = time_bars(events, args.interval)
        if args.fill_gaps:
            bars = fill_gaps(bars)
    elif args.every_trades is not None:
        bars = count_bars(events, args.every_trades)
    elif args.every_volume is not None:
        bars = volume_bars(events, Decimal(args.every_volume))
    elif args.every_notional is not None:
        bars = dollar_bars(events, Decimal(args.every_notional))
    else:
        bars = imbalance_bars(events, Decimal(args.every_imbalance),
                              by=args.imbalance_by)
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


def _cmd_book(args: argparse.Namespace) -> int:
    import csv as _csv

    book = OrderBook(
        args.symbol, args.venue,
        max_depth=args.max_depth,
        strict_sequence=not args.ignore_sequence,
    )

    def deltas():
        with open_text(args.input) as f:
            for lineno, row in enumerate(_csv.DictReader(f), start=2):
                if not any((v or "").strip() for v in row.values()):
                    continue
                try:
                    raw_ts = (row.get("ts") or row.get("timestamp") or "").strip()
                    ts_ns = (epoch_to_ns(float(raw_ts), args.ts_unit)
                             if args.ts_unit else iso_to_ns(raw_ts))
                    name = (row.get("side") or "").strip().lower()
                    if name in ("buy", "b", "bid"):
                        side = Side.BUY
                    elif name in ("sell", "s", "ask"):
                        side = Side.SELL
                    else:
                        raise ValueError(f"unknown side {name!r}")
                    seq = (row.get("seq") or "").strip()
                    yield BookDelta(
                        ts_ns=ts_ns, side=side,
                        price=Decimal(str(row["price"]).strip()),
                        size=Decimal(str(row["size"]).strip()),
                        seq=int(seq) if seq else None,
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{args.input}:{lineno}: {exc}") from exc

    quotes = list(replay_book(book, deltas(),
                              top_of_book_only=not args.every_update))
    n = _write(quotes, args.output, as_float=args.as_float)
    print(f"wrote {n} quote(s) -> {args.output}")
    if book.is_crossed:
        print("warning: the final book is crossed (best bid >= best ask)",
              file=sys.stderr)
    return 0


def _cmd_nbbo(args: argparse.Namespace) -> int:
    events = read_jsonl_events(args.input) if _is_jsonl(args.input) else []
    if not _is_jsonl(args.input):
        print("error: nbbo needs an NDJSON quote file (.jsonl/.ndjson)",
              file=sys.stderr)
        return 1
    quotes = sorted((e for e in events if e.event_type.value == "quote"),
                    key=lambda e: e.ts_ns)
    if not quotes:
        print("error: no quote events in the input", file=sys.stderr)
        return 1

    symbol = args.symbol or quotes[0].symbol
    book = Consolidator(symbol, max_age_ns=args.max_age)
    out = [top for q in quotes if (top := book.update(q)) is not None]
    n = _write(out, args.output, as_float=args.as_float)
    print(f"wrote {n} consolidated quote(s) -> {args.output}")

    if book.leadership:
        print("venue leadership (updates at the top):", file=sys.stderr)
        for venue in sorted(book.leadership):
            c = book.leadership[venue]
            print(f"  {venue}: bid {c['bid']}, ask {c['ask']}", file=sys.stderr)
    if book.crossed_updates:
        print(f"warning: {book.crossed_updates} update(s) produced a crossed "
              f"consolidated book — check the venue clocks", file=sys.stderr)
    stale = book.stale_venues()
    if stale:
        print(f"warning: stale at the end: {', '.join(stale)}", file=sys.stderr)
    return 0


def _cmd_tca(args: argparse.Namespace) -> int:
    import csv as _csv

    fills = []
    with open_text(args.input) as fh:
        for lineno, row in enumerate(_csv.DictReader(fh), start=2):
            if not any((v or "").strip() for v in row.values()):
                continue
            try:
                raw_ts = (row.get("ts") or row.get("timestamp") or "").strip()
                ts_ns = (epoch_to_ns(float(raw_ts), args.ts_unit)
                         if args.ts_unit else iso_to_ns(raw_ts))
                name = (row.get("side") or "").strip().lower()
                side = Side.BUY if name in ("buy", "b", "bid") else Side.SELL
                fills.append(Fill(ts_ns=ts_ns,
                                  price=Decimal(str(row["price"]).strip()),
                                  size=Decimal(str(row["size"]).strip()),
                                  side=side))
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{args.input}:{lineno}: {exc}") from exc
    if not fills:
        print("error: no fills in the input", file=sys.stderr)
        return 1

    market = read_jsonl_events(args.market) if _is_jsonl(args.market) else \
        read_csv_trades(args.market, venue="market", ts_unit=args.ts_unit)

    r = evaluate(
        fills, market,
        decision_price=Decimal(args.decision_price) if args.decision_price else None,
        twap_interval_ns=args.twap,
        exclude_own=not args.keep_own,
        tolerance_ns=args.tolerance,
    )
    assert r is not None

    def fmt(v, unit=""):
        return "n/a" if v is None else f"{v:.4f}{unit}".rstrip("0").rstrip(".") + unit * 0

    print(f"side                {r.side.value}")
    print(f"filled size         {r.filled_size}")
    print(f"average price       {r.average_price}")
    print(f"market VWAP         {fmt(r.vwap)}")
    if r.twap is not None:
        print(f"market TWAP         {fmt(r.twap)}")
    print(f"vs VWAP             {fmt(r.slippage_vs_vwap_bps)} bps"
          f"   (positive = better)")
    if r.shortfall_bps is not None:
        print(f"vs decision price   {fmt(r.shortfall_bps)} bps")
    if r.participation_rate is not None:
        print(f"participation       {r.participation_rate * 100:.2f}%")
    print(f"own prints removed  {r.own_prints_removed}")
    if r.participation_rate is not None and r.participation_rate > Decimal("0.1"):
        print("note: above 10% participation a VWAP score largely measures "
              "your own impact", file=sys.stderr)
    return 0


def _cmd_align(args: argparse.Namespace) -> int:
    import csv as _csv

    columns = []
    for spec in args.inputs:
        if "=" not in spec:
            print(f"error: expected NAME=path, got {spec!r}", file=sys.stderr)
            return 1
        name, path = spec.split("=", 1)
        if not name or not path:
            print(f"error: expected NAME=path, got {spec!r}", file=sys.stderr)
            return 1
        columns.append((name, path))
    names = [n for n, _ in columns]
    if len(set(names)) != len(names):
        print("error: duplicate column names", file=sys.stderr)
        return 1

    streams = {}
    for name, path in columns:
        if _is_jsonl(path):
            events = read_jsonl_events(path)
        else:
            events = read_csv_trades(path, venue=name, ts_unit=args.ts_unit)
        streams[name] = events

    rows = align(
        streams,
        interval_ns=args.interval,
        field=Field(args.field),
        max_age_ns=args.max_age,
        require_all=args.require_all,
    )
    if not rows:
        print("error: nothing to align (no stream carries the requested field)",
              file=sys.stderr)
        return 1

    header = ["ts_ns"] + names + [f"{n}_age_ns" for n in names]
    with open_text(args.output, "w") as fh:
        w = _csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(
                [r.ts_ns]
                + ["" if r.values[n] is None else str(r.values[n]) for n in names]
                + ["" if r.ages_ns[n] is None else r.ages_ns[n] for n in names]
            )

    complete = sum(1 for r in rows if r.complete)
    print(f"wrote {len(rows)} row(s) to {args.output}", file=sys.stderr)
    print(f"complete rows: {complete}/{len(rows)}", file=sys.stderr)
    stale = sum(1 for r in rows if r.stale)
    if stale:
        print(f"note: {stale} row(s) dropped a column for being older than "
              f"the staleness window", file=sys.stderr)
    if args.max_age is None:
        print("note: no --max-age given, so a stream that stops contributes "
              "its last price to every later row", file=sys.stderr)
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    import csv as _csv

    with open_text(args.input) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        print("error: empty input", file=sys.stderr)
        return 1
    header = list(rows[0])
    if "ts_ns" not in header:
        print("error: input needs a ts_ns column (use `mdnorm align` to make one)",
              file=sys.stderr)
        return 1

    wanted = args.columns or [
        c for c in header if c != "ts_ns" and not c.endswith("_age_ns")
    ]
    missing = [c for c in wanted if c not in header]
    if missing:
        print(f"error: no such column(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    def series(name):
        out = []
        for r in rows:
            raw = (r.get(name) or "").strip()
            out.append(Decimal(raw) if raw else None)
        return out

    data = {c: series(c) for c in wanted}
    out_cols = {}
    for c in wanted:
        out_cols[c] = data[c]
        r = returns(data[c], method=ReturnMethod(args.returns))
        out_cols[f"{c}_ret"] = r
        if args.zscore:
            out_cols[f"{c}_z{args.zscore}"] = rolling_zscore(data[c], args.zscore)
        if args.vol:
            ppy = None
            if args.sessions_per_year and args.session_length and args.interval:
                ppy = periods_per_year(
                    args.interval,
                    sessions_per_year=args.sessions_per_year,
                    session_length_ns=args.session_length,
                )
            out_cols[f"{c}_vol{args.vol}"] = realized_volatility(
                r, window=args.vol, periods_per_year=ppy
            )

    if args.correlate:
        if len(wanted) < 2:
            print("error: --correlate needs at least two columns", file=sys.stderr)
            return 1
        a, b = wanted[0], wanted[1]
        out_cols[f"corr_{a}_{b}_{args.correlate}"] = rolling_correlation(
            data[a], data[b], args.correlate
        )

    names = list(out_cols)

    def fmt(v):
        """Trim the working precision down to something readable.

        The library computes at 34 digits so rounding never reaches a figure
        anyone reports; writing all 34 into a CSV is noise, not accuracy.
        """
        if v is None:
            return ""
        with localcontext() as ctx:
            ctx.prec = args.precision
            return str(+v)

    with open_text(args.output, "w") as fh:
        w = _csv.writer(fh)
        w.writerow(["ts_ns"] + names)
        for i, r in enumerate(rows):
            w.writerow([r["ts_ns"]] + [fmt(out_cols[n][i]) for n in names])

    print(f"wrote {len(rows)} row(s), {len(names)} column(s) to {args.output}",
          file=sys.stderr)
    if args.vol and not (args.sessions_per_year and args.session_length
                         and args.interval):
        print("note: volatility is per period. To annualise, pass --interval, "
              "--sessions-per-year and --session-length together; there is no "
              "safe default.", file=sys.stderr)
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
    how = p_bars.add_mutually_exclusive_group(required=True)
    how.add_argument("--interval", type=parse_interval,
                     help="time-bar interval, e.g. 30s, 1m, 4h, 1d")
    how.add_argument("--every-trades", type=int, metavar="N",
                     help="tick bars: one bar per N trades")
    how.add_argument("--every-volume", metavar="V",
                     help="volume bars: close a bar at >= V base units")
    how.add_argument("--every-notional", metavar="X",
                     help="dollar bars: close a bar at >= X traded value")
    how.add_argument("--every-imbalance", metavar="I",
                     help="imbalance bars: close a bar when signed order flow "
                          "reaches +/- I (needs a side; see --infer-sides)")
    p_bars.add_argument("--imbalance-by", choices=["volume", "tick"],
                        default="volume",
                        help="measure imbalance in traded size or trade count "
                             "(default: volume)")
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

    p_b = sub.add_parser("book",
                         help="rebuild an order book from deltas and emit quotes")
    p_b.add_argument("input", help="CSV of book deltas (ts,side,price,size[,seq])")
    p_b.add_argument("-o", "--output", required=True,
                     help="output file (.csv, .jsonl, .ndjson; .gz accepted)")
    p_b.add_argument("--symbol", required=True, help="symbol the book belongs to")
    p_b.add_argument("--venue", default="csv", help="venue label (default: csv)")
    p_b.add_argument("--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
                     help="parse timestamps as epoch in this unit (default: ISO-8601)")
    p_b.add_argument("--max-depth", type=int, default=None, metavar="N",
                     help="keep only the best N levels per side")
    p_b.add_argument("--ignore-sequence", action="store_true",
                     help="do not stop on a sequence gap (the book may be wrong)")
    p_b.add_argument("--every-update", action="store_true",
                     help="emit a quote per delta, not only when the top changes")
    p_b.add_argument("--as-float", action="store_true",
                     help="write numeric values instead of strings")
    p_b.set_defaults(func=_cmd_book)

    p_n = sub.add_parser("nbbo",
                         help="consolidate multi-venue quotes into a best bid and offer")
    p_n.add_argument("input", help="NDJSON quote events from several venues")
    p_n.add_argument("-o", "--output", required=True,
                     help="output file (.csv, .jsonl, .ndjson; .gz accepted)")
    p_n.add_argument("--symbol", default=None,
                     help="symbol to consolidate (default: the first one seen)")
    p_n.add_argument("--max-age", type=parse_interval, default=None,
                     metavar="DURATION",
                     help="drop a venue that has not quoted for this long, "
                          "e.g. 2s — strongly recommended on live data")
    p_n.add_argument("--as-float", action="store_true",
                     help="write numeric values instead of strings")
    p_n.set_defaults(func=_cmd_nbbo)

    p_t = sub.add_parser("tca",
                         help="score your fills against the market they traded in")
    p_t.add_argument("input", help="CSV of your fills (ts,side,price,size)")
    p_t.add_argument("--market", required=True,
                     help="market tape: NDJSON events or a trades CSV")
    p_t.add_argument("--decision-price", default=None, metavar="P",
                     help="price when the decision was made (implementation shortfall)")
    p_t.add_argument("--twap", type=parse_interval, default=None, metavar="INTERVAL",
                     help="also compute TWAP at this sampling interval, e.g. 1m")
    p_t.add_argument("--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
                     help="parse timestamps as epoch in this unit (default: ISO-8601)")
    p_t.add_argument("--tolerance", type=int, default=0, metavar="NS",
                     help="timestamp tolerance when matching your prints on the tape")
    p_t.add_argument("--keep-own", action="store_true",
                     help="do NOT remove your own prints from the benchmark "
                          "(flatters the score; off by default for a reason)")
    p_t.set_defaults(func=_cmd_tca)

    p_a = sub.add_parser("align",
                         help="join several instruments onto one time grid "
                              "(as-of, never forward-looking)")
    p_a.add_argument("inputs", nargs="+", metavar="NAME=PATH",
                     help="one column per instrument, e.g. BTC=btc.csv ETH=eth.jsonl")
    p_a.add_argument("-o", "--output", required=True, help="output CSV")
    p_a.add_argument("--interval", type=parse_interval, required=True,
                     help="grid interval, e.g. 1s, 1m, 1h")
    p_a.add_argument("--field", choices=[f.value for f in Field],
                     default=Field.PRICE.value,
                     help="which value to take (default: price)")
    p_a.add_argument("--max-age", type=parse_interval, default=None,
                     metavar="DURATION",
                     help="blank a column whose newest value is older than "
                          "this, e.g. 5m — recommended, since forward-filling "
                          "otherwise never expires")
    p_a.add_argument("--require-all", action="store_true",
                     help="drop rows where any column is missing")
    p_a.add_argument("--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
                     help="parse CSV timestamps as epoch in this unit "
                          "(default: ISO-8601)")
    p_a.set_defaults(func=_cmd_align)

    p_f = sub.add_parser("features",
                         help="returns, rolling z-scores, volatility and "
                              "correlation on an aligned matrix (never "
                              "full-sample, never a partial window)")
    p_f.add_argument("input", help="CSV with a ts_ns column, e.g. from `mdnorm align`")
    p_f.add_argument("-o", "--output", required=True, help="output CSV")
    p_f.add_argument("--columns", nargs="+", default=None, metavar="NAME",
                     help="columns to use (default: every non-age column)")
    p_f.add_argument("--returns", choices=[m.value for m in ReturnMethod],
                     default=ReturnMethod.SIMPLE.value,
                     help="return convention (default: simple)")
    p_f.add_argument("--zscore", type=int, default=None, metavar="N",
                     help="add a trailing z-score over N observations")
    p_f.add_argument("--vol", type=int, default=None, metavar="N",
                     help="add realized volatility of returns over N observations")
    p_f.add_argument("--precision", type=int, default=12, metavar="N",
                     help="significant digits in the output (default: 12)")
    p_f.add_argument("--correlate", type=int, default=None, metavar="N",
                     help="add the trailing correlation of the first two columns")
    p_f.add_argument("--interval", type=parse_interval, default=None,
                     help="grid interval of the input, for annualising volatility")
    p_f.add_argument("--sessions-per-year", type=int, default=None, metavar="N",
                     help="sessions in a year, e.g. 252 for cash equities, 365 "
                          "for a continuous venue")
    p_f.add_argument("--session-length", type=parse_interval, default=None,
                     metavar="DURATION",
                     help="length of one session, e.g. 6h for equities, 24h "
                          "for a continuous venue")
    p_f.set_defaults(func=_cmd_features)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
