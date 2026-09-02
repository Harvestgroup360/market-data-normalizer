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
    mdnorm labels feats.csv --column BTC --horizon 5 --splits 5 --embargo 60 -o ml.csv
    mdnorm universe matrix.csv --listings listings.csv --pct-rank -o pit.csv
    mdnorm revisions gdp.csv -o published.csv
    mdnorm metrics pnl.csv --column ret --trials 200 --trial-variance 0.02
    mdnorm costs pnl.csv --cost-bps 5 --edge-bps 20 --adv 1e6 --volatility 0.02
    mdnorm instruments symbol_map.csv trades.csv -o keyed.csv
    mdnorm calendar us_2026.csv --session 09:30-16:00 --tz America/New_York
    mdnorm fx prices.csv rates.csv --from EUR --to USD --max-age 1m -o usd.csv
    mdnorm ticks prices.csv --table ticks.csv
    mdnorm arrival feed.csv --interval 1s
    mdnorm seasonality volume.csv --session 09:30-16:00 --bucket 5m

Input format is inferred from the extension: ``.jsonl`` / ``.ndjson`` files
are read as NDJSON (already-normalized events), anything else as a trades
CSV. Output format likewise: ``.jsonl`` / ``.ndjson`` writes NDJSON,
anything else CSV.
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal, localcontext
from typing import List, Optional, cast

from . import __version__
from .adjust import AdjustMethod, adjust_events, read_actions_csv
from .align import AsOfSeries, Field, align, grid
from .features import periods_per_year
from .labels import forward_returns, purged_splits
from .metrics import (drawdowns, equity_curve, hit_rate, max_drawdown,
                      profit_factor, sharpe_report, sortino_ratio)
from .costs import (CostModel, Fees, ImpactModel, Liquidity, apply_costs,
                    breakeven_participation, capacity, cost_report, estimate)
from .instruments import (SymbolMap, key_by_instrument,
                          read_symbol_map_csv, series_segments)
from .arrival import (as_received, as_stamped, delay_report,
                      read_arrivals_csv, view_gap)
from .calendars import read_calendar_csv
from .seasonality import (deseasonalise, full_sample_deseasonalise,
                          profile_leak, read_samples_csv,
                          session_profile)
from .fx import convert_series, read_fx_csv
from .ticksize import Rounding, grid_report, read_tick_table_csv
from .membership import Basis, read_index_changes_csv, survivorship_gap
from .reconcile import reconcile, suggest_shift
from .mixfreq import leak_report, read_periods_csv
from .revisions import RevisionSeries, read_revisions_csv
from .universe import (Universe, cross_section, cross_sectional_rank,
                       cross_sectional_zscore, mask_to_universe,
                       read_listings_csv)
from .align import AlignedRow
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


def _periods_per_year(args: argparse.Namespace) -> Optional[Decimal]:
    """Resolve the annualisation factor, or None, warning about the usual trap.

    An interval longer than one session means fewer than one bar per session,
    which is almost always a daily series described with an intraday session
    length. It halves or quarters every annualised figure and looks entirely
    plausible, so it is called out rather than silently applied.
    """
    if not (args.sessions_per_year and args.session_length and args.interval):
        return None
    ppy = periods_per_year(
        args.interval,
        sessions_per_year=args.sessions_per_year,
        session_length_ns=args.session_length,
    )
    if args.interval > args.session_length:
        print(f"warning: an interval longer than one session gives fewer than "
              f"one bar per session ({ppy} periods a year, against "
              f"{args.sessions_per_year} sessions). For daily bars set "
              f"--session-length equal to --interval.", file=sys.stderr)
    return ppy


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
            ppy = _periods_per_year(args)
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
        for i, row in enumerate(rows):
            w.writerow([row["ts_ns"]]
                       + [fmt(out_cols[n][i]) for n in names])

    print(f"wrote {len(rows)} row(s), {len(names)} column(s) to {args.output}",
          file=sys.stderr)
    if args.vol and not (args.sessions_per_year and args.session_length
                         and args.interval):
        print("note: volatility is per period. To annualise, pass --interval, "
              "--sessions-per-year and --session-length together; there is no "
              "safe default.", file=sys.stderr)
    return 0


def _cmd_labels(args: argparse.Namespace) -> int:
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
    if args.column not in header:
        print(f"error: no column {args.column!r}; have: "
              f"{', '.join(c for c in header if c != 'ts_ns')}", file=sys.stderr)
        return 1

    series = []
    for r in rows:
        raw = (r.get(args.column) or "").strip()
        series.append(Decimal(raw) if raw else None)

    y = forward_returns(series, horizon=args.horizon,
                        method=ReturnMethod(args.returns))
    name = f"{args.column}_fwd{args.horizon}"

    with open_text(args.output, "w") as fh:
        w = _csv.writer(fh)
        w.writerow(header + [name])
        for i, row in enumerate(rows):
            value = ""
            label = y[i]
            if label is not None:
                with localcontext() as ctx:
                    ctx.prec = args.precision
                    value = str(+label)
            w.writerow([row.get(c, "") for c in header] + [value])

    labelled = sum(1 for v in y if v is not None)
    print(f"wrote {len(rows)} row(s) to {args.output}", file=sys.stderr)
    print(f"labelled rows: {labelled}/{len(rows)} "
          f"(the last {args.horizon} have no outcome yet)", file=sys.stderr)

    if args.splits:
        splits = purged_splits(len(rows), n_splits=args.splits,
                               horizon=args.horizon, embargo=args.embargo)
        print(f"\n{args.splits} purged folds "
              f"(horizon {args.horizon}, embargo {args.embargo}):", file=sys.stderr)
        for k, sp in enumerate(splits):
            print(f"  fold {k}: train {len(sp.train):>6}  test {len(sp.test):>6}  "
                  f"purged {sp.purged:>4}  embargoed {sp.embargoed:>4}",
                  file=sys.stderr)
        total = sum(sp.discarded for sp in splits)
        print(f"  {total} training row(s) discarded across all folds to keep "
              f"the test blocks clean", file=sys.stderr)
        if args.embargo == 0:
            print("note: embargo is 0. Set it to at least your longest feature "
                  "window, or rows just after a test block will carry it.",
                  file=sys.stderr)
    return 0


def _cmd_universe(args: argparse.Namespace) -> int:
    import csv as _csv

    with open_text(args.input) as fh:
        raw = list(_csv.DictReader(fh))
    if not raw:
        print("error: empty input", file=sys.stderr)
        return 1
    header = list(raw[0])
    if "ts_ns" not in header:
        print("error: input needs a ts_ns column (use `mdnorm align` to make one)",
              file=sys.stderr)
        return 1

    columns = args.columns or [
        c for c in header if c != "ts_ns" and not c.endswith("_age_ns")
    ]
    missing = [c for c in columns if c not in header]
    if missing:
        print(f"error: no such column(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    rows = []
    for r in raw:
        values = {}
        ages = {}
        for c in columns:
            cell = (r.get(c) or "").strip()
            values[c] = Decimal(cell) if cell else None
            age = (r.get(f"{c}_age_ns") or "").strip()
            ages[c] = int(age) if age else None
        rows.append(AlignedRow(ts_ns=int(r["ts_ns"]), values=values, ages_ns=ages))

    universe = Universe(read_listings_csv(args.listings, ts_unit=args.ts_unit))
    masked, removed = mask_to_universe(rows, universe)

    ops = []
    if args.rank:
        ops.append(("rank", lambda v: cross_sectional_rank(v, pct=False)))
    if args.pct_rank:
        ops.append(("pct", lambda v: cross_sectional_rank(v, pct=True)))
    if args.zscore:
        ops.append(("xz", cross_sectional_zscore))

    extra = {}
    for suffix, fn in ops:
        out_rows = cross_section(masked, fn)
        for c in columns:
            extra[f"{c}_{suffix}"] = [r.values[c] for r in out_rows]

    names = list(columns) + list(extra)

    def fmt(v):
        if v is None:
            return ""
        with localcontext() as ctx:
            ctx.prec = args.precision
            return str(+v)

    with open_text(args.output, "w") as fh:
        w = _csv.writer(fh)
        w.writerow(["ts_ns", "members"] + names)
        for i, aligned in enumerate(masked):
            members = sum(1 for c in columns
                          if aligned.values[c] is not None)
            w.writerow([aligned.ts_ns, members]
                       + [fmt(aligned.values[c]) for c in columns]
                       + [fmt(extra[n][i]) for n in extra])

    sizes = [sum(1 for c in columns if r.values[c] is not None) for r in masked]
    print(f"wrote {len(masked)} row(s) to {args.output}", file=sys.stderr)
    print(f"masked {removed} cell(s) outside their listing window", file=sys.stderr)
    if sizes:
        print(f"cross-section size: min {min(sizes)}, max {max(sizes)}",
              file=sys.stderr)
    if removed == 0:
        print("note: nothing was masked. Over a long window that usually means "
              "the listings file is present-day membership rather than a "
              "historical record — the definition of survivorship bias.",
              file=sys.stderr)
    return 0


def _cmd_revisions(args: argparse.Namespace) -> int:
    import csv as _csv

    series = RevisionSeries(read_revisions_csv(args.input, ts_unit=args.ts_unit))
    if len(series) == 0:
        print("error: no revisions in the input", file=sys.stderr)
        return 1

    summary = series.revision_summary()

    def fmt(v):
        if v is None:
            return "n/a"
        with localcontext() as ctx:
            ctx.prec = args.precision
            return str(+v)

    print(f"events              {summary.events}", file=sys.stderr)
    print(f"revised at least once {summary.revised_events}"
          f"  ({fmt(summary.revised_fraction)})", file=sys.stderr)
    print(f"mean |final - first| {fmt(summary.mean_absolute_change)}",
          file=sys.stderr)
    print(f"max  |final - first| {fmt(summary.max_absolute_change)}",
          file=sys.stderr)
    if summary.revised_events:
        print("note: a study built on final values reads corrections that were "
              "not available at the time. The numbers above are how large that "
              "advantage is.", file=sys.stderr)

    if args.vintage is not None:
        vintage = series.vintage_at(args.vintage)
        with open_text(args.output, "w") as fh:
            w = _csv.writer(fh)
            w.writerow(["event_ts_ns", "value"])
            for event in series.events:
                value = series.as_of(event_ts_ns=event, known_ts_ns=args.vintage)
                if value is not None:
                    w.writerow([event, fmt(value)])
        print(f"\nwrote the vintage as of {args.vintage} "
              f"({len(vintage)} event(s)) to {args.output}", file=sys.stderr)
        return 0

    with open_text(args.output, "w") as fh:
        w = _csv.writer(fh)
        w.writerow(["known_ts_ns", "value"])
        for event in series.events:
            for k in series.published_at(event):
                w.writerow([k, fmt(series.as_of(event_ts_ns=event,
                                                known_ts_ns=k))])
    print(f"\nwrote the publication stream to {args.output} — keyed by when "
          f"each version became readable, so it joins like any other series",
          file=sys.stderr)
    return 0


def _cmd_metrics(args: argparse.Namespace) -> int:
    import csv as _csv

    with open_text(args.input) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        print("error: empty input", file=sys.stderr)
        return 1
    if args.column not in rows[0]:
        print(f"error: no such column: {args.column}", file=sys.stderr)
        return 1

    raw: List[Optional[Decimal]] = []
    for r in rows:
        cell = (r.get(args.column) or "").strip()
        raw.append(Decimal(cell) if cell else None)

    if args.prices:
        series = returns(raw, method=ReturnMethod(args.returns))
    else:
        series = raw

    ppy = _periods_per_year(args)

    if (args.trials is None) != (args.trial_variance is None):
        print("error: --trials and --trial-variance must be given together. "
              "The deflated figure is the one that changes the conclusion; "
              "half of it is not an answer.", file=sys.stderr)
        return 1

    rep = sharpe_report(
        series,
        risk_free=Decimal(args.risk_free),
        periods_per_year=ppy,
        confidence=Decimal(args.confidence),
        trials=args.trials,
        trial_sharpe_variance=(None if args.trial_variance is None
                               else Decimal(args.trial_variance)),
    )

    def fmt(v):
        if v is None:
            return "n/a"
        with localcontext() as ctx:
            ctx.prec = args.precision
            return str(+v)

    curve = equity_curve(series)
    worst = max_drawdown(curve)
    closed = [d for d in drawdowns(curve) if d.recovered]

    print(f"observations         {rep.observations}", file=sys.stderr)
    if rep.skipped:
        print(f"missing              {rep.skipped}", file=sys.stderr)
    print(f"Sharpe (per period)  {fmt(rep.sharpe)}", file=sys.stderr)
    print(f"Sharpe (annualised)  {fmt(rep.sharpe_annualised)}", file=sys.stderr)
    print(f"Sortino (per period) {fmt(sortino_ratio(series))}", file=sys.stderr)
    print(f"skewness             {fmt(rep.skewness)}", file=sys.stderr)
    print(f"kurtosis (non-excess) {fmt(rep.kurtosis)}", file=sys.stderr)
    print(f"hit rate             {fmt(hit_rate(series))}", file=sys.stderr)
    print(f"profit factor        {fmt(profit_factor(series))}", file=sys.stderr)
    if worst is None:
        print("max drawdown         none in this sample", file=sys.stderr)
    else:
        print(f"max drawdown         {fmt(worst.depth)}"
              f"  (peak {worst.peak_index} -> trough {worst.trough_index}"
              f"{'' if worst.recovered else ', not recovered'})",
              file=sys.stderr)
        print(f"drawdowns recovered  {len(closed)} of {len(drawdowns(curve))}",
              file=sys.stderr)
    print(f"P(Sharpe > 0)        {fmt(rep.probabilistic)}", file=sys.stderr)
    print(f"min track record     {fmt(rep.min_track_record)} period(s) "
          f"at {args.confidence} confidence", file=sys.stderr)
    if rep.trials is not None:
        print(f"deflated ({rep.trials} trials) {fmt(rep.deflated)}",
              file=sys.stderr)

    for warning in rep.warnings:
        print(f"note: {warning}", file=sys.stderr)

    if args.output:
        with open_text(args.output, "w") as fh:
            w = _csv.writer(fh)
            w.writerow(["metric", "value"])
            w.writerow(["observations", rep.observations])
            w.writerow(["missing", rep.skipped])
            w.writerow(["sharpe_per_period", fmt(rep.sharpe)])
            w.writerow(["sharpe_annualised", fmt(rep.sharpe_annualised)])
            w.writerow(["sortino_per_period", fmt(sortino_ratio(series))])
            w.writerow(["skewness", fmt(rep.skewness)])
            w.writerow(["kurtosis", fmt(rep.kurtosis)])
            w.writerow(["hit_rate", fmt(hit_rate(series))])
            w.writerow(["profit_factor", fmt(profit_factor(series))])
            w.writerow(["max_drawdown", "n/a" if worst is None
                        else fmt(worst.depth)])
            w.writerow(["probabilistic_sharpe", fmt(rep.probabilistic)])
            w.writerow(["min_track_record_periods", fmt(rep.min_track_record)])
            w.writerow(["deflated_sharpe", fmt(rep.deflated)])
            w.writerow(["trials", "n/a" if rep.trials is None else rep.trials])
        print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


def _cmd_costs(args: argparse.Namespace) -> int:
    import csv as _csv

    fees = Fees(taker_bps=Decimal(args.fee_bps),
                per_unit=Decimal(args.fee_per_unit),
                minimum=Decimal(args.fee_minimum))
    impact = (None if args.impact_coefficient is None
              else ImpactModel(coefficient=Decimal(args.impact_coefficient),
                               exponent=Decimal(args.impact_exponent)))
    model = CostModel(fees=fees, impact=impact,
                      spread_fraction=Decimal(args.spread_fraction))

    liquidity = None
    if (args.adv is not None or args.volatility is not None
            or Decimal(args.spread_bps) != 0):
        if args.adv is None or args.volatility is None:
            print("error: --adv and --volatility must be given together to "
                  "describe liquidity", file=sys.stderr)
            return 1
        liquidity = Liquidity(adv=Decimal(args.adv),
                              volatility=Decimal(args.volatility),
                              spread_bps=Decimal(args.spread_bps))
    if impact is not None and liquidity is None:
        print("error: an impact model needs --adv and --volatility. Without "
              "them the cost cannot depend on trade size.", file=sys.stderr)
        return 1

    def fmt(v):
        if v is None:
            return "n/a"
        with localcontext() as ctx:
            ctx.prec = args.precision
            return str(+Decimal(v))

    # -- what one trade costs -------------------------------------------------
    cost_bps = None
    if args.notional is not None and args.quantity is not None:
        b = estimate(model, notional=Decimal(args.notional),
                     quantity=Decimal(args.quantity), liquidity=liquidity)
        cost_bps = b.total_bps
        print(f"notional             {fmt(b.notional)}", file=sys.stderr)
        print(f"participation        {fmt(b.participation)}", file=sys.stderr)
        print(f"  commission         {fmt(b.commission_bps)} bps", file=sys.stderr)
        print(f"  spread             {fmt(b.spread_bps)} bps", file=sys.stderr)
        print(f"  impact             {fmt(b.impact_bps)} bps", file=sys.stderr)
        print(f"  total              {fmt(b.total_bps)} bps "
              f"= {fmt(b.total)}", file=sys.stderr)
        for warning in b.warnings:
            print(f"note: {warning}", file=sys.stderr)
    elif args.cost_bps is not None:
        cost_bps = Decimal(args.cost_bps)
    elif args.input:
        print("error: give either --cost-bps, or --notional with --quantity so "
              "the cost can be priced from the model", file=sys.stderr)
        return 1

    # -- what it does to a return series -------------------------------------
    if args.input:
        with open_text(args.input) as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            print("error: empty input", file=sys.stderr)
            return 1
        for col in (args.column, args.turnover_column):
            if col not in rows[0]:
                print(f"error: no such column: {col}", file=sys.stderr)
                return 1

        def series(name):
            out = []
            for r in rows:
                cell = (r.get(name) or "").strip()
                out.append(Decimal(cell) if cell else None)
            return out

        gross = series(args.column)
        turn = series(args.turnover_column)
        # Either branch above set it, or we returned; state that for the
        # reader as well as the checker.
        priced = cast(Decimal, cost_bps)
        rep = cost_report(gross, turn, cost_bps=priced)
        print(f"\nperiods              {rep.periods}", file=sys.stderr)
        print(f"average turnover     {fmt(rep.average_turnover)}", file=sys.stderr)
        print(f"gross return         {fmt(rep.gross_return)}", file=sys.stderr)
        print(f"net return           {fmt(rep.net_return)}", file=sys.stderr)
        print(f"cost                 {fmt(rep.cost)}", file=sys.stderr)
        print(f"share of gross       {fmt(rep.cost_fraction)}", file=sys.stderr)
        for warning in rep.warnings:
            print(f"note: {warning}", file=sys.stderr)

        if args.output:
            net = apply_costs(gross, turn, cost_bps=priced)
            with open_text(args.output, "w") as fh:
                w = _csv.writer(fh)
                head = list(rows[0])
                w.writerow(head + ["net"])
                for r, n in zip(rows, net):
                    w.writerow([r[h] for h in head] + ["" if n is None else fmt(n)])
            print(f"\nwrote {args.output}", file=sys.stderr)

    # -- how big can it get ---------------------------------------------------
    if args.edge_bps is not None:
        if liquidity is None:
            print("error: --edge-bps needs --adv and --volatility",
                  file=sys.stderr)
            return 1
        edge = Decimal(args.edge_bps)
        p = breakeven_participation(edge, model=model, liquidity=liquidity)
        cap = capacity(edge, model=model, liquidity=liquidity)
        print(f"\nedge                 {fmt(edge)} bps", file=sys.stderr)
        if p is None:
            fixed = liquidity.spread_bps * model.spread_fraction + fees.taker_bps
            if impact is None:
                print("breakeven            n/a — no impact model, so the cost "
                      "never grows with size and there is nothing to solve for",
                      file=sys.stderr)
            else:
                print(f"breakeven            none — fixed costs of {fmt(fixed)} "
                      f"bps already exceed the edge, so no trade size works",
                      file=sys.stderr)
        else:
            print(f"breakeven            {fmt(p)} of daily volume",
                  file=sys.stderr)
            print(f"capacity             {fmt(cap)} per day", file=sys.stderr)
    return 0


def _cmd_mixfreq(args: argparse.Namespace) -> int:
    import csv as _csv

    try:
        series = read_periods_csv(
            args.periods,
            start_column=args.start_field,
            end_column=args.end_field,
            value_column=args.value_field,
            publication_lag_ns=args.lag,
            name=args.name,
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(series) == 0:
        print("error: no periods in input", file=sys.stderr)
        return 1

    periods = series.periods
    first, last = periods[0].start_ns, periods[-1].end_ns
    points = grid(first, last, args.interval)
    rep = leak_report(series, points)

    print(f"periods              {len(series)}", file=sys.stderr)
    print(f"grid points          {rep.grid_points}", file=sys.stderr)
    print(f"knowable             {rep.knowable_points}", file=sys.stderr)
    print(f"label join answers   {rep.label_points}", file=sys.stderr)
    print(f"of those, too early  {rep.leaking_points}", file=sys.stderr)
    if rep.max_lead_ns is not None:
        print(f"worst read-ahead     {rep.max_lead_ns / 1e9:,.0f}s",
              file=sys.stderr)
    if rep.leaking_fraction is not None:
        pct = rep.leaking_fraction * 100
        print(f"share leaking        {pct:.1f}%", file=sys.stderr)

    if rep.leaking_points == rep.grid_points and rep.grid_points:
        print("note: every grid point leaks. That is what back-to-back "
              "periods do to a label join — the moment one value becomes "
              "readable the label has already moved to the next one.",
              file=sys.stderr)
    if args.lag == 0:
        print("note: publication lag is zero, which claims each value is "
              "readable the instant its period ends. If a vendor sends it "
              "later, pass --lag; the figures above are optimistic by "
              "exactly that much.", file=sys.stderr)

    if not args.output:
        return 0

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["ts_ns", "knowable", "label"])
        knowable = series.knowable_series()
        labelled = series.labelled_series()
        for t in points:
            k = knowable.at(t)[0]
            l = labelled.at(t)[0]
            w.writerow([t, "" if k is None else str(k),
                        "" if l is None else str(l)])
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_membership(args: argparse.Namespace) -> int:
    try:
        history = read_index_changes_csv(
            args.changes,
            instrument_column=args.instrument_field,
            kind_column=args.action_field,
            effective_column=args.effective_field,
            announced_column=args.announced_field,
            name=args.name,
        )
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(history) == 0:
        print("error: no changes in input", file=sys.stderr)
        return 1

    rep = history.report()
    print(f"instruments          {rep.instruments}", file=sys.stderr)
    print(f"changes              {rep.changes}", file=sys.stderr)
    print(f"  additions          {rep.additions}", file=sys.stderr)
    print(f"  deletions          {rep.deletions}", file=sys.stderr)
    print(f"never removed        {rep.never_removed}", file=sys.stderr)
    print(f"no announcement      {rep.without_announcement}", file=sys.stderr)
    if rep.announcement_coverage is not None:
        print(f"announcement cover   {rep.announcement_coverage * 100:.0f}%",
              file=sys.stderr)
    if rep.max_uncertainty_ns is not None:
        print(f"worst dating window  {rep.max_uncertainty_ns / 86_400e9:,.0f}d",
              file=sys.stderr)

    if rep.deletions == 0 and rep.instruments:
        print("note: nothing ever left this index. Over any real history that "
              "does not happen, so this file is very likely a list of today's "
              "members rather than a record of who was in it when.",
              file=sys.stderr)
    if rep.without_announcement == rep.changes:
        print("note: no change carries an announcement date, so the announced "
              "and effective bases are identical here. Any study of the "
              "announcement effect is measuring the effective date.",
              file=sys.stderr)

    if args.at is not None:
        basis = Basis(args.basis)
        members = history.members_at(args.at, basis=basis)
        print(f"members at {args.at} on the {basis.value} basis: "
              f"{len(members)}", file=sys.stderr)
        missed, phantom = survivorship_gap(history, args.at, basis=basis)
        print(f"a today-list would drop  {len(missed)}", file=sys.stderr)
        print(f"a today-list would add   {len(phantom)}", file=sys.stderr)
        for iid in missed[:args.list_limit]:
            print(f"  dropped: {iid}", file=sys.stderr)
        if len(missed) > args.list_limit:
            print(f"  ... and {len(missed) - args.list_limit} more",
                  file=sys.stderr)
        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as fh:
                fh.write("instrument_id\n")
                for iid in members:
                    fh.write(f"{iid}\n")
            print(f"wrote {args.output}", file=sys.stderr)
    elif args.output:
        print("error: --output needs --at, so there is a moment to write",
              file=sys.stderr)
        return 1
    return 0


def _cmd_ticks(args: argparse.Namespace) -> int:
    import csv as _csv

    table = read_tick_table_csv(args.table)
    prices = []
    with open_text(args.input) as fh:
        for row in _csv.DictReader(fh):
            prices.append(Decimal(row[args.value_field].strip()))
    if not prices:
        print("error: the input has no prices", file=sys.stderr)
        return 1

    rep = grid_report(prices, table)
    print(f"prices               {rep.total}", file=sys.stderr)
    print(f"on the grid          {rep.on_grid}", file=sys.stderr)
    print(f"off the grid         {rep.off_grid}", file=sys.stderr)
    if rep.below_table:
        print(f"below the table      {rep.below_table}", file=sys.stderr)
    if rep.share_on_grid is not None:
        print(f"share on grid        {rep.share_on_grid * 100:.2f}%",
              file=sys.stderr)
    if rep.worst_offset is not None:
        print(f"worst offset         {rep.worst_offset} (e.g. {rep.example})",
              file=sys.stderr)

    if rep.looks_raw is True:
        print("note: every price sat on the grid, which is what raw prints "
              "from one venue look like.", file=sys.stderr)
    elif rep.looks_raw is False:
        print(f"note: {rep.off_grid} price(s) could not have been quoted on "
              f"this grid, so this series is not raw prints. In rough order "
              f"of frequency: a mid or a VWAP rather than a print; a history "
              f"back-adjusted for splits and dividends; several venues with "
              f"different grids in one file; or the wrong table for the "
              f"period.", file=sys.stderr)
    elif rep.below_table:
        print("note: nothing could be judged — every price fell below the "
              "table. Extend the table rather than reading this as a pass.",
              file=sys.stderr)

    if not args.output:
        return 0
    mode = Rounding(args.round)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow([args.value_field, "on_grid", "rounded"])
        for p in prices:
            if p < table.floor:
                w.writerow([p, "", ""])
                continue
            w.writerow([p, int(table.on_grid(p)), table.round(p, mode)])
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_fx(args: argparse.Namespace) -> int:
    import csv as _csv

    rows = []
    with open_text(args.input) as fh:
        for row in _csv.DictReader(fh):
            rows.append((int(row[args.ts_field]),
                         Decimal(row[args.value_field].strip())))
    if not rows:
        print("error: the input has no observations", file=sys.stderr)
        return 1
    prices = AsOfSeries(rows)
    rates = read_fx_csv(args.rates, allow_inverse=not args.no_inverse)

    converted, dropped = convert_series(
        prices, rates, base=args.base, to=args.quote,
        max_age_ns=args.max_age, via=args.via)

    print(f"observations         {len(prices)}", file=sys.stderr)
    print(f"converted            {len(converted)}", file=sys.stderr)
    print(f"no usable rate       {dropped}", file=sys.stderr)
    print(f"rate pairs available {[str(p) for p in rates.pairs]}",
          file=sys.stderr)

    if dropped:
        share = dropped * 100 // len(prices)
        print(f"note: {dropped} observation(s), {share}% of the input, had no "
              f"rate within {args.max_age / 1e9:g}s and were dropped rather than "
              f"converted against an older one. FX stops over weekends while "
              f"other venues do not; widen --max-age deliberately or accept "
              f"the gap, but do not do neither.", file=sys.stderr)
    if len(converted):
        # A non-empty series has both a first and a last timestamp.
        span = (cast(int, prices.first_ts_ns), cast(int, prices.last_ts_ns))
        first = rates.convert(Decimal(1), args.base, args.quote, span[0],
                              max_age_ns=args.max_age, via=args.via)
        last = rates.convert(Decimal(1), args.base, args.quote, span[1],
                             max_age_ns=args.max_age, via=args.via)
        if first is not None and last is not None and first.rate != last.rate:
            move = (last.rate / first.rate - 1) * 100
            print(f"note: the rate moved {move:+.2f}% across this span, so a "
                  f"single-rate conversion would have restated the whole "
                  f"series by a number that did not exist until the end of "
                  f"it.", file=sys.stderr)
        if any(leg.inverted for c in (first, last) if c for leg in c.legs):
            print("note: the rate was used upside-down relative to how it is "
                  "stored; check that the pair in the file means what you "
                  "think it means.", file=sys.stderr)
        if first is not None and first.crossed:
            print(f"note: crossed through {args.via}, so both legs' spreads "
                  f"and both staleness windows apply.", file=sys.stderr)

    if not args.output:
        return 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow([args.ts_field, args.value_field])
        for ts, value in zip(_fx_timestamps(converted), _fx_values(converted)):
            w.writerow([ts, value])
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _fx_timestamps(series):
    return list(series._ts)      # noqa: SLF001


def _fx_values(series):
    return [series.at(t)[0] for t in _fx_timestamps(series)]


def _cmd_calendar(args: argparse.Namespace) -> int:
    from datetime import date as _date

    def as_day(text: Optional[str]) -> Optional[_date]:
        return _date.fromisoformat(text) if text else None

    session = parse_session(args.session, args.tz)
    cal = read_calendar_csv(args.calendar, session,
                            first_day=as_day(args.first_day),
                            last_day=as_day(args.last_day),
                            name=args.calendar)
    first, last = cal.covers
    start = as_day(getattr(args, "from_day", None)) or first
    end = as_day(args.to_day) or last

    rep = cal.report(start, end)
    seconds = cal.trading_seconds_between(start, end)
    minutes = seconds // 60

    print(f"calendar covers      {first}..{last}", file=sys.stderr)
    print(f"reported range       {rep.first_day}..{rep.last_day} "
          f"({rep.calendar_days} calendar days)", file=sys.stderr)
    print(f"trading days         {rep.trading_days}", file=sys.stderr)
    print(f"  holidays           {rep.holidays}", file=sys.stderr)
    print(f"  weekend days       {rep.weekend_days}", file=sys.stderr)
    print(f"early closes         {rep.early_closes}", file=sys.stderr)
    print(f"trading minutes      {minutes}", file=sys.stderr)

    if rep.trading_days:
        full = 0
        for day in cal.trading_days_between(start, end):
            span = cal.session_on(day)
            if span:
                full = max(full, span[1] - span[0])
        lost = full // 1_000_000_000 * rep.trading_days - seconds
        if lost:
            print(f"note: early closes cost {lost // 60} minute(s) against a "
                  f"flat {full // 60_000_000_000}-minute session. An "
                  f"annualisation that multiplies sessions by a fixed length "
                  f"is high by that much.", file=sys.stderr)
        if (rep.first_day.month, rep.first_day.day) == (1, 1) and \
                (rep.last_day.month, rep.last_day.day) == (12, 31) and \
                rep.first_day.year == rep.last_day.year:
            print(f"for `mdnorm features`: --sessions-per-year "
                  f"{rep.trading_days} --session-length "
                  f"{full // 1_000_000_000}s", file=sys.stderr)
            if rep.trading_days != 252:
                import math
                skew = math.sqrt(252 / rep.trading_days) - 1
                direction = "overstates" if skew > 0 else "understates"
                print(f"note: {rep.first_day.year} has {rep.trading_days} "
                      f"sessions here, not the conventional 252. Annualising "
                      f"a volatility on 252 {direction} it by "
                      f"{abs(skew) * 100:.2f}%.", file=sys.stderr)
        else:
            print("note: the range is not a whole year, so it does not give a "
                  "sessions-per-year figure. Report a calendar year to get "
                  "one.", file=sys.stderr)

    if not args.output:
        return 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        fh.write("date,close,early\n")
        for day in cal.trading_days_between(start, end):
            close = cal.close_time(day)
            early = "1" if close != session.end else "0"
            fh.write(f"{day},{close},{early}\n")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _fmt_ns(ns: Optional[int]) -> str:
    """A duration in nanoseconds, in whatever unit reads as a number."""
    if ns is None:
        return "-"
    sign = "-" if ns < 0 else ""
    n = abs(ns)
    if n < 1_000:
        return f"{sign}{n}ns"
    if n < 1_000_000:
        return f"{sign}{n / 1_000:.3g}us"
    if n < 1_000_000_000:
        return f"{sign}{n / 1_000_000:.4g}ms"
    seconds = n // 1_000_000_000
    if seconds < 90:
        return f"{sign}{n / 1_000_000_000:.4g}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{sign}{minutes}m" + (f"{rest}s" if rest else "")
    hours, minutes = divmod(minutes, 60)
    return f"{sign}{hours}h" + (f"{minutes}m" if minutes else "")


def _cmd_arrival(args: argparse.Namespace) -> int:
    try:
        arrivals = read_arrivals_csv(
            args.input, venue_column=args.venue_field,
            received_column=args.received_field,
            value_column=args.value_field)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rep = delay_report(arrivals)
    print(f"observations         {rep.observations}", file=sys.stderr)
    print(f"min delay            {_fmt_ns(rep.min_ns)}", file=sys.stderr)
    print(f"median delay         {_fmt_ns(rep.median_ns)}", file=sys.stderr)
    print(f"p95 delay            {_fmt_ns(rep.p95_ns)}", file=sys.stderr)
    print(f"max delay            {_fmt_ns(rep.max_ns)}", file=sys.stderr)
    if rep.tail_ratio is not None:
        print(f"p95 / median         {rep.tail_ratio:.2f}x", file=sys.stderr)
    print(f"received before sent {rep.negative}", file=sys.stderr)
    print(f"out of order         {rep.out_of_order}", file=sys.stderr)

    if rep.negative:
        share = rep.clock_skew_share
        assert share is not None  # negative > 0 implies observations > 0
        print(f"note: {share * 100:.2f}% of rows were received before the "
              f"venue says they happened. That is a clock disagreement, not "
              f"a latency, and the delays above are measured against the "
              f"same clocks.", file=sys.stderr)
    if rep.out_of_order:
        print(f"note: {rep.out_of_order} message(s) overtook the one before "
              f"them. Anything that assumes receipt order matches venue "
              f"order is wrong that often.", file=sys.stderr)
    if rep.median_ns is not None and rep.median_ns > 0:
        print(f"for `AsOfSeries.delayed`: by_ns={rep.median_ns} for the "
              f"typical case, {rep.p95_ns} for the case worth sizing "
              f"against.", file=sys.stderr)

    if args.interval:
        stamps = [a.received_ns for a in arrivals] + \
                 [a.venue_ns for a in arrivals]
        start, stop = min(stamps), max(stamps)
        points = list(range(start, stop + 1, args.interval))
        gap = view_gap(arrivals, points)
        share = gap.share
        print(f"grid points          {gap.grid_points} "
              f"every {_fmt_ns(args.interval)}", file=sys.stderr)
        if share is not None:
            print(f"views disagree at    {gap.differ} "
                  f"({share * 100:.2f}%)", file=sys.stderr)
        print(f"first disagreement   {gap.earliest_gain_ns}", file=sys.stderr)
        print(f"largest foresight    {_fmt_ns(gap.largest_gain_ns)}",
              file=sys.stderr)
        if gap.differ:
            print("note: at those points a venue-stamped join shows a value "
                  "the process did not have yet. Whether that matters is a "
                  "question about the horizon your signal acts on, not about "
                  "the size of the number.", file=sys.stderr)

    if not args.output:
        return 0
    which = as_stamped(arrivals) if args.stamped else as_received(arrivals)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        fh.write("ts_ns,value\n")
        for ts, val in zip(_fx_timestamps(which), _fx_values(which)):
            fh.write(f"{ts},{val}\n")
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_seasonality(args: argparse.Namespace) -> int:
    session = parse_session(args.session, args.tz)
    cal = None
    if args.calendar:
        cal = read_calendar_csv(args.calendar, session, name=args.calendar)
    try:
        samples = read_samples_csv(args.input, ts_column=args.ts_field,
                                   value_column=args.value_field)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    shape = session_profile(samples, session, bucket_ns=args.bucket,
                            min_observations=args.min_observations,
                            calendar=cal, include_short=args.include_short)
    if shape.sessions == 0:
        print("error: no samples fell inside the session; check --session "
              "and --tz", file=sys.stderr)
        return 1

    print(f"sessions used        {shape.sessions}", file=sys.stderr)
    if cal is not None:
        print(f"sessions excluded    {shape.excluded} (short or non-trading)",
              file=sys.stderr)
    else:
        print("note: no --calendar, so every day is treated as full length. "
              "A half-day's close lands in a bucket that is mid-afternoon on "
              "every other day.", file=sys.stderr)
    print(f"buckets              {len(shape)} of "
          f"{_fmt_ns(args.bucket)}", file=sys.stderr)
    print(f"buckets with no data {shape.empty_buckets}", file=sys.stderr)

    # Pair each observed bucket with its value so the comparison is over
    # Decimals rather than over an Optional the reader has to reason about.
    seen = [(b.start_offset_ns, v) for b in shape.observed
            if (v := b.value) is not None]
    if seen:
        peak = max(seen, key=lambda pair: pair[1])[0]
        trough = min(seen, key=lambda pair: pair[1])[0]
        high, low = shape.factor_at(peak), shape.factor_at(trough)
        assert high is not None and low is not None   # both buckets observed
        print(f"heaviest bucket      +{_fmt_ns(peak)} into the "
              f"session ({high:.2f}x)",
              file=sys.stderr)
        print(f"lightest bucket      +{_fmt_ns(trough)} into "
              f"the session ({low:.2f}x)",
              file=sys.stderr)

    leak = profile_leak(samples, session, bucket_ns=args.bucket,
                        min_sessions=args.min_sessions,
                        min_observations=args.min_observations,
                        tolerance=Decimal(args.tolerance), calendar=cal,
                        include_short=args.include_short)
    print(f"comparable samples   {leak.knowable} of {leak.samples}",
          file=sys.stderr)
    if leak.knowable == 0:
        print(f"note: nothing to compare — fewer than --min-sessions "
              f"({args.min_sessions}) days of history exist before any "
              f"sample. Lower it or supply more data.", file=sys.stderr)
    else:
        share = leak.differing_fraction
        assert share is not None and leak.median_gap is not None
        assert leak.max_gap is not None
        print(f"factor differs by    >{args.tolerance}: {leak.differ} "
              f"({share * 100:.2f}%)", file=sys.stderr)
        print(f"median gap           {leak.median_gap * 100:.2f}%",
              file=sys.stderr)
        print(f"largest gap          {leak.max_gap * 100:.2f}%",
              file=sys.stderr)
        if leak.max_gap > Decimal("0.05"):
            print("note: the full-sample profile and the one available at the "
                  "time disagree by more than five per cent somewhere. The "
                  "worst cases are early in the sample, where the full-sample "
                  "curve is drawing on the most future.", file=sys.stderr)

    if not args.output:
        return 0
    rows = (full_sample_deseasonalise(
                samples, session, bucket_ns=args.bucket,
                min_observations=args.min_observations, calendar=cal,
                include_short=args.include_short)
            if args.full_sample else
            deseasonalise(samples, session, bucket_ns=args.bucket,
                          min_sessions=args.min_sessions,
                          min_observations=args.min_observations,
                          calendar=cal, include_short=args.include_short))
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        fh.write("ts_ns,value\n")
        for s in rows:
            fh.write(f"{s.ts_ns},{s.value}\n")
    print(f"wrote {args.output} ({len(rows)} rows of {len(samples)})",
          file=sys.stderr)
    if not args.full_sample:
        print("note: rows before --min-sessions days of history are absent "
              "rather than adjusted by a profile built from too little.",
              file=sys.stderr)
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    import csv as _csv

    def load(path: str) -> AsOfSeries:
        rows = []
        with open_text(path) as fh:
            for row in _csv.DictReader(fh):
                rows.append((int(row[args.ts_field]),
                             Decimal(row[args.value_field].strip())))
        return AsOfSeries(rows)

    try:
        left, right = load(args.left), load(args.right)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if len(left) == 0 or len(right) == 0:
        print("error: both inputs must contain observations", file=sys.stderr)
        return 1

    abs_tol = Decimal(args.absolute) if args.absolute is not None else None
    rel_tol = Decimal(args.relative) if args.relative is not None else None
    rep, mismatches = reconcile(left, right, absolute_tolerance=abs_tol,
                                relative_tolerance=rel_tol,
                                shift_right_ns=args.shift,
                                limit=args.list_limit)

    print(f"left observations    {rep.left_points}", file=sys.stderr)
    print(f"right observations   {rep.right_points}", file=sys.stderr)
    print(f"shared timestamps    {rep.common_timestamps}", file=sys.stderr)
    print(f"  agreed             {rep.agreed}", file=sys.stderr)
    print(f"  differed           {rep.value_mismatches}", file=sys.stderr)
    print(f"only in left         {rep.only_left}", file=sys.stderr)
    print(f"only in right        {rep.only_right}", file=sys.stderr)
    if rep.agreement is not None:
        print(f"agreement            {rep.agreement * 100:.2f}%", file=sys.stderr)
    if rep.max_absolute_difference is not None:
        print(f"worst difference     {rep.max_absolute_difference}",
              file=sys.stderr)

    if rep.common_timestamps == 0:
        s = suggest_shift(left, right, max_shift_ns=args.max_shift)
        if s is not None and s.explains is not None:
            print(f"note: nothing lines up, but shifting the right source by "
                  f"{s.shift_ns} ns would align {s.explains * 100:.0f}% of the "
                  f"sample. That is a clock difference, not a disagreement — "
                  f"confirm it and pass --shift.", file=sys.stderr)
        else:
            print("note: the two sources share no timestamps at all and no "
                  "constant offset explains it. Check that they describe the "
                  "same instrument and the same timestamp convention.",
                  file=sys.stderr)
    if args.absolute is None and args.relative is None:
        print("note: no tolerance given, so values had to match exactly. Two "
              "feeds differ in the last digits for reasons that are not "
              "errors; pass --absolute or --relative to say how much you "
              "accept.", file=sys.stderr)

    if not args.output:
        return 0
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["ts_ns", "kind", "left", "right", "difference"])
        for m in mismatches:
            w.writerow([m.ts_ns, m.kind.value,
                        "" if m.left is None else str(m.left),
                        "" if m.right is None else str(m.right),
                        "" if m.difference is None else str(m.difference)])
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_instruments(args: argparse.Namespace) -> int:
    import csv as _csv

    try:
        assignments = read_symbol_map_csv(args.map)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    try:
        smap = SymbolMap(assignments)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    rep = smap.report()
    print(f"assignments          {rep.assignments}", file=sys.stderr)
    print(f"symbols              {rep.symbols}", file=sys.stderr)
    print(f"instruments          {rep.instruments}", file=sys.stderr)
    print(f"reused symbols       {rep.reused_symbols}", file=sys.stderr)
    print(f"renamed instruments  {rep.renamed_instruments}", file=sys.stderr)
    print(f"open-ended bindings  {rep.open_ended}", file=sys.stderr)

    reused = smap.reused_symbols()
    for symbol, n in reused[:args.list_limit]:
        owners = " -> ".join(a.instrument_id for a in smap.assignments_of(symbol))
        print(f"  reused: {symbol} names {n} instruments: {owners}",
              file=sys.stderr)
    if len(reused) > args.list_limit:
        print(f"  ... and {len(reused) - args.list_limit} more",
              file=sys.stderr)

    if not reused:
        print("note: no ticker in this file ever named more than one "
              "instrument. Over a long history that is unusual; check that the "
              "file is point-in-time rather than a snapshot of today, because "
              "a snapshot cannot express reuse at all.", file=sys.stderr)
    if rep.open_ended == rep.assignments and rep.assignments:
        print("note: every binding is open-ended, so nothing in this file has "
              "an end date. It cannot tell you what a ticker meant in the past.",
              file=sys.stderr)

    if not args.input:
        return 0

    with open_text(args.input) as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        print("error: empty input", file=sys.stderr)
        return 1
    for field in (args.symbol_field, args.ts_field):
        if field not in rows[0]:
            print(f"error: no such column: {field}", file=sys.stderr)
            return 1

    typed = []
    bad = 0
    for r in rows:
        raw = (r.get(args.ts_field) or "").strip()
        try:
            ts = int(raw)
        except ValueError:
            bad += 1
            continue
        d = dict(r)
        d[args.ts_field] = ts
        typed.append(d)
    if bad:
        print(f"\nnote: {bad} row(s) had an unparseable {args.ts_field} and "
              f"were dropped before mapping", file=sys.stderr)

    keyed, counts = key_by_instrument(
        typed, smap,
        symbol_field=args.symbol_field, ts_field=args.ts_field,
        drop_unmapped=not args.keep_unmapped,
    )
    print(f"\nrows mapped          {counts['mapped']}", file=sys.stderr)
    print(f"rows unmapped        {counts['unmapped']}"
          f"{'  (kept)' if args.keep_unmapped else '  (dropped)'}",
          file=sys.stderr)
    print(f"rows reassigned      {counts['reassigned']}", file=sys.stderr)
    if counts["reassigned"]:
        print("note: those rows carry a ticker that named a different "
              "instrument at the time than it names now. Keyed on the string, "
              "they would have been spliced onto the wrong history.",
              file=sys.stderr)
    elif counts["mapped"]:
        print("note: no row needed reassigning. Either no ticker in this data "
              "was ever reused, or the map cannot express it.", file=sys.stderr)

    if args.segments:
        stamps = [r[args.ts_field] for r in typed
                  if r.get(args.symbol_field) == args.segments]
        stamps.sort()
        segs, unresolved = series_segments(args.segments, stamps, smap)
        print(f"\n{args.segments}: {len(segs)} segment(s), "
              f"{unresolved} unresolved observation(s)", file=sys.stderr)
        for sg in segs:
            print(f"  {sg.instrument_id}: rows {sg.start_index}..{sg.stop_index}"
                  f" ({len(sg)}), {sg.start_ns} to {sg.end_ns}", file=sys.stderr)
        if len(segs) > 1:
            print("note: more than one segment means this ticker is not one "
                  "instrument. Any statistic spanning the boundary mixes two.",
                  file=sys.stderr)

    if args.output:
        names = list(keyed[0]) if keyed else []
        with open_text(args.output, "w") as fh:
            w = _csv.writer(fh)
            w.writerow(names)
            for r in keyed:
                w.writerow([r.get(n, "") for n in names])
        print(f"\nwrote {len(keyed)} row(s) to {args.output}", file=sys.stderr)
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
                          "for a continuous venue. For daily bars set "
                          "this equal to --interval, not to the hours the "
                          "venue is open.")
    p_f.set_defaults(func=_cmd_features)

    p_l = sub.add_parser("labels",
                         help="add a forward-return label and report purged, "
                              "embargoed cross-validation folds")
    p_l.add_argument("input", help="CSV with a ts_ns column, e.g. from `mdnorm align`")
    p_l.add_argument("-o", "--output", required=True, help="output CSV")
    p_l.add_argument("--column", required=True, metavar="NAME",
                     help="price column to label")
    p_l.add_argument("--horizon", type=int, required=True, metavar="N",
                     help="label the return over the next N rows")
    p_l.add_argument("--returns", choices=[m.value for m in ReturnMethod],
                     default=ReturnMethod.SIMPLE.value,
                     help="return convention (default: simple)")
    p_l.add_argument("--splits", type=int, default=None, metavar="K",
                     help="also report K purged cross-validation folds")
    p_l.add_argument("--embargo", type=int, default=0, metavar="N",
                     help="drop N training rows after each test block; set it "
                          "to at least your longest feature window")
    p_l.add_argument("--precision", type=int, default=12, metavar="N",
                     help="significant digits in the output (default: 12)")
    p_l.set_defaults(func=_cmd_labels)

    p_u = sub.add_parser("universe",
                         help="mask a matrix to point-in-time listings and "
                              "rank across the names that really existed")
    p_u.add_argument("input", help="CSV with a ts_ns column, e.g. from `mdnorm align`")
    p_u.add_argument("-o", "--output", required=True, help="output CSV")
    p_u.add_argument("--listings", required=True, metavar="PATH",
                     help="CSV of symbol,listed,delisted (empty delisted = still listed)")
    p_u.add_argument("--columns", nargs="+", default=None, metavar="NAME",
                     help="columns to use (default: every non-age column)")
    p_u.add_argument("--rank", action="store_true",
                     help="add a cross-sectional rank per row")
    p_u.add_argument("--pct-rank", action="store_true",
                     help="add a cross-sectional percentile rank per row")
    p_u.add_argument("--zscore", action="store_true",
                     help="add a cross-sectional z-score per row")
    p_u.add_argument("--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
                     help="parse listing timestamps as epoch in this unit "
                          "(default: ISO-8601)")
    p_u.add_argument("--precision", type=int, default=12, metavar="N",
                     help="significant digits in the output (default: 12)")
    p_u.set_defaults(func=_cmd_universe)

    p_r = sub.add_parser("revisions",
                         help="measure how far corrections move a series, and "
                              "emit either the publication stream or a vintage")
    p_r.add_argument("input", help="CSV of event,known,value")
    p_r.add_argument("-o", "--output", required=True, help="output CSV")
    p_r.add_argument("--vintage", type=int, default=None, metavar="TS_NS",
                     help="write the dataset as it looked at this instant "
                          "(keyed by event time) instead of the publication "
                          "stream (keyed by when each version was readable)")
    p_r.add_argument("--ts-unit", choices=["s", "ms", "us", "ns"], default=None,
                     help="parse timestamps as epoch in this unit "
                          "(default: ISO-8601)")
    p_r.add_argument("--precision", type=int, default=12, metavar="N",
                     help="significant digits in the output (default: 12)")
    p_r.set_defaults(func=_cmd_revisions)

    p_m = sub.add_parser("metrics",
                         help="Sharpe, Sortino, drawdowns — and how much of "
                              "the result the search itself explains")
    p_m.add_argument("input", help="CSV containing a column of returns or prices")
    p_m.add_argument("--column", required=True, metavar="NAME",
                     help="column to read")
    p_m.add_argument("--prices", action="store_true",
                     help="the column holds prices; convert to returns first")
    p_m.add_argument("--returns", choices=[m.value for m in ReturnMethod],
                     default=ReturnMethod.SIMPLE.value,
                     help="return convention when --prices is given")
    p_m.add_argument("-o", "--output", default=None,
                     help="also write the figures to a CSV")
    p_m.add_argument("--risk-free", default="0", metavar="RATE",
                     help="risk-free rate per period, not per year (default: 0)")
    p_m.add_argument("--confidence", default="0.95", metavar="P",
                     help="confidence for the minimum track record length "
                          "(default: 0.95)")
    p_m.add_argument("--trials", type=int, default=None, metavar="N",
                     help="how many configurations were evaluated before this "
                          "one was chosen; enables the deflated Sharpe ratio")
    p_m.add_argument("--trial-variance", default=None, metavar="V",
                     help="variance of the Sharpe ratios across those trials")
    p_m.add_argument("--precision", type=int, default=6, metavar="N",
                     help="significant digits in the output (default: 6)")
    p_m.add_argument("--interval", type=parse_interval, default=None,
                     help="observation interval, for annualising the ratios")
    p_m.add_argument("--sessions-per-year", type=int, default=None, metavar="N",
                     help="sessions in a year, e.g. 252 for cash equities, 365 "
                          "for a continuous venue")
    p_m.add_argument("--session-length", type=parse_interval, default=None,
                     metavar="DURATION",
                     help="length of one session, e.g. 6h for equities, 24h "
                          "for a continuous venue. For daily bars set "
                          "this equal to --interval, not to the hours the "
                          "venue is open.")
    p_m.set_defaults(func=_cmd_metrics)

    p_c2 = sub.add_parser("costs",
                          help="price a trade, apply the cost to a return "
                               "series, and report the size at which the edge "
                               "runs out")
    p_c2.add_argument("input", nargs="?", default=None,
                      help="CSV of gross returns and turnover (optional)")
    p_c2.add_argument("--column", default="ret", metavar="NAME",
                      help="gross return column (default: ret)")
    p_c2.add_argument("--turnover-column", default="turnover", metavar="NAME",
                      help="one-sided turnover column (default: turnover)")
    p_c2.add_argument("-o", "--output", default=None,
                      help="write the input back with a net return column")
    p_c2.add_argument("--cost-bps", default=None, metavar="BPS",
                      help="charge this round-trip cost instead of pricing one")
    p_c2.add_argument("--notional", default=None, metavar="X",
                      help="notional of one trade, to price it from the model")
    p_c2.add_argument("--quantity", default=None, metavar="X",
                      help="quantity of one trade, in the units --adv uses")
    p_c2.add_argument("--fee-bps", default="0", metavar="BPS",
                      help="taker fee in basis points of notional")
    p_c2.add_argument("--fee-per-unit", default="0", metavar="X",
                      help="commission per share or contract")
    p_c2.add_argument("--fee-minimum", default="0", metavar="X",
                      help="minimum commission per order")
    p_c2.add_argument("--spread-bps", default="0", metavar="BPS",
                      help="quoted spread, not half of it")
    p_c2.add_argument("--spread-fraction", default="0.5", metavar="F",
                      help="how much of the spread you pay: 0.5 to cross, "
                           "0 if always passive (default: 0.5)")
    p_c2.add_argument("--impact-coefficient", default=None, metavar="C",
                      help="impact constant; there is no default because it "
                           "is calibrated to your own fills")
    p_c2.add_argument("--impact-exponent", default="0.5", metavar="E",
                      help="1/2 for the square-root law (default: 0.5)")
    p_c2.add_argument("--adv", default=None, metavar="X",
                      help="average daily volume, same units as --quantity")
    p_c2.add_argument("--volatility", default=None, metavar="X",
                      help="daily volatility as a fraction, e.g. 0.02")
    p_c2.add_argument("--edge-bps", default=None, metavar="BPS",
                      help="gross round-trip edge, to report breakeven "
                           "participation and capacity")
    p_c2.add_argument("--precision", type=int, default=6, metavar="N",
                      help="significant digits in the output (default: 6)")
    p_c2.set_defaults(func=_cmd_costs)

    p_x = sub.add_parser("mixfreq",
                         help="join a slow series onto a fast grid by when "
                              "each value became knowable, and measure what "
                              "joining on its label would have leaked")
    p_x.add_argument("periods", help="CSV of start,end,value")
    p_x.add_argument("--interval", type=int, required=True, metavar="NS",
                     help="grid spacing in nanoseconds")
    p_x.add_argument("--lag", type=int, default=0, metavar="NS",
                     help="publication lag after each period ends "
                          "(default: 0, which claims instant delivery)")
    p_x.add_argument("-o", "--output", default=None,
                     help="write both joins side by side to a CSV")
    p_x.add_argument("--start-field", default="start", metavar="NAME")
    p_x.add_argument("--end-field", default="end", metavar="NAME")
    p_x.add_argument("--value-field", default="value", metavar="NAME")
    p_x.add_argument("--name", default="value", metavar="NAME",
                     help="column name for the series (default: value)")
    p_x.set_defaults(func=_cmd_mixfreq)


    p_mb = sub.add_parser("membership",
                          help="read an index add/delete file, report what it "
                               "can support, and size the survivorship gap")
    p_mb.add_argument("changes", help="CSV of instrument_id,action,effective[,announced]")
    p_mb.add_argument("--at", type=int, default=None, metavar="NS",
                      help="report the composition at this timestamp")
    p_mb.add_argument("--basis", choices=["effective", "announced"],
                      default="effective",
                      help="which date decides membership (default: effective)")
    p_mb.add_argument("-o", "--output", default=None,
                      help="write the members at --at to a CSV")
    p_mb.add_argument("--list-limit", type=int, default=20, metavar="N",
                      help="how many dropped names to list (default: 20)")
    p_mb.add_argument("--instrument-field", default="instrument_id", metavar="NAME")
    p_mb.add_argument("--action-field", default="action", metavar="NAME")
    p_mb.add_argument("--effective-field", default="effective", metavar="NAME")
    p_mb.add_argument("--announced-field", default="announced", metavar="NAME")
    p_mb.add_argument("--name", default="", metavar="NAME")
    p_mb.set_defaults(func=_cmd_membership)


    p_tk = sub.add_parser("ticks",
                          help="check whether a price series sits on a "
                               "venue's tick grid, which raw prints do and "
                               "derived series do not")
    p_tk.add_argument("input", help="CSV with a price column")
    p_tk.add_argument("--table", required=True,
                      help="CSV of min_price,tick")
    p_tk.add_argument("--round", default="down",
                      choices=[m.value for m in Rounding],
                      help="rounding rule for the output column; the tie "
                           "rules differ only on an exact half-tick, which on "
                           "a grid is every mid (default: down)")
    p_tk.add_argument("-o", "--output", default=None,
                      help="write each price with its grid status and its "
                           "rounded value")
    p_tk.add_argument("--value-field", default="price", metavar="NAME",
                      help="price column in the input (default: price)")
    p_tk.set_defaults(func=_cmd_ticks)


    p_fx = sub.add_parser("fx",
                          help="convert a price series into another currency "
                               "as of each observation, refusing rates that "
                               "are too stale to use")
    p_fx.add_argument("input", help="CSV of ts_ns,value")
    p_fx.add_argument("rates", help="CSV of pair,ts_ns,rate")
    p_fx.add_argument("--from", dest="base", required=True, metavar="CCY",
                      help="the currency the input is priced in")
    p_fx.add_argument("--to", dest="quote", required=True, metavar="CCY",
                      help="the currency to convert into")
    p_fx.add_argument("--max-age", type=parse_interval, required=True,
                      metavar="DUR",
                      help="how old a rate may be and still be used, e.g. 1m. "
                           "Required: an as-of join with no limit will convert "
                           "a Sunday print with Friday's rate")
    p_fx.add_argument("--via", default=None, metavar="CCY",
                      help="vehicle currency for a cross; no path is searched "
                           "for, so state it or the conversion is refused")
    p_fx.add_argument("--no-inverse", action="store_true",
                      help="refuse to answer a pair by inverting the one "
                           "stored the other way round")
    p_fx.add_argument("-o", "--output", default=None,
                      help="write the converted series to a CSV")
    p_fx.add_argument("--ts-field", default="ts_ns", metavar="NAME")
    p_fx.add_argument("--value-field", default="value", metavar="NAME")
    p_fx.set_defaults(func=_cmd_fx)


    p_cal = sub.add_parser("calendar",
                           help="count the sessions and minutes a venue was "
                                "actually open, given its holidays and "
                                "half-days")
    p_cal.add_argument("calendar", help="CSV of date,kind[,close,name]")
    p_cal.add_argument("--session", metavar="HH:MM-HH:MM", required=True,
                       help="the recurring session, e.g. 09:30-16:00")
    p_cal.add_argument("--tz", default="UTC", metavar="ZONE",
                       help="timezone for --session, e.g. America/New_York")
    p_cal.add_argument("--first-day", default=None, metavar="YYYY-MM-DD",
                       help="state the first day the file covers instead of "
                            "inferring it from the exceptions in it")
    p_cal.add_argument("--last-day", default=None, metavar="YYYY-MM-DD",
                       help="state the last day the file covers")
    p_cal.add_argument("--from", dest="from_day", default=None,
                       metavar="YYYY-MM-DD",
                       help="report from this day (default: the whole file)")
    p_cal.add_argument("--to", dest="to_day", default=None,
                       metavar="YYYY-MM-DD", help="report up to this day")
    p_cal.add_argument("-o", "--output", default=None,
                       help="write the trading days and their closes to a CSV")
    p_cal.set_defaults(func=_cmd_calendar)


    p_ar = sub.add_parser("arrival",
                          help="measure the delay between the venue stamp "
                               "and the moment you received it, and say what "
                               "keying research on the venue stamp buys")
    p_ar.add_argument("input", help="CSV of venue_ns,received_ns[,value]")
    p_ar.add_argument("--interval", type=parse_interval, default=None,
                      metavar="D",
                      help="compare the venue-stamped and receipt-stamped "
                           "views on a grid of this spacing, e.g. 1s")
    p_ar.add_argument("-o", "--output", default=None,
                      help="write the receipt-stamped series to a CSV — the "
                           "one you could actually have acted on")
    p_ar.add_argument("--stamped", action="store_true",
                      help="write the venue-stamped series instead; the right "
                           "series for what the market did, the wrong one for "
                           "what you could have done")
    p_ar.add_argument("--venue-field", default="venue_ns", metavar="NAME")
    p_ar.add_argument("--received-field", default="received_ns",
                      metavar="NAME")
    p_ar.add_argument("--value-field", default="value", metavar="NAME")
    p_ar.set_defaults(func=_cmd_arrival)


    p_sn = sub.add_parser("seasonality",
                          help="measure the shape of the trading day and how "
                               "much a profile fitted on the whole sample "
                               "claims over the one available at the time")
    p_sn.add_argument("input", help="CSV of ts_ns,value")
    p_sn.add_argument("--session", metavar="HH:MM-HH:MM", required=True,
                      help="the recurring session, e.g. 09:30-16:00")
    p_sn.add_argument("--tz", default="UTC", metavar="ZONE",
                      help="timezone for --session, e.g. America/New_York")
    p_sn.add_argument("--bucket", type=parse_interval, required=True,
                      metavar="D",
                      help="width of each slot of the day, e.g. 5m. There is "
                           "no default: finer buckets describe the curve "
                           "better and put less evidence in each")
    p_sn.add_argument("--min-sessions", type=int, default=20, metavar="N",
                      help="days of history before a point-in-time profile is "
                           "used at all (default: 20)")
    p_sn.add_argument("--min-observations", type=int, default=1, metavar="N",
                      help="observations a bucket needs before it reports a "
                           "value instead of nothing (default: 1)")
    p_sn.add_argument("--tolerance", default="0.01", metavar="X",
                      help="relative gap worth counting as a disagreement "
                           "(default: 0.01)")
    p_sn.add_argument("--calendar", default=None, metavar="CSV",
                      help="calendar of date,kind[,close,name] so early "
                           "closes are left out instead of contaminating "
                           "the middle of the day")
    p_sn.add_argument("--include-short", action="store_true",
                      help="keep early-close sessions in the profile anyway")
    p_sn.add_argument("-o", "--output", default=None,
                      help="write the deseasonalised series to a CSV")
    p_sn.add_argument("--full-sample", action="store_true",
                      help="write the series adjusted by one profile over "
                           "everything; right for describing a market, wrong "
                           "as an input to something that trades")
    p_sn.add_argument("--ts-field", default="ts_ns", metavar="NAME")
    p_sn.add_argument("--value-field", default="value", metavar="NAME")
    p_sn.set_defaults(func=_cmd_seasonality)


    p_rc = sub.add_parser("reconcile",
                          help="compare two sources of the same series and "
                               "report where they disagree, keeping coverage "
                               "gaps apart from value differences")
    p_rc.add_argument("left", help="CSV of ts_ns,value")
    p_rc.add_argument("right", help="CSV of ts_ns,value")
    p_rc.add_argument("--absolute", default=None, metavar="X",
                      help="accept differences up to this absolute size")
    p_rc.add_argument("--relative", default=None, metavar="X",
                      help="accept differences up to this fraction of the left "
                           "value, e.g. 0.0001 for one basis point")
    p_rc.add_argument("--shift", type=int, default=0, metavar="NS",
                      help="move the right source forward by this many "
                           "nanoseconds before comparing")
    p_rc.add_argument("--max-shift", type=int, default=1_000_000_000,
                      metavar="NS",
                      help="how far to look for a clock offset when nothing "
                           "lines up (default: one second)")
    p_rc.add_argument("-o", "--output", default=None,
                      help="write the mismatches to a CSV")
    p_rc.add_argument("--list-limit", type=int, default=1000, metavar="N",
                      help="cap the mismatches written (default: 1000)")
    p_rc.add_argument("--ts-field", default="ts_ns", metavar="NAME")
    p_rc.add_argument("--value-field", default="value", metavar="NAME")
    p_rc.set_defaults(func=_cmd_reconcile)


    p_i = sub.add_parser("instruments",
                         help="resolve tickers to instruments as of each "
                              "timestamp, and report where a ticker was reused")
    p_i.add_argument("map", help="CSV of symbol,instrument_id,start_ns[,end_ns,venue]")
    p_i.add_argument("input", nargs="?", default=None,
                     help="CSV of rows to re-key (optional)")
    p_i.add_argument("-o", "--output", default=None,
                     help="write the re-keyed rows to a CSV")
    p_i.add_argument("--symbol-field", default="symbol", metavar="NAME",
                     help="ticker column in the input (default: symbol)")
    p_i.add_argument("--ts-field", default="ts_ns", metavar="NAME",
                     help="timestamp column in the input (default: ts_ns)")
    p_i.add_argument("--keep-unmapped", action="store_true",
                     help="keep rows the map cannot resolve instead of "
                          "dropping them; they will all share a missing key")
    p_i.add_argument("--segments", default=None, metavar="SYMBOL",
                     help="split one ticker's history where it changed instrument")
    p_i.add_argument("--list-limit", type=int, default=20, metavar="N",
                     help="how many reused tickers to list (default: 20)")
    p_i.set_defaults(func=_cmd_instruments)



    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
