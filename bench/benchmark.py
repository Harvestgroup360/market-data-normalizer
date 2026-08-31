#!/usr/bin/env python3
"""Throughput of the paths people actually use, measured rather than asserted.

Run it:

    python bench/benchmark.py
    python bench/benchmark.py --json results.json
    python bench/benchmark.py --scale 4        # bigger inputs, slower run

Standard library only, like the library it measures. Each case is warmed up
and then repeated; the figure reported is the **minimum**, because noise on a
shared machine only ever adds time. A mean would mostly measure the neighbours.

Absolute numbers are worth little without the machine they came from, so the
machine is printed with them. What travels between machines is the ratios —
particularly the last case, which separates the cost of ``Decimal`` from the
cost of Python itself. That distinction decides whether a faster
implementation means a different language or a different number type, and
guessing at it is how projects end up rewriting the wrong thing.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mdnorm import (  # noqa: E402
    Field,
    SymbolAssignment,
    SymbolMap,
    align,
    from_csv_row,
    returns,
    rolling_mean,
    rolling_sum,
    rolling_zscore,
    sharpe_report,
    time_bars,
)

NS = 1_000_000_000
MINUTE = 60 * NS
DAY = 86_400 * NS
T0 = 1_700_000_000 * NS


# -- harness -----------------------------------------------------------------


def measure(fn, *, repeats: int = 5, warmup: int = 1) -> float:
    """Best wall-clock time of ``repeats`` runs, in seconds."""
    for _ in range(warmup):
        fn()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - start)
    finally:
        if was_enabled:
            gc.enable()
    return best


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, unit: str, n: int, seconds: float,
            note: str = "") -> None:
        self.rows.append({"case": name, "unit": unit, "n": n,
                          "seconds": seconds,
                          "per_second": n / seconds if seconds else None,
                          "ns_each": seconds / n * 1e9 if n else None,
                          "note": note})

    def render(self) -> str:
        w = max(len(r["case"]) for r in self.rows) + 2
        out = [f"{'case'.ljust(w)}{'n':>12}  {'total':>10}  {'per second':>14}"
               f"  {'each':>11}",
               "-" * (w + 54)]
        for r in self.rows:
            out.append(
                f"{r['case'].ljust(w)}{r['n']:>12,}  {r['seconds']*1000:>8.1f}ms"
                f"  {r['per_second']:>14,.0f}  {r['ns_each']:>9,.0f}ns")
            if r["note"]:
                out.append(" " * w + r["note"])
        return "\n".join(out)


# -- fixtures ----------------------------------------------------------------


def csv_rows(n: int) -> list[dict]:
    return [{"symbol": "BTCUSDT",
             "ts": str(T0 // NS + i),
             "price": f"{30000 + (i % 500) * 0.25:.2f}",
             "size": "0.015",
             "side": "buy" if i % 2 else "sell"} for i in range(n)]


def events(n: int) -> list:
    return [from_csv_row(r, venue="bench", ts_unit="s") for r in csv_rows(n)]


def decimals(n: int) -> list[Decimal]:
    return [Decimal(f"{100 + (i % 997) * 0.01:.2f}") for i in range(n)]


# -- cases -------------------------------------------------------------------


def run(scale: int) -> Report:
    rep = Report()

    # 1. ingest -------------------------------------------------------------
    n = 50_000 * scale
    rows = csv_rows(n)
    rep.add("normalize CSV rows", "row", n,
            measure(lambda: [from_csv_row(r, venue="bench", ts_unit="s")
                             for r in rows]),
            "from_csv_row: parse, canonicalise the symbol, build a MarketEvent")

    # 2. bars ---------------------------------------------------------------
    n = 50_000 * scale
    evs = events(n)
    rep.add("aggregate 1m bars", "event", n,
            measure(lambda: time_bars(evs, MINUTE)),
            "time_bars over the same events")

    # 3. align --------------------------------------------------------------
    per = 10_000 * scale
    streams = {f"S{i}": events(per) for i in range(4)}
    total = per * 4
    rep.add("align 4 streams", "event", total,
            measure(lambda: align(streams, interval_ns=MINUTE,
                                  field=Field.PRICE, max_age_ns=5 * MINUTE)),
            "as-of join onto a one-minute grid, with a staleness limit")

    # 4. features -----------------------------------------------------------
    n = 200_000 * scale
    px = decimals(n)
    rep.add("returns", "point", n, measure(lambda: returns(px)))
    rep.add("rolling sum, window 60", "point", n,
            measure(lambda: rolling_sum(px, 60), repeats=3),
            "the total is slid, and recomputed only where sliding would round")
    rep.add("rolling mean, window 60", "point", n,
            measure(lambda: rolling_mean(px, 60), repeats=3))
    rep.add("rolling mean, window 250", "point", n,
            measure(lambda: rolling_mean(px, 250), repeats=3),
            "same cost as window 60: the sum no longer depends on the window")
    rep.add("rolling z-score, window 60", "point", n,
            measure(lambda: rolling_zscore(px, 60), repeats=3),
            "the mean is slid; the variance pass is not, and it dominates")

    # 5. evaluation ---------------------------------------------------------
    n = 100_000 * scale
    rr = [x for x in returns(decimals(n)) if x is not None]
    rep.add("sharpe_report", "observation", len(rr),
            measure(lambda: sharpe_report(rr)),
            "four moments, the ratio, and its probabilistic form")

    # 6. instrument identity ------------------------------------------------
    n_sym = 2_000 * scale
    assignments = []
    for i in range(n_sym):
        s = f"SYM{i:05d}"
        assignments.append(SymbolAssignment(s, f"ID-{i}-A", start_ns=T0,
                                            end_ns=T0 + 500 * DAY))
        assignments.append(SymbolAssignment(s, f"ID-{i}-B",
                                            start_ns=T0 + 900 * DAY))
    smap = SymbolMap(assignments)
    probes = [(f"SYM{i % n_sym:05d}", T0 + (i % 1400) * DAY)
              for i in range(200_000 * scale)]
    rep.add("resolve ticker to instrument", "lookup", len(probes),
            measure(lambda: [smap.instrument_at(s, t) for s, t in probes]),
            f"binary search over {len(assignments):,} assignments")

    # 7. the question that decides what a faster version would be -----------
    n = 500_000 * scale
    ds = decimals(n)
    fs = [float(x) for x in ds]

    def sum_float():
        s = 0.0
        for x in fs:
            s += x
        return s

    def sum_decimal():
        with localcontext() as ctx:
            ctx.prec = 34
            s = Decimal(0)
            for x in ds:
                s += x
            return s

    tf = measure(sum_float)
    td = measure(sum_decimal)
    rep.add("accumulate, float", "value", n, tf)
    rep.add("accumulate, Decimal", "value", n, td,
            f"{td / tf:.1f}x the float loop on identical values")
    rep.rows[-1]["decimal_tax"] = td / tf
    return rep


# -- main --------------------------------------------------------------------


def machine() -> dict:
    cpu = ""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "cpu": cpu or platform.processor() or "unknown"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scale", type=int, default=1,
                    help="multiply every input size (default: 1)")
    ap.add_argument("--json", metavar="FILE", default=None,
                    help="also write the raw numbers here")
    args = ap.parse_args()
    if args.scale < 1:
        ap.error("scale must be at least 1")

    info = machine()
    print("machine")
    for k, v in info.items():
        print(f"  {k:16} {v}")
    print(f"  {'scale':16} {args.scale}")
    print()

    rep = run(args.scale)
    print(rep.render())
    print()
    tax = next((r.get("decimal_tax") for r in rep.rows if "decimal_tax" in r), None)
    if tax:
        print(f"Decimal costs {tax:.1f}x a float loop here. Everything above runs "
              f"at 34-digit\nprecision, so that multiple is the price of exact "
              f"decimal prices — not the\nprice of Python, which is the larger "
              f"share of every figure in this table.")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"machine": info, "scale": args.scale,
                        "results": rep.rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
