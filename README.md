# market-data-normalizer (`mdnorm`)

[![CI](https://github.com/Harvestgroup360/market-data-normalizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Harvestgroup360/market-data-normalizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Normalize heterogeneous market-data feeds — CSV tick dumps, exchange
WebSocket JSON, and FIX — into a single, exchange-agnostic event schema, so
downstream research and execution code never has to care where a tick came
from.

Zero runtime dependencies. Pure Python (3.10+). `Decimal` prices, integer
nanosecond timestamps.

## Why

Every venue spells the same thing differently: `BTCUSDT` vs `XBT/USD`,
millisecond epochs vs FIX `UTCTimestamp`, `is_buyer_maker` booleans vs side
codes. Research notebooks and backtesters end up littered with per-venue
parsing branches. `mdnorm` pushes that mess to the edge and hands the rest of
your stack one clean type.

## Install

```bash
pip install -e .
```

## Quick start

```python
from mdnorm import from_csv_row, from_ws_json, from_fix

# CSV row (ISO-8601 timestamp)
from_csv_row(
    {"symbol": "btc/usd", "ts": "2026-01-02T00:00:00Z",
     "price": "42000.5", "size": "0.25", "side": "buy"},
    venue="coinbase",
)

# Exchange WebSocket trade message
from_ws_json({"s": "BTCUSDT", "p": "42000.5", "q": "0.25",
              "T": 1767312000000, "m": False}, venue="binance")

# FIX execution report (SOH-delimited in the wild; "|" here for readability)
from_fix("55=BTC/USD|31=42000.5|32=0.25|54=1|60=20260102-00:00:00",
         venue="lmax", sep="|")
```

All three calls above produce the **same** `MarketEvent`.

### Quotes (bid/ask)

```python
from mdnorm import from_ws_quote

q = from_ws_quote(
    {"s": "BTCUSDT", "b": "41999.5", "B": "1.2",
     "a": "42000.5", "A": "0.8", "T": 1767312000000},
    venue="binance",
)
q.mid_price   # Decimal("42000.0")
q.spread      # Decimal("1.0")
```

`from_csv_quote` does the same for CSV rows with bid/ask columns.

### OHLCV bars

```python
from mdnorm import time_bars

bars = time_bars(events, interval_ns=60_000_000_000)  # 1-minute bars
bars[0].open, bars[0].high, bars[0].low, bars[0].close, bars[0].volume, bars[0].vwap
```

`time_bars` reduces a stream of trade events into fixed-interval OHLCV `Bar`s
(with VWAP and trade count), sorting out-of-order input and skipping quotes.

`resample_bars(bars, interval_ns)` downsamples bars to a coarser interval
(e.g. 1-minute → 5-minute) with correct OHLC aggregation and volume-weighted
VWAP.

`fill_gaps(bars)` returns a gapless series, inserting flat zero-volume bars
(OHLC = previous close) for any interval with no trades — a continuous grid for
backtests and feature pipelines.

### Event-driven bars

Time bars are not the only clock. Sample by activity instead:

```python
from decimal import Decimal
from mdnorm import count_bars, volume_bars, dollar_bars

count_bars(events, every=500)                       # tick bars
volume_bars(events, min_volume=Decimal("100"))      # volume bars
dollar_bars(events, min_notional=Decimal("1e6"))    # dollar bars
```

### Trading sessions

Filter a feed down to the hours that matter, with daylight saving handled
for you:

```python
from mdnorm import US_EQUITY_RTH, filter_session, group_by_session_date

rth = filter_session(events, US_EQUITY_RTH)        # 09:30-16:00 New York
by_day = group_by_session_date(events, US_EQUITY_RTH)
```

Overnight windows (a session that opens at 18:00 and closes at 17:00 the
next day) are supported, and `session_date` keeps a whole night in one
bucket. From the command line:

```console
$ mdnorm bars trades.csv --interval 5m --session 09:30-16:00 --tz America/New_York -o rth.csv
```

### Data quality

```python
from mdnorm.quality import find_issues, clean

find_issues(events)          # list of QualityIssue (outlier / gap / out_of_order / non_positive)
cleaned, issues = clean(events)  # drop bad ticks & invalid rows, keep a report
```

`clean` removes price outliers and non-positive price/size records and returns
the surviving events plus everything it flagged.

### Serialization

```python
from mdnorm import to_records

to_records(events)                 # list of flat dicts (Decimals as strings)
to_records(bars, as_float=True)    # numeric output for DataFrames
```

`to_records` (and `event_to_dict` / `bar_to_dict`) flatten events and bars into
plain, JSON-serialisable dicts — drop straight into `pandas.DataFrame`, a
`csv.DictWriter`, or `json.dumps`.

### Consolidating streams

```python
from mdnorm import merge_streams, dedupe

timeline = dedupe(merge_streams(binance_events, coinbase_events))
```

`merge_streams` interleaves multiple venue feeds into one timestamp-ordered
timeline; `dedupe` drops exact duplicate events left behind by reconnects and
replays.

### CSV files

```python
from mdnorm import read_csv_trades, write_records_csv

events = read_csv_trades("trades.csv", venue="coinbase")   # file -> events
write_records_csv(bars, "bars.csv", as_float=True)          # events/bars -> file
```

`read_csv_trades` parses a whole CSV of trades into normalized events;
`write_records_csv` writes events or bars back out. Standard library only.

### NDJSON / JSON Lines

```python
from mdnorm import write_jsonl, read_jsonl_events

write_jsonl(events, "events.jsonl")          # one JSON object per line
events2 = read_jsonl_events("events.jsonl")  # lossless round-trip

# large files: stream lazily, .gz handled transparently
for e in iter_jsonl_events("dump.jsonl.gz"):
    ...
```

### Pipelines

Declare a processing chain once, reuse it everywhere:

```python
from decimal import Decimal
from mdnorm import Pipeline

pipe = (
    Pipeline()
    .dedupe()
    .clean(max_return=Decimal("0.1"))
    .time_bars(60_000_000_000)   # 1-minute bars
    .fill_gaps()
)
bars = pipe.run(events)
print(pipe.last_issues)          # quality report from clean()
```

### Command line

The common conversions ship as a zero-dependency CLI:

```console
$ mdnorm bars trades.csv --venue binance --interval 1m -o bars.csv
$ mdnorm quality trades.csv --max-gap 5m
$ mdnorm convert trades.csv -o trades.jsonl
```

Also available as `python -m mdnorm`.

## The unified schema

```python
@dataclass(frozen=True, slots=True)
class MarketEvent:
    symbol: str          # canonical "BASE-QUOTE", e.g. "BTC-USD"
    venue: str           # source venue
    event_type: EventType  # TRADE | QUOTE
    ts_ns: int           # nanoseconds since Unix epoch (UTC)
    price: Decimal | None
    size:  Decimal | None
    side:  Side | None     # BUY | SELL
    # ... plus bid/ask fields for quotes
```

## Design notes

- **Money is `Decimal`.** Prices and sizes never touch binary floats, so
  `42000.10` stays `42000.10`.
- **Time is integer nanoseconds, UTC.** One comparable integer regardless of
  whether the source gave seconds, milliseconds, or a FIX timestamp string.
- **Symbols are canonicalized** to `BASE-QUOTE`, with venue aliases resolved
  (`XBT` → `BTC`) and quote currencies detected longest-match-first so
  `USDT` wins over `USD`.
- **Normalizers are pure functions** — one raw record in, one `MarketEvent`
  out — which keeps them trivial to unit-test and compose into any streaming
  or batch pipeline.

## Architecture

```
raw feed ──► normalizer ─────────────► MarketEvent ──► your pipeline
 (CSV /      (from_csv_row /            (unified,       (research,
  WS JSON /   from_ws_json /             immutable)      backtest,
  FIX)        from_fix)                                  execution)
                    │
                    ├── symbols.canonical_symbol()   BTCUSDT → BTC-USDT
                    └── timeutil.*_to_ns()           any time → ns UTC
```

## Tests

```bash
pip install pytest
pytest -q
```

The suite includes a cross-venue equivalence test proving CSV, WebSocket and
FIX representations of one trade collapse to an identical event.

## License

MIT © HarvestGroup360 (AMII LTD). See [LICENSE](LICENSE).

---

Maintained by [HarvestGroup360](https://harvestgroup360.com) as part of our
open quantitative-infrastructure tooling.
