# market-data-normalizer (`mdnorm`)

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
