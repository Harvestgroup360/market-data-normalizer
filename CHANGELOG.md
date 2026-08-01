# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.7.0] - 2026-08-01

### Added
- Serialization: `event_to_dict`, `bar_to_dict` and `to_records` flatten
  events and bars into plain, JSON-serialisable dicts (Decimals as strings by
  default, `as_float=True` for numeric output) — ready for pandas / CSV / JSON.

## [0.6.0] - 2026-07-31

### Added
- `fill_gaps(bars)` — return a gapless bar series, inserting flat zero-volume
  bars (OHLC = previous close) for any missing interval. Pairs with
  `time_bars` and `resample_bars` for a continuous grid.

## [0.5.0] - 2026-07-30

### Added
- `resample_bars(bars, interval_ns)` — downsample OHLCV bars to a coarser
  interval (e.g. 1-minute to 5-minute) with correct OHLC aggregation and
  volume-weighted VWAP.

## [0.4.0] - 2026-07-30

### Added
- Data-quality module: `find_issues` and `clean` detect and drop bad ticks
  (price outliers), gaps, out-of-order records and non-positive price/size,
  returning a structured `QualityIssue` report.

## [0.3.0] - 2026-07-28

### Added
- OHLCV time-bar aggregation: `time_bars(events, interval_ns)` and the `Bar`
  type (open/high/low/close/volume/trades/vwap). Handles out-of-order input
  and ignores non-trade events.

## [0.2.0] - 2026-07-27

### Added
- Quote (bid/ask) normalization: `from_ws_quote` (exchange book-ticker
  messages) and `from_csv_quote` (CSV bid/ask rows).
- `MarketEvent.mid_price` and `MarketEvent.spread` convenience properties.
- Dedicated test suite for quote events (`tests/test_quotes.py`).

## [0.1.0] - 2026-07-27

### Added
- Initial release.
- Unified `MarketEvent` schema (Decimal prices, ns UTC timestamps,
  canonical `BASE-QUOTE` symbols).
- Trade normalizers for CSV, exchange WebSocket JSON, and FIX.
- Cross-venue equivalence tests.
