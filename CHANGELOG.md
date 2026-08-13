# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.5.0] - 2026-08-13

### Added
- Trade classification and microstructure metrics (`mdnorm.micro`). Most
  tapes omit the aggressor side; `infer_sides` fills it in using the tick
  rule, the quote rule, or Lee-Ready (quote rule with a tick-rule fallback
  at the mid, the default). A venue-reported side is never overwritten
  unless `overwrite=True`, and trades that cannot be resolved keep
  `side=None` instead of being guessed at. `lag_ns` matches trades against
  an earlier quote for feeds with reporting delay.
- Metrics that need a side: `signed_volume` and `trade_imbalance`
  (normalised to [-1, 1], `None` when nothing is classified).
- Metrics that do not: `effective_spreads` / `mean_effective_spread`
  against the prevailing quote, and `roll_spread`, Roll's (1984) implied
  spread from the serial covariance of price changes. Both return `None`
  rather than a misleading zero when undefined.
- `imbalance_bars`: sampling driven by directional order flow rather than
  time or volume, measured `by="volume"` or `by="tick"`. Unclassified
  trades count toward OHLCV but contribute no imbalance, so an unclassified
  stream yields one bar rather than a wrong answer.
- Matching `Pipeline` steps (`.infer_sides(...)`, `.imbalance_bars(...)`)
  and CLI flags `--infer-sides`, `--side-rule` and `--every-imbalance`
  with `--imbalance-by`.

### Notes
- Quote lookup is indexed by binary search, so classifying trades against a
  quote stream is O(n log n) rather than a scan per trade.
- Roll's estimator assumes serially uncorrelated trade signs and is biased
  upward when they are not; the limitation has its own test rather than
  being smoothed over.

## [1.4.0] - 2026-08-11

### Added
- Corporate actions and contract rolls (`mdnorm.adjust`): back-adjust a
  price series for stock splits, cash dividends and futures rolls so the
  discontinuities they create stop reading as returns. `split`, `dividend`
  and `roll` build the actions; `adjust_events` and `adjust_bars` apply
  them; `adjustment_at` reports the factors in force at any timestamp.
- Both conventions are supported: `AdjustMethod.RATIO` (default, preserves
  returns and keeps prices positive) and `AdjustMethod.DIFFERENCE`
  (preserves price differences, the usual choice for futures).
- Splits scale size and volume as well as price. Dividends resolve their
  reference price from the last print before the ex-date when one is not
  supplied. Adjustment applies strictly before an action's timestamp, so
  the ex-date's own prints are untouched.
- `read_actions_csv` reads actions from a CSV file (`ts,kind,value,
  ref_price`), with the offending line number reported on bad input.
- Matching `Pipeline` step (`.adjust(...)`, dispatching on events or bars)
  and CLI flags `--actions FILE` and `--adjust ratio|difference`.

### Changed
- Adjustment factors compose as exact rationals rather than decimals. A
  1-for-2 followed by a 1-for-3 now restates 600 to exactly 100; carrying
  the intermediate factors as `Decimal` produced 99.99999999999999999999999996.

## [1.3.1] - 2026-08-10

### Changed
- Packaging: the project is now published on PyPI as
  `market-data-normalizer` (`pip install market-data-normalizer`; the
  import name stays `mdnorm`). Added Python version classifiers, project
  URLs for the changelog and issue tracker, and an install section in the
  README. Releases are published from CI via PyPI Trusted Publishing, so
  no long-lived API token exists. No library code changed in this release.

## [1.3.0] - 2026-08-08

### Added
- Trading sessions and calendar filtering (`mdnorm.sessions`): a `Session`
  describes a recurring local-time window — `in_session`, `filter_session`,
  `session_date` and `group_by_session_date` decide what belongs to it.
  Handles intraday windows, overnight sessions that cross midnight, and
  daylight-saving transitions via `zoneinfo`; ready-made `US_EQUITY_RTH`
  and `US_FUTURES_OVERNIGHT` are included.
- Matching `Pipeline` step (`.session(...)`) and CLI flags `--session
  HH:MM-HH:MM` and `--tz ZONE`, applied before aggregation.

### Fixed
- `canonical_symbol` mangled single-listed instruments: tickers without a
  quote leg were split by a blind 3-character rule, turning `AAPL` into
  `A-APL`. Equities, ETFs and indices now keep their ticker (`AAPL`,
  `SPY`, `BRK.B`), while traded pairs are unchanged.

## [1.2.0] - 2026-08-07

### Added
- Event-driven bars, the standard alternatives to time bars: ``count_bars``
  (one bar per N trades), ``volume_bars`` (close at a cumulative base-unit
  threshold) and ``dollar_bars`` (close at a traded-notional threshold).
  For these bars ``start_ns`` is the first trade's timestamp and
  ``interval_ns`` the realized span; the trailing partial bar is included.
- Matching ``Pipeline`` steps (``.count_bars()``, ``.volume_bars()``,
  ``.dollar_bars()``) and CLI flags (``--every-trades``, ``--every-volume``,
  ``--every-notional``) as alternatives to ``--interval``.

## [1.1.0] - 2026-08-05

### Added
- Transparent gzip support across all file I/O: any ``.gz`` path
  (``.csv.gz``, ``.jsonl.gz``, ``.ndjson.gz``) is compressed/decompressed
  automatically in ``read_csv_trades`` / ``write_records_csv`` /
  ``write_jsonl`` / ``read_jsonl_events`` and the CLI. Standard library only.
- Streaming readers for large files: ``iter_csv_trades`` and
  ``iter_jsonl_events`` yield normalized events one at a time instead of
  loading the whole file into memory.

## [1.0.0] - 2026-08-04

### Added
- `Pipeline` — declarative, reusable processing chains: compose `dedupe`,
  `clean`, `time_bars`, `resample`, `fill_gaps` (plus custom steps via
  `apply`) and run the same pipeline across venues and files. Quality
  reports from `clean` are exposed on `pipeline.last_issues`.
- NDJSON / JSON Lines I/O: `write_jsonl` (events and bars, one compact JSON
  object per line) and `read_jsonl_events` / `event_from_dict` for lossless
  round-trips. Standard library only.
- Command-line interface: `mdnorm bars`, `mdnorm quality` and
  `mdnorm convert` (CSV <-> NDJSON), with human-friendly intervals
  (`30s`, `1m`, `4h`, `1d`). Installed as the `mdnorm` console script;
  also runnable as `python -m mdnorm`.

### Changed
- Project status raised to stable (`Development Status :: 5`); the public
  API of `0.x` is carried over unchanged.

## [0.9.0] - 2026-08-03

### Added
- File-level CSV I/O: `read_csv_trades(path, ...)` reads a CSV of trades into
  normalized events, and `write_records_csv(items, path)` writes events/bars
  to a CSV (union of fields). Standard library only.

## [0.8.0] - 2026-08-02

### Added
- Stream consolidation: `merge_streams(*streams)` merges multiple venue feeds
  into one timestamp-ordered timeline (stable), and `dedupe(events)` drops exact
  duplicate events from reconnects/replays, preserving first-seen order.

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
