# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.12.0] - 2026-08-20

### Added
- Point-in-time membership and cross-sectional operations
  (`mdnorm.universe`): `Listing`, `Universe`, `mask_to_universe`,
  `cross_sectional_rank`, `cross_sectional_zscore`, `cross_section` and
  `read_listings_csv`.
- **Survivorship.** A universe assembled today did not exist in the past. Rank
  the currently listed names against each other across ten years and every
  instrument in the study is one that survived. `Universe.members_at(ts)`
  answers who was actually tradable at a moment, and `cross_section(...,
  universe=...)` masks each row to that answer before ranking.
- A listing interval is half-open: `listed_ns` inclusive, `delisted_ns`
  exclusive, so an instrument is not a member on the day it stops trading. A
  symbol may list, delist and relist; every interval is kept.
- `mask_to_universe` reports how many cells it removed. Over a long window a
  count of zero usually means the listings file is present-day membership
  rather than a historical record — which is the definition of the bias.
- Missing names are ranked neither last nor middle: they are not in the
  ordering at all. Ties share an average rank. Percentile ranks use the number
  of members actually present, so the denominator follows the size of the
  cross-section. A flat cross-section has no z-score rather than a row of
  zeros, matching the treatment of zero dispersion in `mdnorm.features`.
- New CLI subcommand: `mdnorm universe matrix.csv --listings listings.csv
  --rank --pct-rank --zscore -o pit.csv`. It writes the number of members per
  row, prints the masked-cell count and the minimum and maximum cross-section
  size, and says so when nothing was masked.

### Notes
- A forward-filled price does not stop when an instrument does, so without a
  universe a delisted name keeps taking a place in the ranking. There is a test
  that demonstrates exactly that, and the same case end to end from ticks.
- This completes the three biases the library is built around: reading a value
  before it was observable (`align`), testing on labels that overlap the
  training set (`labels`), and studying a sample chosen after the fact
  (`universe`).

## [1.11.0] - 2026-08-19

### Added
- Labels and leak-free splitting (`mdnorm.labels`): `forward_returns`,
  `purged_splits`, `purged_train_test`, and a `Split` carrying the train and
  test indices together with how many rows it discarded and why.
- `forward_returns(values, horizon=N)` is the only function in the library that
  looks forward, and it does so on purpose — it produces a label, not a
  feature. It lives in its own module for that reason, and the final `N` rows
  are `None` because their outcome has not happened yet.
- **Purging.** A label with a horizon makes neighbouring rows describe the same
  stretch of future. A training row whose label window reaches into a test
  block has already shown the model most of the answer, and shuffling does not
  help, because the rows genuinely are different rows that merely share an
  outcome. `purged_splits` removes them and reports the count.
- **Embargo.** Features have memory, so a rolling statistic computed just after
  a test period is built partly from inside it. `embargo=N` drops the `N`
  training rows following each test block. It defaults to 0 because the correct
  value is the longest feature window in your dataset, which this function
  cannot know; the CLI says so when it is left at zero.
- A parametrised test asserts the core invariant directly across six
  combinations of sample count, fold count, horizon and embargo: no training
  index in any fold has a label window overlapping that fold's test block. A
  second test builds the naive contiguous split and shows it failing the same
  check with five leaking rows.
- New CLI subcommand: `mdnorm labels feats.csv --column BTC --horizon 5
  --splits 5 --embargo 60 -o ml.csv`, which appends the label column and prints
  a per-fold report of train size, test size, purged and embargoed counts.

### Notes
- Test blocks are contiguous and in time order, and every sample is tested
  exactly once. A shuffled split of a series with overlapping labels leaks in
  both directions at once.
- A fold's training set includes rows from after its test block, which is what
  cross-validation means and what the embargo is for. Strictly walk-forward
  evaluation is a matter of taking the training indices below the block.
- The scheme follows López de Prado, *Advances in Financial Machine Learning*
  (2018), ch. 7.

## [1.10.0] - 2026-08-18

### Added
- Features (`mdnorm.features`) computed on the matrix `align` produces:
  `returns` (simple or log), `rolling_mean`, `rolling_std`, `rolling_zscore`,
  `rolling_correlation`, `realized_volatility`, plus `column` and `timestamps`
  for getting series in and out of an aligned matrix.
- **Every statistic is trailing.** The value at index `i` is computed from
  `values[i - window + 1 : i + 1]` and nothing else. A full-sample z-score —
  standardising against the mean and standard deviation of the entire series —
  hands every observation knowledge of a distribution nobody had at the time.
  A parametrised test pins causality as a property across all seven functions:
  change the tail of the input and every earlier output must be identical. A
  second test demonstrates the full-sample form failing exactly that check.
- A partial window returns `None` rather than a short statistic wearing the
  full window's name, and a gap inside a window propagates instead of being
  stepped over.
- Zero dispersion gives `None`, not `0`, from `rolling_zscore` and
  `rolling_correlation`. A frozen column is usually an unexpired forward-fill,
  and reading its correlation as zero is how a dead feed looks like a hedge.
- `periods_per_year(interval_ns, sessions_per_year=..., session_length_ns=...)`
  requires the calendar to be stated. `realized_volatility` returns per-period
  volatility unless a factor is supplied, because no annualisation default is
  safe: the same minute bars are 525,600 periods a year on a continuous venue
  and 98,280 on a cash equity session.
- New CLI subcommand: `mdnorm features matrix.csv --returns log --zscore 60
  --vol 60 --interval 1m --sessions-per-year 365 --session-length 24h -o
  feats.csv`. Without the three annualisation flags it says the volatility is
  per period rather than silently picking a calendar.

### Fixed
- `returns([])` returned `[None]` instead of `[]`. Output length now always
  equals input length, so a feature series can be written back alongside the
  grid it came from without an off-by-one.

### Notes
- `rolling_std` defaults to the sample form (`ddof=1`); a rolling window is a
  sample of a longer process, not a population.
- Statistics are computed at 34 digits of working precision. The CLI trims the
  output to 12 significant digits by default (`--precision`), since writing 34
  into a CSV is noise rather than accuracy.

## [1.9.0] - 2026-08-17

### Added
- As-of alignment (`mdnorm.align`): `align` puts several instruments on one
  regular time grid, `align_on` aligns to timestamps you supply, and
  `align_bars` does it for bar series. Each `AlignedRow` carries a value per
  column *and* the age of that value, so a row is self-describing.
- `AsOfSeries` is the queryable primitive: `at(ts)` returns the last value
  observed at or **before** `ts`, never the nearest in either direction.
  Reaching forward for a closer observation is how look-ahead bias usually
  enters an as-of join, and it produces a better backtest rather than an error.
- `AsOfSeries.from_bars` timestamps every bar at `end_ns`, not at its label.
  A one-minute bar labelled 09:30 covers everything until 09:31, so joining on
  the label imports an interval of the future; `align_bars` consequently gives
  the last *closed* bar per column.
- `max_age_ns` expires a forward-filled value. A halted or delisted stream
  otherwise contributes its last price to every later row, and a frozen price
  correlates with nothing, which reads as diversification. `AlignedRow.stale`
  (had data, too old) and `.missing` (never had data) are reported separately.
- `AsOfSeries.delayed(by_ns)` shifts observation times forward by a delivery
  delay, so alignment reflects when a feed was actually available rather than
  when the source stamped it.
- `Field` (price / mid / bid / ask) and `BarField` select which number a column
  takes. Events lacking the requested field are skipped, not filled — a trade
  has no mid, and neither does a one-sided quote.
- New CLI subcommand: `mdnorm align BTC=btc.csv ETH=eth.jsonl --interval 1m
  --max-age 5m -o matrix.csv`. It reports how many rows came out complete, and
  says so when `--max-age` is omitted.

### Notes
- The default window runs from the first grid point that can hold data to the
  first one at or after the last observation, so no leading row is empty and
  no observation is left out of every row. Pinning `start_ns` and `end_ns`
  makes separate runs line up row for row.
- `grid()` refuses to build more than 10 million rows. A nanosecond grid over a
  day is a wrong interval, not a request, and allocating it is worse than
  failing.
- Nothing here interpolates, smooths, or invents a value between observations.

## [1.8.0] - 2026-08-16

### Added
- Execution benchmarks (`mdnorm.execution`): `vwap`, `twap`,
  `average_fill_price`, `slippage_bps`, `implementation_shortfall_bps`,
  `participation_rate`, and `evaluate()` returning an `ExecutionSummary`
  with all of them together. A `Fill` records one of your own executions.
- `exclude_fills` removes your own prints from a public tape before the
  benchmark is computed. Leaving them in means benchmarking yourself partly
  against yourself, and the larger your share of volume the more the
  benchmark bends toward your own average price. `evaluate` does it by
  default; `exclude_own=False` opts out.
- Participation rate is measured against the *full* tape, including your own
  volume, and is reported alongside the score so the two cannot be read
  apart. The CLI warns above 10%, where a VWAP score largely measures impact.
- Sign convention fixed and documented: positive basis points always mean
  better than the benchmark. Mixed-side fills raise rather than netting into
  a number with no meaning.
- `evaluate` accepts explicit `start_ns` / `end_ns`. The default window —
  first fill to last — is right for a worked order and degenerate for a
  single fill, where the only print in the window is your own.
- New CLI subcommand: `mdnorm tca fills.csv --market tape.jsonl
  --decision-price 100`.

### Notes
- TWAP skips intervals that never traded rather than carrying the previous
  price forward, consistent with the rest of the library refusing to invent
  data to fill a silence.
- A test pins the degenerate single-fill case: when the only print in the
  window is your own, removing it leaves no market, so the summary returns a
  null VWAP and 100% participation instead of a fabricated score.

## [1.7.0] - 2026-08-15

### Added
- Multi-venue quote consolidation (`mdnorm.consolidate`). `Consolidator`
  keeps the best bid and offer across venues for one symbol; `consolidate()`
  runs a whole stream and emits one event per change in the consolidated top.
  Output carries the `CONSOLIDATED` venue label, since the two sides can come
  from different places.
- `max_age_ns` retires a venue that has stopped quoting, with `stale_venues()`
  and `fresh_venues()` to inspect it. Without a cutoff a disconnected feed
  keeps contributing its last quote forever, and because a stale price is
  often the best price, the dead venue ends up setting the top of book.
- `is_crossed` and `crossed_updates` report a bid above an offer across
  venues rather than hiding it — nearly always clock skew between feeds, and
  worth investigating rather than trading.
- Deterministic tie-breaking: equal best prices resolve by size, then by
  venue name, so the same input always produces the same output.
- `leadership` counts how many updates each venue spent at the top of each
  side, and `VenueTop` reports which venue is showing the current best.
- New CLI subcommand: `mdnorm nbbo quotes.jsonl --max-age 2s -o top.jsonl`,
  which also prints venue leadership, a crossed-book count and any venues
  left stale at the end.

### Notes
- Emission is decided on the values a consolidated event actually carries,
  not on the internal `VenueTop` objects: a venue re-sending an identical
  quote changes its source timestamp without changing anything a consumer
  would see, and must not produce an event.
- A guard against two-sided-empty quotes was dropped from the consolidator
  after a test showed `MarketEvent` already rejects them at construction.
  The test now pins that upstream behaviour instead.

## [1.6.0] - 2026-08-14

### Added
- Order book reconstruction (`mdnorm.book`). `OrderBook` maintains a
  single-symbol limit order book from a snapshot plus incremental
  `BookDelta` updates, exposing `best_bid`, `best_ask`, `mid`, `spread`,
  `depth(side, levels)` and `imbalance(levels)` — resting-size imbalance,
  distinct from the executed-trade imbalance in `mdnorm.micro`.
- `SequenceGapError`: a skipped sequence number raises immediately, naming
  how many updates went missing, instead of leaving a book that is wrong in
  a way nothing downstream can detect. Duplicate and replayed messages are
  rejected the same way, the book is left untouched when a delta is
  rejected, and a fresh snapshot resynchronises. `strict_sequence=False`
  for feeds without sequence numbers.
- `is_crossed` reports a bid at or above the ask rather than normalising it
  away; the spread goes negative and stays visible.
- `to_quote()` emits the top of book as a `MarketEvent`, so a reconstructed
  book feeds session filtering, trade classification and effective spreads
  unchanged. `replay_book()` streams those quotes, by default only when the
  top actually changes.
- `max_depth` trims to a fixed number of levels per side for feeds that
  publish bounded depth.
- New CLI subcommand: `mdnorm book deltas.csv --symbol BTC-USD -o quotes.jsonl`,
  with `--max-depth`, `--every-update` and `--ignore-sequence`.

### Notes
- Price levels are kept in a sorted list with binary-search insertion, so
  applying a delta is O(log n) in the number of levels rather than a re-sort.
  50,000 deltas replay in about 0.1 s.
- `replay_book` seeds its comparison from the book's current top, so
  replaying a deep delta onto a populated book emits nothing rather than a
  spurious first quote.

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

## [1.4.0] - 2026-08-12

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

## [1.3.0] - 2026-08-09

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

## [1.1.0] - 2026-08-06

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

## [0.7.0] - 2026-08-02

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

## [0.3.0] - 2026-07-29

### Added
- OHLCV time-bar aggregation: `time_bars(events, interval_ns)` and the `Bar`
  type (open/high/low/close/volume/trades/vwap). Handles out-of-order input
  and ignores non-trade events.

## [0.2.0] - 2026-07-28

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
