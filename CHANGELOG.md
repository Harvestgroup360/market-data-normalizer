# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.28.0] - 2026-09-02

### Added
- `seasonality`: the shape of the trading day, fitted without reading the rest
  of the year. Volume, spread and volatility all follow a curve inside a
  session, and a statistic computed across a day without removing it is mostly
  measuring the time of day. Removing it is the easy half.
- `expanding_profiles` hands each session a profile built only from the
  sessions **before** it, so the curve a day is divided by is one that existed
  at that day's open. `deseasonalise` uses it.
- `session_profile` builds the ordinary full-sample version and
  `full_sample_deseasonalise` applies it. Both ship deliberately: fitted over
  everything is the better estimate for describing a market and the wrong
  input to anything that trades, and having both is what lets `profile_leak`
  measure the gap rather than argue about it.
- `profile_leak` compares the two factor by factor and reports the share that
  disagree, the median gap and the largest. The worst cases cluster early in
  the sample, which is where a full-sample curve is drawing on the most
  future.
- No default bucket size. Five minutes over a 6½-hour session is 78 buckets
  and 288 on a venue that never closes; where that trade sits is a property of
  the data rather than of this library.
- A bucket below `min_observations` reports nothing rather than the mean.
  Filling a thin bucket with the average makes the adjusted series look
  well-behaved in exactly the places where nothing is known about it. A sample
  with no factor leaves the output — a point silently divided by one is a
  point claiming to have been adjusted.
- Early closes are excluded when a `TradingCalendar` is supplied, and counted.
  Bucketing by offset from the open puts a half-day's closing surge into a
  bucket that is mid-afternoon on every other day, which spoils both.
- Nothing is emitted before `min_sessions` days of history exist. A profile
  built from three days is one day's noise wearing a curve's clothes, and
  dividing by it manufactures outliers instead of removing them — the same
  rule as a rolling window that emits nothing until it is full.
- `mdnorm seasonality volume.csv --session 09:30-16:00 --bucket 5m`, which
  prints the curve, names the heaviest and lightest parts of the day, and
  reports the leak.
- 51 tests, including one that gives a single day a wild shape and checks the
  profile handed to that day is byte-identical to a profile of the days before
  it, and one that pins the arithmetic: adjusting by a factor that is itself a
  quotient rounds twice, and the implementation rearranges it to round once.

## [1.27.0] - 2026-09-01

### Added
- `arrival`: the delay between when a venue says something happened and when
  this process found out. `align.AsOfSeries.delayed` has taken a delay since
  it was written, with a docstring saying a delay of zero is a claim about
  your infrastructure rather than a default — and the library offered no way
  to measure the one you have. This is the missing half.
- `delay_report` gives the min, median, p95 and max by **nearest rank**, so
  every figure it prints is a delay that actually happened. There is no mean,
  deliberately: a transport distribution has a tail and the mean mostly
  measures it. `tail_ratio` (p95 over median) says whether the typical case
  and the bad case are the same problem.
- Receipts before the venue stamp are counted as clock skew and **never
  clamped to zero**. Clamping turns a clock problem into a latency figure that
  looks fine. Messages that overtook the one before them are counted too,
  rather than sorted away by a pipeline that then never mentions it.
- `as_received` and `as_stamped` build the two series the same rows can
  produce: the one a strategy could have acted on, and the optimistic one
  research usually builds. Both ship, because the difference between them is
  only measurable if you can hold them side by side.
- `view_gap` asks both views what they knew at each point of a grid and
  reports where they disagree, with `largest_gain_ns` — the most unearned
  foresight it found. That number is meant to be compared against the horizon
  a signal acts on, not judged on its own size.
- `assume_delay_ns=` for files that carry no receipt column at all. It sets
  `assumed=True` on the report. It is not a fallback that quietly fills in for
  missing evidence; it is a way of writing down what you decided to believe.
- `mdnorm arrival feed.csv --interval 1s`, which prints the report, names the
  `by_ns` values to hand to `delayed`, and can write out either series.
- 47 tests, including one that measures a feed's median delay, hands it to
  `AsOfSeries.delayed`, and checks the shifted series matches the received one
  exactly — the round trip the module exists to make possible.

## [1.26.0] - 2026-08-31

### Added
- The PEP 561 `py.typed` marker, so a type checker will actually use this
  package's annotations. The `Typing :: Typed` classifier is back with it —
  removed one release ago because it was not true, restored now that it is.
- A test that the marker ships in the installed package. The classifier
  outlived the marker once; this is what stops that recurring.

### Changed
- `mypy` goes from **76 errors in 11 files to zero**, with every one of the
  1,011 tests unchanged. Nearly all of them were invariants the code enforces
  at runtime but never stated in the types: a `TRADE` cannot be constructed
  without a price, so the trade filters cannot yield one; a window past the
  gap guard holds no `None`; an as-of lookup keyed by the series' own
  timestamps cannot miss.
- Twenty-one of those are resolved with `typing.cast` rather than a proof, and
  each carries a comment naming the check that guarantees it. That is the
  honest description: a `cast` asserts an invariant, it does not demonstrate
  one, and it is only acceptable where the guarantee is visible on the same
  screen. `CONTRIBUTING.md` says where the line is.
- The rest are real fixes rather than annotations. Three loops in `cli.py` and
  one in `universe.py` reused a name that was already bound to a different
  type in the same scope — the kind of shadowing that is legal, confusing to
  read, and occasionally a bug.
- `filter_session` is now generic in its element type: filtering bars returns
  `List[Bar]` rather than a union every caller has to narrow again. Behaviour
  is unchanged; the signature simply stopped losing information.
- The CI type-check job stops being reporting-only and fails the build. It was
  advisory for exactly one release, which was long enough.

## [1.25.0] - 2026-08-31

### Removed
- The `Typing :: Typed` classifier, which was not true. The package is
  annotated, but it never shipped the PEP 561 `py.typed` marker a type checker
  needs, so the classifier promised something no dependent could actually use.
  Running `mypy` against the package reports **76 errors in 11 files**, nearly
  all of them places where an invariant the code enforces at runtime is not
  expressed in the types. Shipping the marker would have pushed those errors
  into everyone else's type-checking.

### Added
- A reporting-only `types` job in CI that prints the `mypy` count on every run.
  It does not fail the build: the point is to make the number visible so it can
  be watched going down. Earning the classifier back is a roadmap item.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue templates for a
  wrong answer and for a proposal, a pull-request template, and `CITATION.cff`.
  The issue templates ask the two questions that decide most of these: what
  input reproduces it, and what would the feature have to assume.

## [1.24.0] - 2026-08-31

### Added
- `rolling_sum`: the trailing sum, the primitive underneath `rolling_mean` and
  useful on its own for trailing volume, turnover and trade counts.

### Changed
- The trailing sum is slid instead of recomputed, on every step where sliding
  it is provably exact. `BENCHMARKS.md` said in 1.17.0 that we would not do
  this, because a running `Decimal` total accumulates a different rounding
  history than a fresh sum and a library claiming reproducible numbers cannot
  have a statistic that depends on where the window sits. That objection was
  right about the rounding and wrong that it settled the question: `Decimal`
  raises an `Inexact` flag on exactly the operations that round, so every
  update runs with the flag cleared and is discarded the moment it would.
  `rolling_mean` over 200,000 points: window 60 from 843 ms to 143 ms, window
  250 from 3,016 ms to 140 ms. The window has left the cost function.
- On ordinary data every output is unchanged. Where one changes, it is because
  summing a window forwards rounded an intermediate partial the slid total
  never held, and there the slid total is the exact sum. Verified against
  `Fraction` rather than asserted: the property tested is that the slid answer
  is never further from the true sum than the recomputed one.
- The variance pass inside `rolling_std` and `rolling_zscore` is deliberately
  not slid; the identity that would allow it is a different sequence of
  roundings. This is why `rolling_zscore` barely moved.

## [1.23.1] - 2026-08-30

### Fixed
- Documentation only. The 1.23.0 text claimed that rounding a target to the
  nearest tick hands a backtest "a real gain distributed evenly across every
  order". That is stronger than the argument supports — nearest-rounding is
  symmetric, so on a symmetric target it is a wash in expectation. What is
  true and narrower: it moves an order to a price the strategy never asked
  for, and half the time that is the more aggressive side, which lifts the
  fill rate wherever a backtest fills limit orders at their limit. The
  unarguable case is the mid: a market quoted one tick wide has a mid exactly
  half a tick from both sides, so it is not a price the venue could accept.

## [1.23.0] - 2026-08-30

### Added
- `ticksize`: the price grid a venue actually accepts. `TickBand`,
  `TickTable`, `TickSchedule`, `Rounding`, `GridReport`, `grid_report`,
  `spread_in_ticks`, `read_tick_table_csv`, and the `mdnorm ticks` command.
- The diagnostic this buys: raw prints sit on the grid by construction, so a
  series that does not is a mid, a VWAP, an average across venues, a
  back-adjusted history, or an error. One pass over the file tells them apart.
- No default tick size. The familiar penny is wrong below a dollar, wrong for
  sub-penny programmes, wrong for crypto by orders of magnitude and wrong for
  the same instrument before the last regime revision, so a tick table is
  point-in-time data and `TickSchedule` refuses to answer before its first one.
- Rounding takes no default mode. On a grid an exact half-tick is not an edge
  case but every mid, so the tie rule is a systematic choice with a direction.
  `executable` rounds a buy down and a sell up.

## [1.22.0] - 2026-08-29

### Added
- `fx`: currency conversion as of the moment rather than as of the end.
  `CurrencyPair`, `Quote`, `Conversion`, `FxRates`, `ReturnDecomposition`,
  `convert_series`, `convert_bars`, `decompose_return`, `read_fx_csv`, and the
  `mdnorm fx` command.
- No function takes a scalar rate. One rate applied to a whole history
  restates it with a number that did not exist until the end of it.
- `max_age_ns` is required rather than defaulted: FX stops over weekends while
  other venues keep trading, so an as-of join with no age limit converts a
  Sunday print with Friday's close and the result looks fresh.
- Direction lives in `CurrencyPair` rather than in a naming convention, and an
  inversion is recorded in the result. No path through the currency graph is
  searched for — state the vehicle with `via=` or the cross is refused.
- `decompose_return` gives the exact identity and the additive shorthand side
  by side, so the dropped cross term is a number rather than a footnote.

## [1.21.0] - 2026-08-29

### Added
- `calendars`: holidays, half-days, and the year that is not 252 sessions.
  `Holiday`, `EarlyClose`, `TradingCalendar`, `CalendarReport`,
  `read_calendar_csv`, and the `mdnorm calendar` command.
- A calendar refuses to answer for a date its source file never covered.
  Treating an unknown weekday as open converts a missing file into a confident
  wrong answer on exactly the dates most likely to be unusual.
- `trading_seconds_between` counts what the venue was actually open, which is
  not sessions times session length once the range contains an early close.
- The command prints the `--sessions-per-year` and `--session-length` that
  `mdnorm features` wants, so the annualisation constant this project refuses
  to ship as a default is computed from a file instead of remembered.

## [1.20.0] - 2026-08-28

### Added
- `reconcile`: comparing two sources that claim to describe the same series.
  `MismatchKind`, `Mismatch`, `ReconcileReport`, `ShiftSuggestion`,
  `reconcile`, `reconcile_bars`, `suggest_shift`, and the `mdnorm reconcile`
  command.
- Coverage gaps are counted apart from value differences and never added
  together, because they have different causes and different fixes. Agreement
  is computed over shared timestamps only, so a feed that carries less does
  not look like a feed that lies.
- No default tolerance. Called with none, values must match exactly.
- Zero overlap is diagnosed as a clock offset rather than a disagreement:
  `suggest_shift` reports the constant offset and how much of the sample it
  would explain, and does not apply it.

## [1.19.0] - 2026-08-27

### Added
- `membership`: who was in an index, and when they were told. `Basis`,
  `ChangeKind`, `IndexChange`, `IndexSnapshot`, `InferredChange`,
  `MembershipHistory`, `MembershipReport`, `survivorship_gap`,
  `read_index_changes_csv`, and the `mdnorm membership` command.
- Announcement and effective dates are kept apart and `Basis` has no default,
  because they answer different questions and a study can rank on one while
  trading the effect of the other.
- `from_snapshots` dates an inferred change at the later snapshot and records
  the width of the window, rather than picking a date inside a window the data
  only bounds.
- `survivorship_gap` measures the error in both directions: a today-list drops
  the names that left and holds the names that joined from before they joined.

## [1.18.0] - 2026-08-26

### Added
- `mixfreq`: a daily number on an intraday grid. `Period`, `PeriodSeries`,
  `LeakReport`, `leak_report`, `read_periods_csv`, and the `mdnorm mixfreq`
  command.
- A period series carries the moment each value became knowable rather than
  the period it describes. `knowable_series()` is safe; `labelled_series()`
  leaks and is kept deliberately so the difference can be measured.
- The result worth stating: on back-to-back periods every grid point leaks,
  not most of them, because each period's label is the previous period's close.
- `publication_lag_ns` is a second, separate delay with no default value.

## [1.17.0] - 2026-08-25

### Added
- `bench/benchmark.py` and `BENCHMARKS.md`: throughput for the paths people
  actually use, measured rather than asserted. Standard library only, warm-up
  plus best-of-N, machine printed alongside the figures, `--scale` and
  `--json`. Published because the roadmap promised it before any discussion of
  a Rust port, so the argument could be about numbers.
- **The result reorders that discussion.** Exact `Decimal` arithmetic costs
  **3.1x** a float loop over identical values, not the order of magnitude the
  folklore suggests. Everything in the library already runs at 34-digit
  precision, so the gap between this and a fast implementation is mostly
  Python and the algorithm — and only a small part of it is the exactness we
  chose deliberately.
- The benchmark found the hot path in our own code: a rolling z-score at
  window 60 cost 44x a return on the same series, because the trailing
  statistics were O(n x window).

### Changed
- Trailing statistics do less work and produce identical numbers.
  `_windows` carries the position of the most recent gap instead of rescanning
  the window at every index, and `rolling_zscore` no longer computes the
  window mean twice — once directly and once inside `rolling_std`.
  On 200,000 points at window 60: `rolling_mean` 1.53x, `rolling_zscore`
  1.29x, `rolling_std` 1.08x.
- **Every output is identical, including its exponent**, verified against the
  previous implementation across 144 combinations of length, gap density,
  window and `ddof`, comparing the string form of every `Decimal` rather than
  just equality.

### Not changed, deliberately
- A running sum would make the trailing statistics O(n) and several times
  faster. `Decimal` addition rounds to the working precision, so a running
  total carries a different rounding history than a fresh sum over the same
  window, and the two disagree in the last digits by an amount that depends on
  how far into the series you are. A library whose central claim is exact,
  reproducible numbers cannot have a statistic that quietly depends on where
  the window sits. The 1.08x on `rolling_std` is what that costs.

### Tests
- 9 new tests (761 total) pinning the boundaries of the carried gap check,
  where an off-by-one would otherwise hide.

## [1.16.0] - 2026-08-24

### Added
- Point-in-time instrument identity (`mdnorm.instruments`): `SymbolAssignment`,
  `SymbolMap`, `SymbolMapReport`, `Segment`, `key_by_instrument`,
  `series_segments` and `read_symbol_map_csv`.
- **A ticker is not an identifier.** Exchanges reuse ticker strings, and a
  price history keyed on the string splices two unrelated instruments into one
  series with no gap, no duplicate and no error. Every price in it genuinely
  traded; the wrong part is the assumption that the column header names one
  thing, made once when the matrix is built.
- Reuse flatters. A delisting is usually a fall and a new listing starts at a
  normal price, so the splice inserts a jump, and half the time that jump is
  upward — indistinguishable from a takeover premium in a name the model was
  holding.
- `instrument_at` returns `None` in the gap between a delisting and a
  reassignment rather than the instrument that later took the letters.
  Substituting the next owner there is the splice itself.
- Overlapping assignments raise on construction. A ticker bound to two
  instruments at the same moment is a broken reference file, and resolving it
  silently is how the error survives into a study.
- `SymbolMap.report()` counts reused symbols, renamed instruments and
  open-ended bindings. A file with one open binding per ticker cannot express
  reuse, so zero reuse over a long history is a statement about the file — the
  same diagnostic shape as a purge that removes nothing.
- `key_by_instrument` re-keys rows by the instrument in force at their own
  timestamp and reports `mapped`, `unmapped` and `reassigned`; `series_segments`
  splits a ticker's history wherever it changed instrument, so a statistic is
  never computed across the boundary.
- `mdnorm instruments` subcommand.
- 66 new tests (752 total).

### Documentation
- `ROADMAP.md`: what exists, what has been asked for and what has been decided
  against. It records the native Rust port requested twice under our LinkedIn
  post, with the case for it, the reason a second implementation is a second
  place for the guarantees to drift, and the benchmark we would publish before
  writing any of it. No dates, and the decided-against list is as long as the
  proposals.

## [1.15.0] - 2026-08-23

### Added
- Forward transaction costs (`mdnorm.costs`): `Fees`, `Liquidity`,
  `ImpactModel`, `CostModel`, `estimate`, `apply_costs`, `cost_report`,
  `breakeven_participation` and `capacity`. `mdnorm.execution` measures what
  fills actually cost; this prices a trade a backtest never made.
- **Zero cost is not a default, it is a claim.** A backtest that charges
  nothing has asserted that it trades at the midpoint, in unlimited size, for
  free. This is the crudest way a result flatters you and it survives every
  other check in the library, because nothing in the data is wrong.
- **A cost that does not depend on size is not a cost model.** `ImpactModel`
  charges `coefficient * volatility * participation ** exponent`, the
  square-root law at the default exponent, so a strategy that only works at
  size is visibly one. `estimate` warns every time an impact model is absent,
  and again when participation goes beyond the range such models are usually
  calibrated over.
- **No default impact coefficient**, for the same reason there is no default
  annualisation factor: a plausible wrong constant rescales every cost in the
  report and changes nothing about its shape. `ImpactModel` requires one.
- `breakeven_participation` and `capacity` — the fraction of daily volume, and
  the quantity, at which a stated edge is exactly consumed. `None` when the
  fixed costs already exceed the edge, which is a different failure from a
  small capacity and is reported as one.
- `cost_report` compounds both series and states the share of the gross return
  that trading took, warning when a strategy is profitable before costs and
  not after them, and when turnover was zero throughout so the model was never
  exercised.
- `apply_costs` charges twice the one-sided turnover produced by
  `mdnorm.metrics.turnover`, since replacing a book is one sale and one
  purchase. A `None` in either series propagates rather than becoming a free
  period.
- `mdnorm costs` subcommand: price a trade, apply the cost to a return series,
  and report the size at which the edge runs out.

### Fixed
- The annualisation arguments could produce a silently halved figure.
  `--interval 1d --session-length 6h` describes a quarter of a bar per session
  and yields 63 periods a year rather than 252, which understates every
  annualised ratio by a factor of two and looks entirely plausible. `mdnorm
  metrics` and `mdnorm features` now say so when the interval exceeds one
  session, and the `--session-length` help states that daily bars want it set
  equal to `--interval`. The calculation was always correct; the guidance
  around it invited exactly the mistake this library exists to prevent.

### Changed
- 76 new tests (686 total).

## [1.14.0] - 2026-08-22

### Added
- Performance statistics (`mdnorm.metrics`): `sharpe_ratio`, `sortino_ratio`,
  `calmar_ratio`, `hit_rate`, `profit_factor`, `turnover`, `equity_curve`,
  `drawdowns`, `max_drawdown`, `moments`, and the `SharpeReport` that ties them
  together. No new runtime dependencies: the normal CDF and its inverse are
  built from `math.erf` and refined by a Halley step.
- **Selection is the fifth way a backtest flatters you, and it survives a
  perfect pipeline.** No value read early, no label overlapping a test block,
  no sample chosen after the fact, no figure revised — and the number is still
  wrong, because it was picked as the best of many. `expected_max_sharpe`
  reports the ratio a search of a given size produces from strategies that are
  all worthless; `deflated_sharpe_ratio` measures a result against that instead
  of against zero. Bailey and López de Prado (2012, 2014).
- `probabilistic_sharpe_ratio` and `min_track_record_length`: a ratio of 1.0
  from sixty observations and the same ratio from six hundred are the same
  number and not the same evidence. Negative skew and fat tails both lower the
  probability, so a strategy that sells insurance scores worse than its
  headline figure suggests.
- `SharpeReport.warnings` states what the ratio leaves out — that it is per
  period, that observations were missing, that the sample is shorter than the
  minimum track record length, that no trial count was supplied. Requesting a
  deflated ratio with only half its inputs is an error rather than a silent
  omission.
- No default annualisation, matching `mdnorm.features`: `annualise_sharpe` is a
  separate call that requires the calendar.
- Zero dispersion returns `None` rather than zero or infinity. A series that
  never moved has no Sharpe, a sample with no losing period has no measurable
  downside, and a curve that never fell has no drawdown; each is a fact about
  the sample length.
- A drawdown still open at the end of the sample keeps `recovery_index=None`
  instead of being closed at the last observation — the one most often dropped
  from a table, because it has no end date to put in it.
- `mdnorm metrics` subcommand, reporting the figures above and every warning
  attached to them.
- 105 new tests (610 total), including a causality property on the equity curve
  and on closed drawdowns.

## [1.13.0] - 2026-08-21

### Added
- Bitemporal observations (`mdnorm.revisions`): `Revision`, `RevisionSeries`,
  `RevisionSummary` and `read_revisions_csv`. An observation carries two
  timestamps — the period it describes and the moment it became knowable — and
  the same period may be published more than once.
- `as_of(event_ts_ns=..., known_ts_ns=...)` returns the version that had been
  released by a given moment, and `None` before the first release rather than
  the first release brought forward.
- **Using a corrected value is look-ahead that no timestamp check catches.**
  The row is dated correctly and the value was genuinely published; nothing
  marks it as unavailable until later. Every guard in `mdnorm.align` passes and
  the study is still wrong. `first_release` and `final` make the two visible
  side by side.
- Two objects for two different questions. `known_series()` is keyed by
  publication time and answers "what was the newest number available at t" —
  safe to join as a feature. `vintage_at(t)` is keyed by event time and
  reproduces the table as it appeared that day. A test demonstrates a vintage
  read at the wrong moment reporting a value nobody had.
- `revision_summary()` reports how many events were ever revised, the mean and
  maximum absolute distance from first release to final value, and the revised
  fraction. Republishing an unchanged number does not count as a revision.
- A `Revision` whose `known_ts_ns` precedes its `event_ts_ns` is rejected: a
  value available before the period it describes is a forecast, which is a
  different object with different properties.
- New CLI subcommand: `mdnorm revisions gdp.csv -o published.csv`, printing the
  revision diagnostics and writing either the publication stream or, with
  `--vintage`, the dataset as of an instant.

### Notes
- `known_series()` returns an `AsOfSeries`, so a revised feed drops straight
  into `mdnorm.align` alongside tick data with no special handling.
- A delivery delay and a revision are different problems: `AsOfSeries.delayed`
  shifts a value that never changes, while a revision replaces it. There is a
  test contrasting the two.

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
