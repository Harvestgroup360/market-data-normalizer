# market-data-normalizer (`mdnorm`)

[![CI](https://github.com/Harvestgroup360/market-data-normalizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Harvestgroup360/market-data-normalizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyPI](https://img.shields.io/pypi/v/market-data-normalizer.svg)](https://pypi.org/project/market-data-normalizer/)

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

```console
pip install market-data-normalizer
```

The distribution is named `market-data-normalizer`; the import name is
`mdnorm`:

```python
import mdnorm
```

Pure Python, no runtime dependencies, Python 3.10+.

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

### Corporate actions and contract rolls

A raw price series is not continuous. A 4-for-1 split divides the printed
price by four overnight, a cash dividend drops it by the amount paid, and a
futures roll steps it by the spread between the two contracts. None of them
are market moves, but all of them look like returns:

```python
from decimal import Decimal
from mdnorm import adjust_bars, split, dividend, roll, iso_to_ns

actions = [
    split(iso_to_ns("2026-06-06T00:00:00Z"), Decimal("4")),
    dividend(iso_to_ns("2026-05-09T00:00:00Z"), Decimal("0.25")),
]
clean = adjust_bars(bars, actions)
```

Back-adjustment leaves the most recent segment at the prices that actually
printed and restates everything before each event, so the joins are seamless:

```text
raw closes    500   502   498   504  │  126  125.5   127  126.5
raw returns       +0.4% -0.8% +1.2%  │ -75.0% -0.4% +1.2% -0.4%
                                     ^ the split, not a crash

adj closes    125  125.5 124.5  126  │  126  125.5   127  126.5
adj returns       +0.4% -0.8% +1.2%  │  +0.0% -0.4% +1.2% -0.4%
```

Splits scale volume as well as price. Dividends take their reference price
from the last print before the ex-date unless you pass one. Rolls support
both conventions — `AdjustMethod.RATIO` (default, preserves returns) and
`AdjustMethod.DIFFERENCE` (preserves price differences, the usual choice for
futures). Factors are composed as exact rationals, so a 1-for-2 followed by a
1-for-3 restates 600 to exactly 100 rather than 99.999...96.

Actions can come from a file, and the CLI wires it up:

```console
$ mdnorm bars trades.csv --interval 1d --actions actions.csv -o adjusted.csv
$ mdnorm bars tape.jsonl --infer-sides --every-imbalance 500 -o imbalance.csv
$ mdnorm book deltas.csv --symbol BTC-USD -o quotes.jsonl
$ mdnorm nbbo quotes.jsonl --max-age 2s -o top.jsonl
$ mdnorm tca fills.csv --market tape.jsonl --decision-price 100
```

```text
ts,kind,value,ref_price
2026-06-06T00:00:00Z,split,4,
2026-05-09T00:00:00Z,dividend,0.25,190.50
2026-03-14T00:00:00Z,roll,5312.50,5290.25
```

### Who crossed the spread

Most trade tapes give you a price and a size but not the aggressor. That one
missing field is what separates a price series from an order-flow series, and
signed volume, order imbalance and imbalance bars are all defined in terms of
it. `mdnorm.micro` infers it, using the three rules the literature settled on:

```python
from mdnorm import SideRule, infer_sides, trade_imbalance, mean_effective_spread

classified = infer_sides(events)                       # Lee-Ready by default
print(trade_imbalance(classified))                     # -1 selling .. +1 buying
print(mean_effective_spread(classified))               # 2 * |price - mid|
```

`SideRule.TICK` compares each trade with the previous different price and
needs trades only. `SideRule.QUOTE` compares the trade with the prevailing
mid. `SideRule.LEE_READY` — the default — uses the quote rule and falls back
to the tick rule at the mid. A side reported by the venue always wins;
inference only fills gaps, and trades it cannot resolve stay `None` rather
than being guessed at. Published accuracy of these rules is roughly 75-85% on
liquid names, so treat an inferred side as an estimate.

`roll_spread` estimates the effective spread from trade prices alone, via the
serial covariance that bid-ask bounce induces. It needs no quotes, which makes
it a useful cross-check on the rest — and it returns `None` rather than zero
when the covariance comes out non-negative and the estimator is undefined.

### Imbalance bars

Once trades carry a side, the sampling clock can follow order flow instead of
time or volume:

```python
from mdnorm import Pipeline

bars = Pipeline().infer_sides().imbalance_bars(Decimal("500")).run(events)
```

A bar runs until buyers have outbought sellers, or the reverse, by the
threshold. Balanced two-sided periods produce one long bar; a sustained
one-sided push produces several short ones. `by="tick"` measures the imbalance
in trade count rather than size. From the command line:

```console
$ mdnorm bars tape.jsonl --infer-sides --every-imbalance 500 -o imbalance.csv
```

### Rebuilding the order book

Exchanges do not send you a book. They send a snapshot and then a stream of
deltas, and the book only exists if you apply every one of them, in order:

```python
from mdnorm import BookDelta, OrderBook, Side, replay_book

book = OrderBook("BTC-USD", "binance")
book.apply_snapshot(ts, bids=[(D("100"), D("2"))], asks=[(D("101"), D("3"))], seq=10)

quotes = list(replay_book(book, deltas))     # one quote per change in the top
print(book.best_bid, book.spread, book.imbalance(levels=5))
```

Two failure modes make a reconstructed book silently untrue, and this
implementation refuses to hide either.

A **sequence gap** means a message was missed, and no later update repairs the
damage — the book is simply wrong from then on, in a way that looks completely
normal. `OrderBook` raises `SequenceGapError` the moment a number is skipped,
naming how many updates went missing, because the correct response is to
resynchronise from a snapshot rather than carry on. Duplicated or replayed
messages are rejected the same way. Feeds without sequence numbers work fine;
pass `strict_sequence=False` to opt out entirely.

A **crossed book** — best bid at or above best ask — is not a market state but
a symptom: a dropped delete, a stale snapshot, two venues merged by mistake.
It is exposed as `is_crossed`, and the spread goes negative rather than being
quietly clamped to zero.

`to_quote()` turns the top of the book into an ordinary `MarketEvent`, so a
reconstructed book feeds straight into session filtering, trade classification
and effective spreads with nothing in between. From the command line:

```console
$ mdnorm book deltas.csv --symbol BTC-USD --venue binance -o quotes.jsonl
```

### One instrument, several venues

When something trades in more than one place, "the price" is a question. The
consolidated top of book is the answer, and it is where three problems live
that a maximum over venues will not warn you about:

```python
from mdnorm import consolidate

top = consolidate(quotes, max_age_ns=2_000_000_000)   # 2s staleness cutoff
```

**A venue that goes quiet keeps voting.** When a feed disconnects, its last
quote stays in the consolidation forever — and a stale price is very often the
*best* price, so the dead venue ends up setting the top of book. This is the
failure that produces a consolidated feed which looks excellent and is
fiction. `max_age_ns` retires a venue that has not spoken recently;
`stale_venues()` names them.

**A consolidated book can appear crossed.** A bid on one venue above the offer
on another looks like free money and is almost always clock skew between two
feeds timestamped by different machines. `is_crossed` reports it and
`crossed_updates` counts it, because the useful response is to check the
clocks rather than to trade the spread.

**Ties need a rule.** Equal best prices are broken by size, then by venue
name, so the same input always produces the same output.

Which venue actually sets the price is a measurement in its own right, and
`leadership` counts it. The pieces compose: an order book becomes a quote,
quotes from several venues consolidate into one, and the result feeds trade
classification and effective spreads unchanged.

```console
$ mdnorm nbbo quotes.jsonl --symbol BTC-USD --max-age 2s -o top.jsonl
```

### Did I execute well?

Once the tape is clean the next question is about you rather than the market,
and every standard benchmark has a way of quietly flattering the person
running it:

```python
from mdnorm import Fill, Side, evaluate

report = evaluate(my_fills, market_trades, decision_price=D("100"))
print(report.slippage_vs_vwap_bps, report.participation_rate)
```

**Your own trades are in the benchmark.** A VWAP over the public tape includes
the prints you just made, so you end up partly benchmarking yourself against
yourself — and the bigger your share of volume, the more the benchmark bends
toward your own average price. `evaluate` removes your fills from the tape
before computing anything; `exclude_fills` does it on its own if you want the
benchmark separately. In the library's own test suite, leaving them in turns a
100 VWAP into 109 and a losing execution into a winning one.

**Participation decides whether the number means anything.** Beating VWAP by
two basis points on 0.1% of volume is a result; the same number on 30% of
volume mostly measures your own impact. The summary always reports the two
together, and the CLI says so out loud above 10%.

**Sign conventions are stated, not assumed.** Positive basis points always
mean better than the benchmark — paying below it on a buy, selling above it on
a sell. Mixed-side fills are refused rather than netted, because one number
covering both directions has no meaning.

By default the window runs from your first fill to your last. That is right
for a worked order and wrong for a single fill — the only print in the window
is then your own — so `start_ns` and `end_ns` let you score against an
interval you chose instead.

TWAP skips intervals that never traded instead of carrying the last price
forward, for the same reason nothing else here invents data.

```console
$ mdnorm tca fills.csv --market tape.jsonl --decision-price 100
```

### Several instruments, one time grid

Research wants a matrix — one row per timestamp, one column per instrument —
and building it from independent tick streams is where look-ahead bias gets in,
because every mistake here makes the backtest *better* rather than raising:

```python
from mdnorm import Field, align

rows = align({"BTC": btc_events, "ETH": eth_events},
             interval_ns=60_000_000_000,       # a one-minute grid
             max_age_ns=5 * 60_000_000_000)    # nothing older than 5 minutes
rows[0].values      # {"BTC": Decimal("60000"), "ETH": Decimal("3000")}
rows[0].ages_ns     # how old each value was at that grid point
rows[0].complete    # False if any column had nothing to show
```

**The join only looks backwards.** A value is visible at a grid point only if
it was observed at or before it. "Nearest observation" is the expensive default
in this area: on a one-minute grid it lets a print from 09:30:20 be read at
09:30:00, and twenty seconds of hindsight is enough to make a mediocre signal
look tradeable.

**A bar labelled 09:30 is not knowable at 09:30.** It contains everything that
traded until 09:31, so joining bars on their label imports an interval of the
future. `AsOfSeries.from_bars` timestamps each bar at its *end*, and
`align_bars` therefore gives you the last **closed** bar per column — one
interval further back than the naive join, and the version you could have
traded.

**Forward-filling has no natural end.** A halted or delisted stream otherwise
contributes its last price forever, and a frozen price correlates with nothing,
which reads as diversification. With `max_age_ns` a quiet column becomes
`None`; the age is still reported, so `row.stale` (had data, too old) and
`row.missing` (never had data) stay distinguishable.

**A feed you get late was not available on time.** `AsOfSeries.delayed(250ms)`
shifts observation times forward by the delivery delay, so alignment reflects
when you could have acted rather than when the source stamped it.

Nothing interpolates or smooths. `align_on` takes timestamps you supply, for
one row per print of a reference instrument, per signal, or per fill.

```console
$ mdnorm align BTC=btc.csv ETH=eth.jsonl --interval 1m --max-age 5m -o matrix.csv
```

### Features that cannot see the future

With the matrix built, the next step is turning prices into returns, z-scores,
volatility and correlations. This is the second place look-ahead gets in, and
it gets in just as quietly:

```python
from mdnorm import ReturnMethod, column, returns, rolling_zscore, realized_volatility

px  = column(rows, "BTC")
r   = returns(px, method=ReturnMethod.LOG)
z   = rolling_zscore(px, window=60)          # trailing, never full-sample
vol = realized_volatility(r, window=60)      # per period until you annualise it
```

**A full-sample z-score is look-ahead.** Subtracting the mean and dividing by
the standard deviation *of the whole series* gives every observation knowledge
of the distribution it sits in — including the part that had not happened yet.
It is one line of code and it is everywhere. Every statistic here is trailing:
the value at `i` comes from `values[i-window+1 : i+1]` and nothing else. There
is a test that pins this as a property — change the tail of the input and every
earlier output must be byte-identical — and a second test that shows the
full-sample form failing it.

**A partial window is not a result.** Until the window fills you get `None`,
not a twenty-period statistic computed from three observations. A gap inside a
window propagates for the same reason: stepping over the hole would compute a
twenty-period number from nineteen and label it twenty.

**A frozen series has no z-score and no correlation.** Zero dispersion makes
both undefined, so both return `None` rather than `0`. Reading that zero as a
correlation is how a dead feed becomes an apparent diversifier.

**There is no default annualisation factor.** √252 is right for daily bars on a
252-day calendar and wrong for almost everything else. `realized_volatility`
returns per-period volatility unless you pass a factor, and `periods_per_year`
makes you state the calendar rather than assume one — the same minute bars are
525,600 periods a year on a continuous venue and 98,280 on a cash equity
session.

```console
$ mdnorm features matrix.csv --returns log --zscore 60 --vol 60 \
      --interval 1m --sessions-per-year 365 --session-length 24h -o feats.csv
```

### Labels, and a split that does not leak

A label is the one series in a research dataset that is *allowed* to look
forward — it is the thing you are predicting. That makes it the series which
quietly contaminates every split it touches:

```python
from mdnorm import forward_returns, purged_splits

y = forward_returns(prices, horizon=5)
for split in purged_splits(len(prices), n_splits=5, horizon=5, embargo=60):
    train, test = split.train, split.test
```

**A label with a horizon makes neighbouring rows overlap.** If the label at
row `i` spans the next five bars, rows `i` through `i+5` all describe the same
stretch of future. Put row `i` in train and row `i+3` in test and the model has
already seen most of the answer. Shuffling does not help — the rows genuinely
are different rows, they merely share an outcome. `purged_splits` drops the
training samples whose label window reaches into each test block, and reports
how many it dropped.

**A gap after the test block is not enough, because features have memory.** A
rolling statistic computed just after a test period is built partly from
observations inside it. The `embargo` removes the training rows immediately
following each block; set it to at least your longest feature window. It
defaults to 0 because the right value is a property of your features, not of
this function.

**`forward_returns` looks forward on purpose.** It is the only function in the
library that does, which is why it lives in `mdnorm.labels` rather than
`mdnorm.features`. Its output belongs on the left-hand side of a model; feeding
it back in as an input is not a subtle mistake.

The purging and embargo scheme follows López de Prado, *Advances in Financial
Machine Learning* (2018), ch. 7.

```console
$ mdnorm labels feats.csv --column BTC --horizon 5 --splits 5 --embargo 60 -o ml.csv
```

### The instruments that existed then

Two of the biases this library guards against are about time. The third is
about membership: a universe assembled today did not exist in the past.

```python
from mdnorm import Universe, Listing, cross_section, cross_sectional_rank

pit = Universe([Listing("AAA", listed_ns=...), Listing("BBB", listed_ns=..., delisted_ns=...)])
ranks = cross_section(rows, cross_sectional_rank, universe=pit)
```

**Survivorship bias produces no strange values anywhere.** Take the names
listed and liquid now, pull their history, rank them against each other over
ten years, and every instrument in the study is one that survived. Unlike a
look-ahead bug there is nothing odd to spot — the numbers are all real, the
sample is just wrong.

**Excluding a name is not the same as it having no data.** A symbol that has
not listed yet, or delisted last month, belongs outside the cross-section
rather than inside it as a blank — because a blank gets treated as missing at
random, and the instruments that disappear from a market are the opposite of
random. `mask_to_universe` returns the number of cells it removed; over a long
window a count of zero usually means the listings file is present-day
membership.

**The size of the cross-section changes, and that is correct.** Percentile
ranks are computed against the members present at that moment, so the
denominator moves as instruments list and delist.

Ties share an average rank, missing names are ranked neither last nor middle,
and a flat cross-section has no z-score rather than a row of zeros.

```console
$ mdnorm universe matrix.csv --listings listings.csv --pct-rank -o pit.csv
$ mdnorm revisions gdp.csv -o published.csv
```

### Values that get corrected later

Every observation so far has had one timestamp: when it happened. A lot of real
data has two — the period it describes, and the moment it became knowable — and
then it gets revised.

```python
from mdnorm import Revision, RevisionSeries

series = RevisionSeries([
    Revision(event_ts_ns=q1, known_ts_ns=april, value=D("2.1")),
    Revision(event_ts_ns=q1, known_ts_ns=may,   value=D("1.6")),   # revised down
])
series.as_of(event_ts_ns=q1, known_ts_ns=april_20)   # 2.1 — what you knew
series.final(event_ts_ns=q1)                          # 1.6 — what is true now
```

**Using the corrected value is look-ahead, and no timestamp check will catch
it.** The row is dated correctly. The value is a real number that was genuinely
published. Nothing marks it as unavailable until three weeks later. Every guard
in `mdnorm.align` passes and the study is still wrong.

**Two honest questions, two different objects.** *What was the newest published
number at time t* is a feature — `known_series()` is keyed by publication time
and joins like any other stream. *What did the whole table look like at time t*
is a vintage — `vintage_at(t)` is keyed by event time and reproduces the sheet
as it appeared that day. Reading a vintage at the wrong moment gives a value
nobody had; there is a test that shows exactly that.

**Measure it rather than assuming.** `revision_summary()` reports how many
events were ever revised and how far first releases sat from final values. If
that number is large, every backtest built on final data has been reading
answers.

```console
$ mdnorm revisions gdp.csv -o published.csv
```

### A daily number on an intraday grid

A daily close, a settlement price, an overnight risk figure — slow series meet
fast grids constantly, and they almost always arrive labelled with the period
they *describe* rather than the moment they became *knowable*.

```python
from mdnorm import PeriodSeries, leak_report, US_EQUITY_RTH, grid

series = PeriodSeries.from_sessions(daily_closes, US_EQUITY_RTH)
feature = series.knowable_series()        # keyed at the close — safe to join
report  = leak_report(series, grid(...))  # what the label join would have cost
```

**A daily bar labelled Tuesday is not knowable on Tuesday morning.** It is
knowable once Tuesday's session closes — Tuesday evening, and later still if
the number has to be published. Join it by its label and every minute of
Tuesday sees a value that summarises, among other things, the rest of Tuesday.

**The session decides the close, not the file.** Daily bars are frequently
stamped midnight to midnight regardless of when the market was open.
`from_sessions` and `from_daily_bars` take a `Session`, so a 16:00 New York
close is 21:00 UTC in January and 20:00 in July without the caller thinking
about it.

**Publication lag is a separate claim.** A settlement price exists at the
close; it reaches you when the vendor sends it. `publication_lag_ns` is where
that goes and it defaults to zero, because a lag of zero is a statement about
your feed that only you can make.

**Measure the leak instead of arguing about it.** `leak_report` counts the grid
points where the label join shows a value that did not yet exist, and how far
ahead the worst one was read. On back-to-back periods the answer is *every
point*: the moment one value becomes readable the label has already moved to
the next. Whether that ruins a study depends on the signal, which is exactly
why the number is worth having.

```console
$ mdnorm mixfreq daily.csv --interval 60000000000 --lag 900000000000 -o joined.csv
```

### How much of the result is the search

Everything above is about getting the data right. The last step is a correct
dataset that still produces a misleading number, because the number was
chosen. A Sharpe ratio from one strategy is an estimate; the same ratio kept
after trying two hundred parameter sets is a maximum, and the maximum of two
hundred draws from noise is not small.

```python
from mdnorm import sharpe_report

rep = sharpe_report(daily_returns, periods_per_year=D(252),
                    trials=500, trial_sharpe_variance=D("0.004"))

rep.sharpe_annualised   # 0.56  — the figure that goes in the deck
rep.probabilistic       # 0.92  — probability the true ratio is above zero
rep.deflated            # 0.008 — after accounting for 500 attempts
rep.demonstrated        # False — the sample is shorter than it needs to be
rep.warnings            # what the headline number does not say
```

**Ratios are per period until you state the calendar.** `sharpe_ratio` divides
mean by standard deviation and stops; `annualise_sharpe` needs a factor, for
the same reason `realized_volatility` does. Being wrong by a constant is the
hardest kind of wrong to notice, because the shape of the series is unchanged.

**A short track record is not evidence.** `min_track_record_length` says how
many periods a ratio needs before it is distinguishable from the benchmark.
A strategy whose minimum is nine years and whose backtest is eighteen months
has not been demonstrated, however good the ratio looks.

**Selection is measurable.** `expected_max_sharpe(trials, variance)` is the
best ratio a search of that size produces from strategies that are all
worthless. `deflated_sharpe_ratio` measures your result against that instead
of against zero, following Bailey and López de Prado (2014). Pass the whole
search, not the survivors.

**Nothing here returns a flattering placeholder.** A series that never moved
has no Sharpe, a sample with no losing period has no measurable downside, a
curve that never fell has no drawdown — all `None`, not zero and not infinity.
Each of them is a statement about the sample being short.

```console
$ mdnorm metrics pnl.csv --column ret --interval 1d \
    --sessions-per-year 252 --session-length 6h \
    --trials 500 --trial-variance 0.004
```

### What the trade costs

`mdnorm.execution` measures what your fills actually cost. This is the other
question: what a backtest should charge itself for a trade it never made. It
is the crudest way a result flatters you and it survives every other check,
because nothing in the data is wrong — the strategy is simply being priced at
a level nobody trades at.

```python
from mdnorm import CostModel, Fees, ImpactModel, Liquidity, estimate, capacity

model = CostModel(fees=Fees(taker_bps=D(1)),
                  impact=ImpactModel(coefficient=D("0.5")))   # no default
liq = Liquidity(adv=D(1_000_000), volatility=D("0.02"), spread_bps=D(4))

b = estimate(model, notional=D(500_000), quantity=D(20_000), liquidity=liq)
b.commission_bps   # 1.0
b.spread_bps       # 2.0   — half of the quoted spread, because you crossed
b.impact_bps       # 14.1  — 2% of daily volume, square-root law
b.total_bps        # 17.1

capacity(D(20), model=model, liquidity=liq)   # 28,900 a day at a 20 bps edge
```

**Zero cost is not a default, it is a claim.** A backtest that charges nothing
has asserted that it trades at the midpoint, in unlimited size, for free.
Written down that way nobody would sign it.

**A cost that does not depend on size is not a cost model.** A flat five basis
points says a strategy can trade a thousand dollars and a billion on identical
terms, so every capacity question has the same answer. `estimate` says so in
its warnings every time an impact model is absent.

**There is no default impact coefficient.** The square-root law is well
supported; the constant in front of it is not universal, and a plausible wrong
one rescales every cost in the report while changing nothing about its shape.
Calibrate it against your own fills — that is what `evaluate` is for.

**The useful output is not the cost.** `breakeven_participation` is the
fraction of daily volume at which the edge is exactly consumed, and `capacity`
is the same figure as a quantity. A two-basis-point edge that breaks even at
0.3% of volume is a different object from the same edge breaking even at 30%,
and no Sharpe ratio distinguishes them.

```console
$ mdnorm costs pnl.csv --column ret --turnover-column turnover \
    --cost-bps 5 --edge-bps 20 --adv 1000000 --volatility 0.02 \
    --spread-bps 4 --fee-bps 1 --impact-coefficient 0.5
```

### A ticker is not an identifier

`canonical_symbol` makes `BTCUSDT` and `XBT/USD` agree on a spelling. This is
the other problem: the same spelling, at two different times, meaning two
different things. Exchanges reuse ticker strings — a company delists and its
symbol is reassigned, a venue renames a pair and the old name reappears
elsewhere.

```python
from mdnorm import SymbolAssignment, SymbolMap, key_by_instrument, series_segments

smap = SymbolMap([
    SymbolAssignment("ABC", "US0000000001", start_ns=t0, end_ns=t1),
    SymbolAssignment("ABC", "US0000000002", start_ns=t2),   # reused later
])

smap.reused_symbols()                 # [("ABC", 2)] — the finding
smap.instrument_at("ABC", t_mid)      # None: in the gap it named nothing
rows, counts = key_by_instrument(rows, smap)
counts["reassigned"]                  # rows the string would have mis-joined
segments, unresolved = series_segments("ABC", timestamps, smap)
```

**The bias is a join, not a bad value.** Every price in a spliced series
genuinely traded, at its own timestamp, under the ticker it carries. What is
wrong is the assumption that the column header names one thing — made once,
silently, when the matrix is built.

**Reuse looks like a merger, and mergers look profitable.** A delisting is
usually a fall and a new listing starts at a normal price, so splicing one onto
the other inserts a jump. Half the time it is upward, and an upward jump in a
name you were holding is indistinguishable from a takeover premium. The series
does not look broken; it looks lucky.

**A gap is not filled with the next owner.** Between the delisting and the
reassignment the ticker named nothing, and `instrument_at` returns `None` there
rather than the instrument that took the letters afterwards. That substitution
is the splice.

**Overlaps are refused.** One ticker bound to two instruments at the same
moment is a broken reference file, and picking one of them quietly is how the
error reaches a study. `SymbolMap` raises instead.

**No reuse in a long history is a finding, not a pass.** A file with one
open-ended binding per ticker cannot express reuse at all, so a zero means the
file rather than the market — the same shape of diagnostic as a purge that
removes nothing.

```console
$ mdnorm instruments symbol_map.csv trades.csv --segments ABC -o keyed.csv
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
$ mdnorm bars trades.csv --interval 1d --actions actions.csv -o adjusted.csv
$ mdnorm align BTC=btc.csv ETH=eth.csv --interval 1m --max-age 5m -o matrix.csv
$ mdnorm features matrix.csv --returns log --zscore 60 --vol 60 -o feats.csv
$ mdnorm labels feats.csv --column BTC --horizon 5 --splits 5 -o ml.csv
$ mdnorm universe matrix.csv --listings listings.csv --pct-rank -o pit.csv
$ mdnorm revisions gdp.csv -o published.csv
$ mdnorm metrics pnl.csv --column ret --trials 500 --trial-variance 0.004
$ mdnorm costs pnl.csv --cost-bps 5 --edge-bps 20 --adv 1e6 --volatility 0.02
$ mdnorm instruments symbol_map.csv trades.csv -o keyed.csv
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

## Benchmarks

Throughput for the hot paths, measured by a script in this repository rather
than asserted: [BENCHMARKS.md](BENCHMARKS.md). The headline finding is that
exact `Decimal` arithmetic costs 3.1× a float loop, not the order of magnitude
usually assumed — so the cost of this library is mostly Python and the
algorithm, not the exactness.

```console
$ python bench/benchmark.py
```

## Roadmap

What exists, what has been asked for, and what we have decided against is in
[ROADMAP.md](ROADMAP.md) — including the native Rust port two people have now
asked for, with an honest account of what it would and would not solve.

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
                    ├── SymbolMap.instrument_at()    which instrument the ticker named then
                    ├── timeutil.*_to_ns()           any time → ns UTC
                    ├── adjust.adjust_events()       splits/divs/rolls
                    ├── micro.infer_sides()          who crossed the spread
                    ├── book.OrderBook()             deltas → live book → quotes
                    ├── consolidate()                many venues → one best bid/offer
                    ├── evaluate()                   your fills vs the market
                    ├── align()                      N instruments → one time grid
                    ├── returns() / rolling_*()      features, trailing windows only
                    ├── purged_splits()              folds whose labels do not overlap
                    ├── Universe.members_at()        who was actually listed then
                    ├── RevisionSeries.as_of()       which version you had then
                    ├── sharpe_report()              and how much of it is the search
                    └── capacity()                   the size at which the edge runs out
```

## Tests

```bash
pip install pytest
pytest -q
```

The suite includes a cross-venue equivalence test proving CSV, WebSocket and
FIX representations of one trade collapse to an identical event, and a
causality property applied across the feature layer: change the tail of an
input, and every output before the change must be byte-identical.

## License

MIT © HarvestGroup360 (AMII LTD). See [LICENSE](LICENSE).

---

Maintained by [HarvestGroup360](https://harvestgroup360.com) as part of our
open quantitative-infrastructure tooling.
