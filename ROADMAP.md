# Roadmap

What exists, what has been asked for, and what we have decided against. No
dates: this is a list of intentions, and a date we missed would tell you less
than the reason a thing is on the list at all.

The organising idea has not changed. Every module here exists because there is
a way for a research pipeline to produce a number that is wrong in a direction
that flatters it, and because that class of error does not announce itself. If
a proposal does not reduce one of those, it probably belongs somewhere else.

## Where the library is

Fifty-eight tagged releases, forty-five of them published to PyPI (the
package went out under Trusted Publishing from 1.3.1 onwards). No runtime
dependencies, Python 3.10+, 1790 tests, and a type checker that passes clean.

| Layer | Modules |
| --- | --- |
| Ingest | `normalizers`, `csvio`, `jsonl`, `streams`, `records`, `symbols` |
| Instrument identity | `instruments`, `universe`, `membership` |
| Cleaning | `quality`, `reconcile`, `ticksize`, `resolution`, `staleness`, `coverage` |
| Aggregation | `bars`, `sessions`, `calendars`, `auctions`, `halts`, `adjust`, `fx` |
| Microstructure | `book`, `consolidate`, `micro` |
| Execution | `execution` |
| Research | `align`, `arrival`, `features`, `labels`, `revisions`, `mixfreq`, `seasonality` |
| Evaluation | `metrics`, `costs`, `compounding`, `serial`, `underwater`, `hurdle`, `rebalance`, `fundfees`, `independence`, `breadth`, `exposure`, `extremes`, `windows`, `multiverse` |
| Reproducibility | `provenance` |
| Measured | [`bench/benchmark.py`](bench/benchmark.py), [BENCHMARKS.md](BENCHMARKS.md) |

Shipped since the last revision of this file: `mixfreq`, `membership`,
`reconcile`, `calendars`, `fx`, `ticksize`, `arrival`, `seasonality`, `resolution`, `auctions`, `independence`, `staleness`, `halts`, `coverage`, `provenance`, `extremes`, `windows`,
`multiverse`, `breadth`, `exposure`, `compounding`, `serial`, `underwater`, `hurdle`, `rebalance` and `fundfees`. The first two were the items that stood under
*Under consideration* below; the other four were not on the list. `reconcile` is
here because comparing two sources of the same series is the check people run
before trusting either, and nothing in the library did it. A slow series now carries the
moment each value became knowable rather than the period it describes, and
`leak_report` counts the grid points a label-keyed join would have answered
too early. The result on back-to-back periods is worth stating plainly: the
naive join is wrong at every point, not most of them. `membership` builds an
index history out of the add/delete files and periodic snapshots vendors
actually ship, keeps the announcement and effective dates apart because they
answer different questions, refuses to pick a date inside a window a snapshot
only bounds, and measures the survivorship gap in both directions. `reconcile` compares
two feeds of the same series, keeps coverage gaps apart from value
differences instead of averaging them into one match rate, and diagnoses the
common case where zero overlap is a clock offset rather than a disagreement.
`calendars` is the smallest of the four and closes the oldest gap: `sessions`
described a recurring window and nothing described the exceptions to it, so a
holiday was indistinguishable from an outage and a half-day was silently
counted as a full one. It also makes the constant this file refuses to ship a
computable number — the sessions in a year, and the minutes in them, come out
of the calendar rather than out of 252. A calendar refuses to answer for a
date its source file never covered, which is the same rule the rest of the
library follows: report the gap, do not fill it.

`multiverse` is the newest and the one that finishes an argument the last two
releases started. `windows` counts the alternatives in *where the sample
begins*; `provenance` records the arguments a run was given; `multiverse`
crosses the arguments themselves. Three cleaning decisions on one unchanged
series produce twelve pipelines and an annualised Sharpe ratio between 0.4254
and 0.8682, and the attribution says the clipping threshold owns 0.3096 of
that 0.4428 — which is a more useful sentence than *the number is unstable*.
It ships with a caveat we considered leaving out and did not: cells of a
shared grid are not independent, so the specification count is an upper bound
on the effective number of trials rather than the number itself. Deflating by
it is conservative and still wrong, and estimating the effective count would
need a model of how the decisions correlate. We do not have one, so we say so
instead of shipping a number that looks like we do.

`fundfees` is the newest, and it is the last step between a backtest and a
statement anyone could invest on. Everything else here is gross; an investor
is paid net, and the fee is not a constant subtracted from the return. Ten
years that earned 116.92 per cent leave 55.44 under two and twenty with annual
crystallisation and a high-water mark, so more than half of the profit did not
reach the investor. On the same returns, monthly crystallisation without a
mark takes 83.29 per cent of it.

The pattern is the one `rebalance` put on the record in the previous release: an
unstated schedule decides the answer. There it was how often the book is
traded back; here it is how often the fee is taken and whether losses are
remembered. Neither appears in the name of the contract, which is why every
field of the schedule is required and none is defaulted.

`rebalance` closes a gap that had been sitting in plain
sight. Every evaluation module here reads a return series, and nothing said
where the return series came from. A weight vector is a decision made once;
what happens to it afterwards is arithmetic, and most backtests quietly snap
the weights back to target at every observation. That earns a return nobody
could have had without trading, and the trading is never reported.

Five names over five years, equal weight, identical returns throughout:
rebalancing every period returns 11.7901 per cent and needs 5.6217 of
one-sided turnover, never rebalancing returns 11.3466 per cent and needs none.
The forty-four basis points between them are gone at a cost of 7.8893 basis
points per unit of turnover. `breakeven_cost_bps` reports that level rather
than a recommendation, and its sign is kept when the more active schedule
earned less, because a negative cost reads as a bargain when it is printed as
a magnitude.

The result worth putting on the record is that turnover is monotone in the
frequency and the return is not. Across every 1, 5, 21, 63 and 252 periods the
returns run 11.7901, 11.3488, 10.1833, 11.3573 and 10.9495 per cent. There is
no ordering to find. The differences are noise, the trading is not, and a
module that picked a frequency would be dressing the first up as the second.

It charges nothing. `costs` prices a trade and this says how much trading a
rule implies; that seam is deliberate, because a cost model that arrived
attached to a schedule would stop being a thing the caller stated. The same
applies to the residual weight, which is carried as cash at a zero return and
pointed at `hurdle` rather than quietly credited with a rate.

`hurdle` is about the other side of every ratio in the
library. A Sharpe ratio measures a return against something, and that
something is usually zero or one constant for the whole sample. Cash paid
close to nothing for a decade and then five per cent, so the first choice
credits a strategy with the cash return and the second one misdates it. On
twenty years of a cash-plus book the per-period Sharpe ratio is 0.486326
against nothing and 0.257868 against the rate actually paid — 1.6847 and
0.8933 annualised — and forty-six per cent of the gross return was the hurdle.

Two of its decisions follow the pattern the last three releases set. The gap
against no hurdle is reported with a direction, because a non-negative rate
can only make that figure larger; the gap between a constant rate and the rate
series is reported without one, because its sign follows the correlation and
the rate's own variance in the sample. The test suite carries a sample of each
sign rather than a sentence claiming both are possible.

The other is that the module converts a quoted rate but will not choose the
convention. Five per cent over 252 periods is 0.000198413 divided and
0.000193631 compounded, and a money-market quote on a 360-day year used
against a 365-day calendar understates the hurdle by 1.39 per cent of itself.
Both are small per period, both run one way for the whole sample, and both
land on the hurdle rather than on the return. No cash curve ships here, for
the same reason no factor data ships with `exposure`.

`underwater` is the smallest idea in the library stated
carefully. A maximum drawdown is a maximum. It is the worst single observation
in a sample, which makes it an order statistic, and order statistics grow with
how long you look. The same unchanged returns give a median worst decline of
0.1344 over a year and 0.2982 over ten; a two-year backtest and a ten-year one
are not reporting the same quantity, and neither of them says so.

What the module adds beside it is the part people actually live through. The
README strategy has a maximum drawdown of 20.9 per cent, which sounds
survivable, and spent 92.94 per cent of five years below a previous high with
one stretch of 281 trading days without a new one. Depth and duration are
different questions and only one of them is usually asked. The ulcer and pain
indices are there for the same reason: they read every observation, so no
single day can set them, and their ratio to the maximum says whether a curve
sat near its worst or dipped once.

`resampled_max_drawdown` ships with its assumption on the outside. It draws
with replacement, which destroys serial correlation, and losses that arrive in
runs make drawdowns deeper than independent losses do. On a positively
autocorrelated series it is therefore a lower bound on the drawdown rather
than an estimate of it — the flattering direction — and `serial` from the
previous release is how a caller finds out whether that describes their data.
The CLI note says so where someone reading output will see it, not only in the
docstring.

`serial` closes a gap the library had been walking past.
Every other module here is careful about the numbers going in; this one is
about the last multiplication on the way out. Annualising a Sharpe ratio by
the square root of the calendar is correct only for independent returns, and
a smoothed series — an appraisal mark, a stale quote, a price on something
that did not trade — is not independent. Twenty years of monthly returns with
a lag-one autocorrelation of 0.238 annualise to 0.671692 by the familiar
factor and to 0.486672 by Lo's, and nothing in the data is wrong in either
case.

Two decisions in it are worth recording. The first is that the report refuses
to promise a direction: negative autocorrelation makes the square root of time
understate, so the gap is called a difference and a flag says which way this
series runs. That is the `total_gap` lesson from 1.41.0, applied before
shipping instead of after. The second is that truncation is surfaced rather
than smoothed over. The factor wants `q - 1` autocorrelations and nobody
annualising daily returns has 251 worth trusting; supplying fewer treats the
rest as zero, which pulls the answer back toward the naive one. That is the
flattering direction on a correlated series, so a truncated result is labelled
a lower bound on the correction rather than an estimate of it.

It also ships a measured caveat rather than a claimed one. The variance ratio
is biased toward one at long horizons because overlapping windows share
observations, and we state the size: on four thousand draws from a process
whose asymptotic ratio at twelve periods is 1.857, the estimator returns about
1.66. The bias makes a dependent series look independent, which is the
direction a reader needs to know about.

`breadth` is `independence` asked sideways. That module
counts how many independent observations overlapping labels leave along the
time axis; this one counts how many independent bets a correlation structure
leaves across the names. Forty series driven by one market are 3.628 effective
bets, so the position count overstates by eleven and — because breadth sits
under a square root in the fundamental law — an information ratio computed on
it is overstated by three. It reports two counts that are deliberately not
interchangeable, and the eigenvalues come from a Jacobi rotation written for
`Decimal`, which keeps the dependency list empty. An early draft of its
documentation claimed the two counts agree on an equicorrelated matrix. They
do not, the arithmetic says so plainly, and the claim was removed before
release rather than after — which is the only reason it is worth mentioning
here.

`exposure` is the newest and the bluntest. It regresses a strategy on factors
the caller supplies and reports how much of the mean return survives. On the
worked example a headline annualised Sharpe of 1.16 becomes 0.55 once the
market is subtracted, with a residual t-statistic of 1.2 — and the market
carries three quarters of the mean. Two commitments come with it. No factor
data will ever ship here, for the same reason no vendor does: bundling one
would make every answer partly a property of whose definition of momentum we
chose. And the module will not say a strategy has alpha. It reports what
survives a list somebody else picked, because the absence of an exposure is
evidence about the factor list rather than about the strategy, and that
distinction is the whole reason the module is worth having.

`compounding` is the newest and the most ordinary, which is why it took this
long to notice was missing. An average monthly return of one per cent
annualises to 12.68 per cent if you compound the average and to 12.05 in the
account, and the gap is the variance — so the statistic flatters a book in
proportion to its risk. Leverage is worse than proportional: two times the
returns cost 3.97 times the drag on our example series. Worth recording that
the module's first draft compared the sum of the returns against the
compounded total and called the difference an overstatement. That comparison
has no fixed sign, the account beats the sum about as often as it trails it,
and the error survived until the numbers were actually run. It is in the
changelog under its own heading.

Not a module, but the change in 1.24.0 belongs in this list: the trailing sum
under `rolling_mean` is now slid instead of recomputed, which took it from
O(n x window) to O(n) — 21x at window 250. `BENCHMARKS.md` said in 1.17.0 that
we would not do this, because a running `Decimal` total rounds differently
from a fresh sum and a library claiming exact numbers cannot have a statistic
that depends on where the window sits. That objection was right; what it
missed is that the rounding is observable. Every update runs with the
`Inexact` flag cleared and is discarded the moment it would round. On ordinary
data nothing changed; where anything did, the slid total is the exact sum and
the old one had lost a digit, which the test suite checks against rational
arithmetic rather than asserting.

`fx` is the newest and the one we expected to be simplest. A price is a number
and a currency, most pipelines carry only the number, and the moment a study
spans two venues that quote differently every figure in it depends on a second
series nobody was watching. The module converts as of each observation and has
no function that takes a single rate, because a single rate restates a whole
history using a number that did not exist until the end of it. `max_age_ns` is
required rather than defaulted, since FX stops over weekends while other venues
do not. Direction is carried in the type instead of inferred from a pair name,
and an inversion is recorded in the result rather than performed quietly. No
path through the currency graph is ever searched for: state the vehicle or the
cross is refused, because choosing a route is choosing whose spreads you pay.

`ticksize` is the smallest module here and the one that answers a question we
had not seen asked anywhere: is this file prints, or is it derived numbers?
A venue only accepts multiples of a tick, so raw prints sit on the grid by
construction, and a series that does not is a mid, a VWAP, an average across
venues, a back-adjusted history or an error. One pass tells them apart. The
module ships no default tick size — the familiar penny is wrong below a
dollar, wrong for sub-penny programmes, wrong for crypto by orders of
magnitude and wrong before the last regime change — and tick tables are
therefore point-in-time data, so `TickSchedule` refuses to answer before the
first table it was given. Rounding takes no default mode either: on a grid an
exact half-tick is not an edge case but every mid, so the tie rule is a
systematic choice with a direction, and `executable` rounds a buy down and a
sell up so that rounding can never improve a backtested fill.

`arrival` is the most recent, and it exists because the library had been
giving an instruction it gave no way to follow. `AsOfSeries.delayed` has taken
a delivery delay since it was written, and its docstring has said all along
that a delay of zero is a claim about your infrastructure rather than a
default — while nothing here would tell you what yours is. Now `delay_report`
measures it from data carrying both stamps. It reports by nearest rank and
publishes no mean, because a transport distribution has a tail and the mean
mostly measures it; every figure it prints is a delay that actually happened.
A receipt earlier than the venue stamp is counted as clock skew and never
clamped, since clamping converts a clock problem into a latency figure that
looks fine, and messages that overtook each other are counted rather than
sorted away. `as_received` and `as_stamped` build both series the same rows
can produce, and `view_gap` measures how far apart they are on a grid, which
is the only way the difference between "what the market did" and "what I could
have done" stops being an argument. There is no default delay anywhere in the
module: state an assumption and the report carries `assumed=True`, because a
report that hides which of the two it used is worse than no report.

`seasonality` is the newest and it extends the same argument one layer up.
The library already refuses to read a value before it was published and
refuses to pick an index universe with hindsight; a profile of the trading day
is the same mistake in a shape people do not recognise as one. The usual
recipe fits one intraday curve over the whole sample and divides every day by
it, so a heavy open in January is judged against a curve that already contains
December, and the adjusted series comes out smoother than anything computable
at the time. Smoother inputs make better-looking signals. So
`expanding_profiles` gives each session a curve built only from the sessions
before it, `session_profile` builds the full-sample version, and `profile_leak`
measures how far apart they are — both ship, because the full-sample fit is
genuinely the better description of a market and genuinely the wrong input to
something that trades, and a difference nobody can compute is a difference
nobody checks. There is no default bucket width, a bucket below the evidence
threshold reports nothing rather than the average, and an early-close session
is left out of the curve instead of dropping its closing surge into a bucket
that is mid-afternoon on every other day.

`resolution` is the newest and the smallest, and it answers a question we had
never seen asked of a market-data file: what can these timestamps actually
distinguish? Everything in this library is an integer nanosecond, which is a
storage decision and not a claim about any feed. A vendor stamping to the
millisecond and handing over nanoseconds has multiplied by a million, and the
six trailing zeros are indistinguishable from precision until somebody divides.
The module divides. It reports the coarsest decimal unit that fits, refuses to
answer from too few distinct values — twenty timestamps all dividing by ten is
a one-in-10^20 coincidence on a real nanosecond feed, three is nothing — and
counts the rows that share a timestamp, because those are in the order the
writer used rather than an order the data records. The payoff is
`classification_risk`, which takes the consequence out of the abstract: for
every trade it compares the quote an as-of join picks against the last quote
provably in an earlier tick, and reports how many side classifications rest on
a tie and how many actually change. On a millisecond feed where trades and
quotes share stamps, that second number is not small.

`auctions` is the newest, and it follows directly from the one before it. The
opening and closing crosses are single prints at a single price, aggregating
orders that never met each other in a book, and nothing about them behaves
like a trade: there is no aggressor to classify, the price sits at the end of
the day's range rather than in it, and the size is often a large fraction of
the session. A VWAP with the closing cross in it is dominated by one print, so
a strategy that never traded the auction is scored against a price it could
not have obtained, and one that only traded the auction beats the benchmark by
construction. Both benchmarks ship, with the distance between them in basis
points, because the failure here is not using the wrong one — it is not saying
which. The windows come from the trading calendar, so a half-day's cross lands
where the venue actually closed, and their extents default to zero rather than
to the thirty seconds everybody uses, since that constant differs by venue and
by decade. Nothing is inferred from print size: a rule that calls anything ten
times the median a cross reclassifies ordinary blocks on a busy day, and the
statistic that comes out describes the threshold instead of the market.

`independence` is the newest and it is the first module here that sits
between two others rather than beside them. `labels` has produced overlapping
forward returns since early on, and `purged_splits` has removed the training
rows whose label windows reach into a test block — which stops the overlap
leaking across a split and does nothing about it inflating the sample within
one. A thousand daily observations of a five-day label carry about two hundred
pieces of information, so every t-statistic, Sharpe and confidence interval
computed on the thousand is out by a factor of roughly 2.2, in the flattering
direction, silently. The overlap case is exact arithmetic: the labels state
their own windows, so counting how many are live at each point gives the
effective total with no model behind it. The autocorrelation case is an
estimate and is marked as one, with the sum truncated at the first
non-positive lag, because continuing into the noise can report an effective
sample larger than the nominal one — the single outcome this module exists to
rule out.

`staleness` is the newest, and it is the second module written to measure
something this library had already put in writing. `align` has warned since it
was written that a frozen price is uncorrelated with everything and therefore
reads as diversification; like the delay in `arrival`, that was advice with no
instrument attached. The run counts are arithmetic and carry no
interpretation, because a flat stretch on an illiquid instrument and a vendor
repeating yesterday's mark produce identical rows and nothing in the data
separates them. What the module does argue is the consequence: a reported
series that carries part of the previous period's move is a moving average of
the true one, a moving average has less variance than what it averages, and a
lower volatility against an unchanged mean raises the Sharpe, lowers the beta
and shrinks every correlation at once. That adjustment is a model rather than
a measurement — a two-period average with weights inferred from the
first-order autocorrelation — so it is flagged as modelled, it declines to
treat a negative autocorrelation as staleness, and it refuses outright above
an autocorrelation of one half, which a two-period average cannot produce.

## Asked for

**A native Rust port of the core normalization and calculation paths.**
Requested three times now, independently, under
[our LinkedIn post](https://www.linkedin.com/company/harvestgroup-360) — for
the normalization logic, for the calculators, and most recently with a
specific question about the binding: PyO3 against a plain C ABI, from someone
offering to work on the execution calculators. In every case the destination
is a low-latency execution path rather than a research one.

No FFI has been chosen, because choosing one is the second decision. The first
is which paths are worth moving, and the answer to that changed in 1.24.0: the
trailing sum went from O(n x window) to O(n) in pure Python, 21x at window
250, without altering an answer. What is left in the hot path is the variance
pass, and that is arithmetic we picked deliberately — a port would have to
reproduce its rounding exactly or stop claiming the same numbers, which is a
harder specification than it sounds and the thing we would want settled before
any binding question.

This is the clearest signal we have received and we are taking it seriously,
so it is worth being precise about what it would and would not be.

The honest case for it: this library is `Decimal` arithmetic in pure Python.
That is a deliberate choice for research — exact decimal prices, no silent
binary rounding, integer nanosecond timestamps — and it is the wrong choice
inside an execution loop, by roughly two orders of magnitude. Nobody should be
calling `rolling_zscore` between a quote and an order.

The honest difficulty: a port is not a translation. The guarantees that make
this library worth using are behavioural — a join that searches backwards
only, a window that emits nothing until it is full, a guard that reports what
it removed, a `None` where a flattering zero would fit. A second
implementation is a second place for those to drift, and a fast library that
disagrees with the slow one about where a fold boundary falls is worse than no
fast library at all. If we do this, the two have to be tested against each
other on the same inputs, and that harness is most of the work.

We said we would publish a benchmark before writing any Rust, and
[BENCHMARKS.md](BENCHMARKS.md) is it. The result changed our view of this
item. Exact decimal arithmetic costs **3.1×** a float loop over the same
values — not the order of magnitude the folklore suggests — so `Decimal` is
not where the time goes. The trailing statistics were, at O(n × window), and
1.17.0 addressed part of that in Python without altering a single output.

That does not close the question; an interpreter is still an interpreter. It
does mean the honest ordering is algorithm first, language second, and that a
port would be buying back interpreter overhead rather than the cost of being
exact.

No commitment, and no date. If you have a concrete latency budget and a path
you need inside it, open an issue with the numbers — that is more useful to us
than a vote.

## Under consideration

**Position sizing and portfolio construction.** Volatility targeting and
constraint handling would close the loop between `features`, `costs` and
`metrics`. Held back deliberately: this is where a data library starts making
investment decisions, and we would rather be sure the layer underneath is
right first.

## Decided against

**A backtest engine.** There are good ones, and the reason strategies fail is
almost never the event loop. Adding one would make this a framework you adopt
rather than a library you call.

**A default annualisation factor, a default impact coefficient, or a default
anything that cannot be right for every market.** Asked for more than once. A
plausible wrong constant rescales an entire report while leaving its shape
untouched, which makes it the hardest kind of error to notice. State the
calendar, state the coefficient.

**Star ratings, benchmark leaderboards, or published comparisons against other
libraries.** We are not a neutral party about our own software.

**Bundling a data vendor.** The library reads what you have. Tying it to one
feed would narrow it to whoever already pays for that feed, which is the
asymmetry we are trying to reduce.

## Contributing

Issues and pull requests are welcome, from anyone. Two things make a proposal
easy to act on: a concrete input that produces a wrong answer, and a statement
of what the right answer is. A test that fails is worth more than a paragraph
that is correct.

MIT licensed. Maintained by [HarvestGroup360](https://harvestgroup360.com).
