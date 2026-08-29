# Roadmap

What exists, what has been asked for, and what we have decided against. No
dates: this is a list of intentions, and a date we missed would tell you less
than the reason a thing is on the list at all.

The organising idea has not changed. Every module here exists because there is
a way for a research pipeline to produce a number that is wrong in a direction
that flatters it, and because that class of error does not announce itself. If
a proposal does not reduce one of those, it probably belongs somewhere else.

## Where the library is

Thirty-three tagged releases, twenty of them published to PyPI (the
package went out under Trusted Publishing from 1.3.1 onwards). No runtime
dependencies, Python 3.10+, 939 tests.

| Layer | Modules |
| --- | --- |
| Ingest | `normalizers`, `csvio`, `jsonl`, `streams`, `records`, `symbols` |
| Instrument identity | `instruments`, `universe`, `membership` |
| Cleaning | `quality`, `reconcile` |
| Aggregation | `bars`, `sessions`, `calendars`, `adjust`, `fx` |
| Microstructure | `book`, `consolidate`, `micro` |
| Execution | `execution` |
| Research | `align`, `features`, `labels`, `revisions`, `mixfreq` |
| Evaluation | `metrics`, `costs` |
| Measured | [`bench/benchmark.py`](bench/benchmark.py), [BENCHMARKS.md](BENCHMARKS.md) |

Shipped since the last revision of this file: `mixfreq`, `membership`,
`reconcile`, `calendars` and `fx`. The first two were the items that stood under
*Under consideration* below; the other three were not on the list. `reconcile` is
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

## Asked for

**A native Rust port of the core normalization and calculation paths.**
Requested twice, independently, under
[our LinkedIn post](https://www.linkedin.com/company/harvestgroup360) — once
for the normalization logic and once for the calculators, in both cases to sit
inside a low-latency execution path rather than a research one.

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
