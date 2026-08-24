# Roadmap

What exists, what has been asked for, and what we have decided against. No
dates: this is a list of intentions, and a date we missed would tell you less
than the reason a thing is on the list at all.

The organising idea has not changed. Every module here exists because there is
a way for a research pipeline to produce a number that is wrong in a direction
that flatters it, and because that class of error does not announce itself. If
a proposal does not reduce one of those, it probably belongs somewhere else.

## Where the library is

Twenty-seven tagged releases, fourteen of them published to PyPI (the
package went out under Trusted Publishing from 1.3.1 onwards). No runtime
dependencies, Python 3.10+, 752 tests.

| Layer | Modules |
| --- | --- |
| Ingest | `normalizers`, `csvio`, `jsonl`, `streams`, `records`, `symbols` |
| Instrument identity | `instruments`, `universe` |
| Cleaning | `quality` |
| Aggregation | `bars`, `sessions`, `adjust` |
| Microstructure | `book`, `consolidate`, `micro` |
| Execution | `execution` |
| Research | `align`, `features`, `labels`, `revisions` |
| Evaluation | `metrics`, `costs` |

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

What we would do first, before writing any Rust: publish a benchmark of the
paths people actually want ported, so the discussion is about measured numbers
rather than about the general reputation of two languages. If it turns out the
bottleneck is `Decimal` rather than Python, that is a different and much
smaller change.

No commitment, and no date. If you have a concrete latency budget and a path
you need inside it, open an issue with the numbers — that is more useful to us
than a vote.

## Under consideration

**Benchmarks as a published artifact.** Timings for the hot paths, produced by
a script in the repository, so the figures can be reproduced and challenged
rather than quoted. Prerequisite for the item above.

**Mixed-frequency alignment.** Joining a daily series onto an intraday grid
correctly is the same as-of problem as everything in `align`, with one extra
trap: the daily bar is not knowable until its session closes, and its label
usually says otherwise.

**Point-in-time index membership from vendor files.** `universe` applies a
membership record; producing one from the files vendors actually ship, with
their revisions and their announcement-versus-effective dates, is a separate
job and a place look-ahead hides well.

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
