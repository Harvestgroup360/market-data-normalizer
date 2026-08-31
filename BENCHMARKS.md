# Benchmarks

Three people have now asked us, independently, for a native Rust port of the
calculation paths. [ROADMAP.md](ROADMAP.md) says that before writing any Rust we would
publish a benchmark of the paths people actually want ported, so the argument
could be about measured numbers rather than the general reputations of two
languages.

This is that benchmark. It is a script in this repository, it uses the
standard library only, and anyone can run it:

```console
$ python bench/benchmark.py
$ python bench/benchmark.py --scale 4 --json results.json
```

Each case is warmed up and repeated; the figure reported is the **minimum**,
because noise on a shared machine only ever adds time. A mean would mostly
measure the neighbours.

## Results

Measured on a modest cloud container, which is the point: these are not
best-case numbers from a tuned machine.

```
python           3.11.15 CPython
platform         Linux 6.18 x86_64, glibc 2.39
cpu              Intel Xeon @ 2.80GHz
```

Re-run in full for 1.24.0 on a different container from the 1.17.0 table, so
the absolute figures are not comparable with the ones quoted in older
revisions of this file. The before-and-after further down was measured back to
back on this machine and is.

| Case | n | per second | each |
| --- | ---: | ---: | ---: |
| Normalize CSV rows into events | 50,000 | 182,983 | 5,465 ns |
| Aggregate 1-minute bars | 50,000 | 891,575 | 1,122 ns |
| Align 4 streams onto a grid | 40,000 | 1,768,146 | 566 ns |
| Returns | 200,000 | 1,389,202 | 720 ns |
| `rolling_sum`, window 60 | 200,000 | 1,994,266 | 501 ns |
| `rolling_mean`, window 60 | 200,000 | 1,467,853 | 681 ns |
| `rolling_mean`, window 250 | 200,000 | 1,459,414 | 685 ns |
| **Rolling z-score, window 60** | 200,000 | **35,717** | **27,998 ns** |
| `sharpe_report` | 100,000 | 592,404 | 1,688 ns |
| Resolve ticker to instrument | 200,000 | 2,447,512 | 409 ns |
| Accumulate 500k values, float | 500,000 | 51,681,739 | 19 ns |
| Accumulate 500k values, `Decimal` | 500,000 | 13,763,130 | 73 ns |

## What it says

**The rolling window is still the hot path, but not all of it is.** The sum
underneath it no longer depends on the window at all — 250 costs the same as
60, because the total is slid rather than resummed. What remains expensive is
the z-score, at 39 times a return on the same series, and that is the variance
pass: a second sweep of the window that subtracts the mean and squares, which
cannot be slid without changing the arithmetic. See below.

**`Decimal` is not the expensive part.** This is the number that surprised us,
and it is the one that matters for the Rust question. Exact decimal arithmetic
costs **3.8×** a float loop over identical values — not the one or two orders
of magnitude the folklore suggests. Everything above already runs at 34-digit
precision. So the gap between this library and a fast one is mostly Python's
interpreter and the algorithm, and only a small part of it is the exactness we
chose on purpose.

That reorders the roadmap. A language change is a large, permanent commitment
that buys back the interpreter overhead. An algorithmic change is a small one
that buys back the O(n × window), and it was available today.

**Ingest is the second cost.** 3.9 µs per CSV row is parsing, symbol
canonicalisation, timestamp conversion and building an immutable event. It is
also the one place where a caller can trivially go faster: read once,
serialise to NDJSON, and never parse that CSV again.

## What we changed, and what we refused to change

Version 1.17.0 makes the trailing statistics do less work without changing a
single number they produce.

- The gap check is carried instead of rescanned. Deciding whether a window
  contains a hole was a sweep of the whole window at every index; it is now
  one comparison against the position of the most recent hole.
- `rolling_zscore` computed the window mean twice — once directly and once
  inside `rolling_std`. It now computes it once.

Measured against the previous release on the same 200,000 points at window 60:

| | before | after | |
| --- | ---: | ---: | ---: |
| `rolling_mean` | 789 ms | 515 ms | 1.53× |
| `rolling_std` | 3,639 ms | 3,385 ms | 1.08× |
| `rolling_zscore` | 4,543 ms | 3,527 ms | 1.29× |

**Every output is identical, including its exponent.** That was verified
against the previous implementation across 144 combinations of series length,
gap density, window and `ddof`, comparing not just equality but the string
form of every `Decimal`.

The obvious remaining optimisation is a running sum: add the new value,
subtract the one leaving the window, and the whole thing becomes O(n). It
would be several times faster than what is here.

We are not doing it. `Decimal` addition rounds to the working precision, so a
running total accumulates a different rounding history than a fresh sum over
the same window — and the two disagree in the last digits, with the
disagreement depending on how far into the series you are. A library whose
central claim is that its numbers are exact and reproducible cannot have a
statistic that quietly depends on where the window sits.

### 1.24.0 — we found a way to do it after all

The paragraph above stands as written, and it is left there because it was the
right objection. What it missed is that the rounding it describes is
**observable**: `Decimal` raises an `Inexact` flag on exactly the operations
that lose a digit.

So the total is slid, with the flag cleared before every update. If the flag
comes back raised, the update is discarded and the window is summed in full.
The slid total is therefore used only on the steps where it is provably the
exact sum, and the cost of that check is one flag read per point.

| 200,000 points | before | after | |
| --- | ---: | ---: | ---: |
| `rolling_mean`, window 60 | 843 ms | 143 ms | **5.9×** |
| `rolling_mean`, window 250 | 3,016 ms | 140 ms | **21.5×** |

Measured back to back on the machine above. The window no longer appears in
the cost.

**On ordinary data every output is unchanged.** Where outputs do change, they
change in one direction and it is worth stating plainly rather than burying:
summing a window forwards can round an intermediate partial that the slid
total never holds — a window containing both `1e25` and `3.14159` is enough —
and there the slid total is the exact sum and the old recomputed one had lost
a digit. This is verified against rational arithmetic in the test suite, not
asserted: `Fraction` is the referee, and the property tested is that the slid
answer is never further from the true sum than the recomputed one.

**The variance pass is still not slid, and will not be.** Sliding it means
replacing the mean of the squared deviations with the mean of the squares
minus the square of the mean. Those are equal in algebra and a different
sequence of roundings in arithmetic, so it would change published numbers to
save time. That is the trade we declined in 1.17.0 and still decline; the
difference is that the sum never required it. This is why `rolling_zscore`
barely moved while `rolling_mean` moved 21×.

### What this means for the Rust question

The measurement that started this file said the interpreter and the algorithm
were the cost, and exact decimals were not. One of those two has now been
taken out in pure Python, at 21× on the case people actually hit, without
giving up an answer. Before committing to a second language it is worth
knowing how much of the remaining cost is the variance pass — which is
arithmetic we have chosen deliberately and would have to reproduce in Rust
exactly, or stop claiming the same numbers.

## What this does not measure

Latency under contention, memory, and anything with a network in it. The cases
here are single-threaded, warm, and CPU-bound, which is the regime a research
pipeline runs in and is *not* the regime an execution loop runs in.

Nobody should be calling `rolling_zscore` between a quote and an order. If
that is your problem, the number that matters to you is not in this table, and
we would rather see your latency budget than guess at it.

## Reproducing

```console
$ git clone https://github.com/Harvestgroup360/market-data-normalizer
$ cd market-data-normalizer
$ python bench/benchmark.py
```

No dependencies. `--scale N` multiplies every input size; `--json FILE` writes
the raw figures. If your numbers disagree with ours, the numbers are the
interesting part — open an issue with them.
