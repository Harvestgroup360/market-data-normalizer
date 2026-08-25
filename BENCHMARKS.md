# Benchmarks

Two people asked us, independently, for a native Rust port of the calculation
paths. [ROADMAP.md](ROADMAP.md) says that before writing any Rust we would
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
cpu              Intel Xeon @ 2.10GHz, 2 cores
```

| Case | n | per second | each |
| --- | ---: | ---: | ---: |
| Normalize CSV rows into events | 50,000 | 258,129 | 3,874 ns |
| Aggregate 1-minute bars | 50,000 | 1,773,239 | 564 ns |
| Align 4 streams onto a grid | 40,000 | 3,076,375 | 325 ns |
| Returns | 200,000 | 2,457,375 | 407 ns |
| **Rolling z-score, window 60** | 200,000 | **55,615** | **17,981 ns** |
| `sharpe_report` | 100,000 | 1,085,889 | 921 ns |
| Resolve ticker to instrument | 200,000 | 4,151,382 | 241 ns |
| Accumulate 500k values, float | 500,000 | 76,750,636 | 13 ns |
| Accumulate 500k values, `Decimal` | 500,000 | 24,634,771 | 41 ns |

## What it says

**The rolling window is the hot path, and nothing else is close.** A z-score
at window 60 costs 44 times a return on the same series. That is not a
mystery: the trailing statistics are O(n × window), because every index sums
its own window from scratch. At window 60 that is sixty additions per point
where a running total would be one.

**`Decimal` is not the expensive part.** This is the number that surprised us,
and it is the one that matters for the Rust question. Exact decimal arithmetic
costs **3.1×** a float loop over identical values — not the one or two orders
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
statistic that quietly depends on where the window sits. The 1.08× on
`rolling_std` is what honesty costs here, and the remaining time is the
two-pass variance, which is the numerically correct way to compute it.

If your work genuinely needs the running-sum version, it is a few lines in
your own code and you now know exactly what you are trading for it.

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
