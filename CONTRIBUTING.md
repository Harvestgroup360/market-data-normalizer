# Contributing

Issues and pull requests are welcome, from anyone. This file says what makes
one easy to act on, and what we will push back on.

## The most useful thing you can send

**A failing test.** A concrete input that produces a wrong answer, plus a
statement of what the right answer is, is worth more to us than a paragraph
that is correct. It converts a discussion into a decision.

If you are not sure the behaviour is wrong, open an issue with the input
anyway and say what you expected. Half of the entries in `CHANGELOG.md` under
*Fixed* started as somebody being confused, not somebody being certain.

## What this library is for

Every module here exists because there is a way for a research pipeline to
produce a number that is wrong in a direction that flatters it, and because
that class of error does not announce itself. A proposal that reduces one of
those is on-topic. A proposal that adds convenience at the cost of an unstated
assumption is not, however convenient.

`ROADMAP.md` has a section headed **Decided against** with the reasoning
beside each entry. If your idea is on that list, the useful move is not to
re-propose it but to attack the reason. That has worked: the sliding window
sum was refused for seven releases and shipped in 1.24.0 when somebody looked
hard enough at the objection.

## House rules that are not negotiable

These are the properties the tests exist to protect. A change that breaks one
of them will be declined even if it is faster or shorter.

- **Nothing looks forward.** A value at index `i` is computed from data at or
  before `i`. There is a property test that pins this: change the tail of an
  input and every earlier output must be byte-identical.
- **No default constant that cannot be right for every market.** No default
  annualisation factor, tick size, impact coefficient, publication lag or
  reconciliation tolerance. Ask the caller; refuse if they did not say.
- **Two kinds of missing stay apart.** Nothing observed yet is not the same as
  observed and stale, and neither is the same as the venue being shut.
- **A partial window is not a result.** Emit nothing until the window is full.
- **Report the gap, do not fill it.** Where the data does not support an
  answer, raise or return `None` and say why — never a plausible substitute.
- **Exact decimal arithmetic.** Prices are `Decimal`, timestamps are integer
  nanoseconds. A change that alters a published number to save time is a
  different proposal from a change that makes the same number arrive sooner,
  and the two get judged differently.

## Running things

```console
pip install pytest
pytest -q                       # the whole suite
python bench/benchmark.py       # the performance figures, standard library only
python -m mdnorm --help         # the command line
```

The suite is fast on purpose. If a change makes it slow, that is worth
discussing before it is worth merging.

## Type annotations

The package is annotated but does **not** ship a PEP 561 `py.typed` marker,
because `mypy` currently reports 76 errors against it and most of them are
places where an invariant we enforce at runtime is not expressed in the types.
Shipping the marker would push those errors into the type-checking of everyone
who depends on us.

The count is printed by CI on every run so it can be watched going down. A
pull request that reduces it without changing behaviour is welcome and does
not need to fix everything at once.

## Style

Nothing exotic. Four spaces, 79 columns, standard library only in the runtime
package. Docstrings explain **why** a thing behaves the way it does — the
reasoning is the part a reader cannot reconstruct from the code, and it is
what makes a decision reversible later.

## Licence

MIT. By contributing you agree your contribution is licensed the same way.
