"""The property this library is built to protect, demonstrated.

Change the tail of an input and every earlier output must be byte-identical.
A full-sample z-score fails this; a trailing one does not. That is the whole
difference between a feature you could have computed at the time and one that
knows how the series ended.

Run with:  python examples/nothing_looks_forward.py
"""
import random
from decimal import Decimal as D, localcontext

from mdnorm import rolling_zscore

random.seed(11)
prices = [D(str(round(100 + random.gauss(0, 1), 4))) for _ in range(200)]

# The same series, with only the last fifty observations replaced.
altered = list(prices)
for i in range(150, 200):
    altered[i] = D(str(round(500 + random.gauss(0, 20), 4)))


def full_sample_zscore(values):
    """The one-liner that appears in a great deal of published work."""
    with localcontext() as ctx:
        ctx.prec = 34
        n = len(values)
        mean = sum(values, D(0)) / n
        var = sum(((v - mean) ** 2 for v in values), D(0)) / (n - 1)
        std = var.sqrt()
        return [(v - mean) / std for v in values]


trailing_a, trailing_b = rolling_zscore(prices, 60), rolling_zscore(altered, 60)
full_a, full_b = full_sample_zscore(prices), full_sample_zscore(altered)

print("Only observations 150-199 differ between the two inputs.")
print()
print("trailing z-score, first 150 outputs identical :",
      trailing_a[:150] == trailing_b[:150])
print("full-sample z-score, first 150 identical      :",
      full_a[:150] == full_b[:150])
print()
changed = sum(1 for x, y in zip(full_a[:150], full_b[:150]) if x != y)
print(f"The full-sample version changed {changed} of the 150 outputs that")
print("precede the edit. Every one of them now knows something about a part of")
print("the series that had not happened yet.")
print()
print("The first 59 trailing outputs are None: a partial window is not a")
print("result, and the early rows of a feature matrix are supposed to be empty.")
print("  first non-None at index:", next(i for i, v in enumerate(trailing_a) if v is not None))
