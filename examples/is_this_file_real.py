"""Is this price file trades, or numbers computed from trades?

A venue accepts multiples of a tick and nothing else, so raw prints sit on the
grid by construction. A series that does not is a mid, a VWAP, an average
across venues, a back-adjusted history, or an error — and those four look
identical in a spreadsheet.

Run with:  python examples/is_this_file_real.py
"""
from decimal import Decimal as D

from mdnorm import TickBand, TickTable, grid_report, spread_in_ticks

# Sub-dollar in hundredths of a cent, above a dollar in cents.
GRID = TickTable([TickBand(D("0"), D("0.0001")),
                  TickBand(D("1"), D("0.01"))], name="two-band")

prints = [D("42.30"), D("42.31"), D("42.29"), D("42.33")]
mids = [(D("42.30") + D("42.31")) / 2, (D("42.31") + D("42.32")) / 2]
vwaps = [D("42.3047"), D("42.3106")]
adjusted = [p / 3 for p in prints]          # a 3:1 back-adjustment

for label, series in (("raw prints", prints), ("mids", mids),
                      ("vwaps", vwaps), ("3:1 adjusted", adjusted)):
    r = grid_report(series, GRID)
    verdict = "could be real" if r.looks_raw else "derived"
    print(f"{label:<14} {r.on_grid}/{r.total} on the grid   {verdict}")

print()
print("Note the adjusted row: some of its prices land back on the grid by")
print("arithmetic luck. A 2:1 split leaves half a cent-grid history on it, so")
print("`looks_raw` requires every price rather than most of them.")

print()
print("The mid of a one-tick market is half a tick from both sides, so it is")
print("never a price the venue could accept:")
print("  spread in ticks, 42.30 / 42.31 :", spread_in_ticks(D("42.30"), D("42.31"), GRID))
print("  the mid sits at                :", (D("42.30") + D("42.31")) / 2)
print("  on the grid?                   :", GRID.on_grid(D("42.305")))
