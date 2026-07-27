"""Minimal end-to-end demo.

Run with:  python examples/demo.py
(from the repo root, after `pip install -e .` or with src on PYTHONPATH)
"""
from mdnorm import from_csv_row, from_fix, from_ws_json

csv_row = {
    "symbol": "btc/usd",
    "ts": "2026-01-02T00:00:00Z",
    "price": "42000.5",
    "size": "0.25",
    "side": "buy",
}
ws_msg = {"s": "BTCUSDT", "p": "42000.5", "q": "0.25", "T": 1767312000000, "m": False}
fix_msg = "55=BTC/USD|31=42000.5|32=0.25|54=1|60=20260102-00:00:00"

for label, ev in [
    ("csv", from_csv_row(csv_row, venue="coinbase")),
    ("ws ", from_ws_json(ws_msg, venue="binance")),
    ("fix", from_fix(fix_msg, venue="lmax", sep="|")),
]:
    print(f"[{label}] {ev.symbol} {ev.side.value:<4} "
          f"{ev.price} x {ev.size} @ {ev.ts_ns} ns  ({ev.venue})")
