"""Canonical symbol normalization.

Venues spell the same instrument in many ways (``BTCUSDT``, ``XBTUSD``,
``btc_usd``). Traded pairs are mapped to a single canonical ``BASE-QUOTE``
form. Single-listed instruments — equities, ETFs, indices — have no quote
leg and keep their plain ticker (``AAPL``, ``SPY``, ``BRK.B``).
"""
from __future__ import annotations

import re

# Common venue aliases -> canonical base asset
_BASE_ALIASES = {
    "XBT": "BTC",
}

# Quote currencies we recognise, longest first so "USDT" wins over "USD".
_QUOTES = ("USDT", "USDC", "USD", "EUR", "GBP", "BTC", "ETH", "JPY")

_SEP = re.compile(r"[-_/ ]")

# A single-listed instrument: a short ticker with no quote leg.
_TICKER = re.compile(r"^[A-Z][A-Z.]{0,4}$")


def canonical_symbol(raw: str) -> str:
    """Return the canonical ``BASE-QUOTE`` form of a venue symbol.

    >>> canonical_symbol("BTCUSDT")
    'BTC-USDT'
    >>> canonical_symbol("xbt/usd")
    'BTC-USD'
    >>> canonical_symbol("ETH_EUR")
    'ETH-EUR'

    Instruments without a quote leg keep their ticker:

    >>> canonical_symbol("aapl")
    'AAPL'
    >>> canonical_symbol("SPY")
    'SPY'
    """
    if not raw or not raw.strip():
        raise ValueError("empty symbol")

    token = _SEP.sub("", raw).upper()

    pair = _split_pair(token)
    if pair is not None:
        base, quote = pair
        return f"{_BASE_ALIASES.get(base, base)}-{quote}"

    # No quote leg: a plain ticker (equity, ETF, index).
    if _TICKER.match(token):
        return token

    # Fallback for long tokens: assume a 3-character quote.
    if len(token) > 3:
        return f"{token[:-3]}-{token[-3:]}"

    raise ValueError(f"cannot parse symbol: {token!r}")


def _split_pair(token: str) -> tuple[str, str] | None:
    """Split a token into ``(base, quote)`` if it ends in a known quote."""
    for q in _QUOTES:
        if token.endswith(q) and len(token) > len(q):
            return token[: -len(q)], q
    return None
