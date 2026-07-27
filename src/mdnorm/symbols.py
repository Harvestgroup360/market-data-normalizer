"""Canonical symbol normalization.

Venues spell the same instrument in many ways (``BTCUSDT``, ``XBTUSD``,
``btc_usd``). We map everything to a single canonical form ``BASE-QUOTE``.
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


def canonical_symbol(raw: str) -> str:
    """Return the canonical ``BASE-QUOTE`` form of a venue symbol.

    >>> canonical_symbol("BTCUSDT")
    'BTC-USDT'
    >>> canonical_symbol("xbt/usd")
    'BTC-USD'
    >>> canonical_symbol("ETH_EUR")
    'ETH-EUR'
    """
    if not raw or not raw.strip():
        raise ValueError("empty symbol")

    token = _SEP.sub("", raw).upper()

    base, quote = _split(token)
    base = _BASE_ALIASES.get(base, base)
    return f"{base}-{quote}"


def _split(token: str) -> tuple[str, str]:
    for q in _QUOTES:
        if token.endswith(q) and len(token) > len(q):
            return token[: -len(q)], q
    # Fallback: assume a 3-char quote if nothing matched.
    if len(token) > 3:
        return token[:-3], token[-3:]
    raise ValueError(f"cannot parse symbol: {token!r}")
