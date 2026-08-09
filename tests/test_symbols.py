"""Canonical symbol tests: traded pairs and single-listed tickers."""
import pytest

from mdnorm import canonical_symbol


@pytest.mark.parametrize("raw,expected", [
    ("BTCUSDT", "BTC-USDT"),
    ("btcusdt", "BTC-USDT"),
    ("BTC-USD", "BTC-USD"),
    ("btc/usd", "BTC-USD"),
    ("ETH_EUR", "ETH-EUR"),
    ("XBTUSD", "BTC-USD"),          # venue alias
    ("SOL USDC", "SOL-USDC"),
    ("ADATRY", "ADA-TRY"),          # unknown quote, 3-char fallback
    ("MATICBRL", "MATIC-BRL"),
])
def test_pairs(raw, expected):
    assert canonical_symbol(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("AAPL", "AAPL"),
    ("aapl", "AAPL"),
    ("SPY", "SPY"),
    ("GOOGL", "GOOGL"),
    ("brk.b", "BRK.B"),
    ("F", "F"),                     # single-letter ticker
])
def test_single_listed_tickers(raw, expected):
    """Equities and ETFs have no quote leg and keep their ticker."""
    assert canonical_symbol(raw) == expected


def test_pair_wins_over_ticker():
    """A recognised quote leg is still treated as a pair, not a ticker."""
    assert canonical_symbol("ETHBTC") == "ETH-BTC"


@pytest.mark.parametrize("bad", ["", "   ", "-"])
def test_rejects_empty(bad):
    with pytest.raises(ValueError):
        canonical_symbol(bad)
