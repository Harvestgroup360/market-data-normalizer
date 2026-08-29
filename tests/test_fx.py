"""FX tests: direction, staleness, crosses that must be stated, cross terms."""
from decimal import Decimal

import pytest

from mdnorm import (
    AsOfSeries,
    Bar,
    Conversion,
    CurrencyPair,
    FxRates,
    Quote,
    ReturnDecomposition,
    convert_bars,
    convert_series,
    decompose_return,
    read_fx_csv,
)

D = Decimal
NS = 1_000_000_000
MINUTE = 60 * NS
HOUR = 3600 * NS
DAY = 24 * HOUR

EURUSD = CurrencyPair("EUR", "USD")
USDJPY = CurrencyPair("USD", "JPY")


def series(pairs):
    return AsOfSeries([(t, D(str(v))) for t, v in pairs])


def rates(**kw):
    return FxRates({EURUSD: series([(0, "1.10"), (HOUR, "1.20")])}, **kw)


# -- the pair itself --------------------------------------------------------


def test_a_pair_carries_its_direction():
    p = CurrencyPair("eur", " usd ")
    assert (p.base, p.quote) == ("EUR", "USD")
    assert str(p) == "EUR/USD"
    assert p.inverse == CurrencyPair("USD", "EUR")


def test_a_currency_against_itself_is_not_a_pair():
    with pytest.raises(ValueError):
        CurrencyPair("USD", "USD")


def test_parsing_requires_a_separator():
    assert CurrencyPair.parse("EUR/USD") == EURUSD
    assert CurrencyPair.parse("EUR-USD") == EURUSD
    with pytest.raises(ValueError) as exc:
        CurrencyPair.parse("EURUSD")
    assert "three-letter codes" in str(exc.value)


def test_an_empty_code_is_rejected():
    with pytest.raises(ValueError):
        CurrencyPair("", "USD")


# -- staleness --------------------------------------------------------------


def test_max_age_is_required():
    with pytest.raises(TypeError):
        rates().quote(EURUSD, 0)


def test_a_rate_older_than_allowed_is_refused_not_reused():
    r = rates()
    assert r.quote(EURUSD, HOUR - 1, max_age_ns=MINUTE) is None
    q = r.quote(EURUSD, HOUR + 30 * NS, max_age_ns=MINUTE)
    assert q is not None and q.rate == D("1.20")


def test_a_conversion_reports_the_age_it_used():
    c = rates().convert(D("100"), "EUR", "USD", 30 * NS, max_age_ns=HOUR)
    assert c.amount == D("110.00")
    assert c.age_ns == 30 * NS
    assert c.legs[0].as_of_ns == 0


def test_nothing_observed_yet_is_also_a_refusal():
    r = FxRates({EURUSD: series([(HOUR, "1.10")])})
    assert r.quote(EURUSD, 0, max_age_ns=DAY) is None


def test_negative_max_age_is_rejected():
    with pytest.raises(ValueError):
        rates().quote(EURUSD, 0, max_age_ns=-1)


# -- direction --------------------------------------------------------------


def test_the_inverse_pair_is_answered_and_says_so():
    q = rates().quote(CurrencyPair("USD", "EUR"), 0, max_age_ns=HOUR)
    assert q.inverted is True
    assert q.pair == EURUSD                 # the series it came from
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 34
        assert q.applied == D(1) / D("1.10")


def test_inversion_can_be_refused():
    r = rates(allow_inverse=False)
    assert r.quote(CurrencyPair("USD", "EUR"), 0, max_age_ns=HOUR) is None
    assert r.has(CurrencyPair("USD", "EUR")) is False


def test_converting_back_returns_the_amount():
    r = rates()
    usd = r.convert(D("100"), "EUR", "USD", 0, max_age_ns=HOUR)
    back = r.convert(usd.amount, "USD", "EUR", 0, max_age_ns=HOUR)
    assert abs(back.amount - 100) < D("0.0000001")


def test_a_currency_converted_to_itself_is_untouched():
    c = rates().convert(D("7"), "USD", "USD", 0, max_age_ns=HOUR)
    assert c.amount == 7 and c.legs == () and c.rate == 1


# -- crosses ----------------------------------------------------------------


def cross_rates():
    return FxRates({
        EURUSD: series([(0, "1.10")]),
        USDJPY: series([(0, "150")]),
    })


def test_a_cross_must_name_its_vehicle():
    with pytest.raises(ValueError) as exc:
        cross_rates().convert(D("1"), "EUR", "JPY", 0, max_age_ns=HOUR)
    assert "state a vehicle currency" in str(exc.value)


def test_a_stated_cross_multiplies_both_legs():
    c = cross_rates().convert(D("1"), "EUR", "JPY", 0, max_age_ns=HOUR,
                              via="USD")
    assert c.rate == D("165.0")
    assert c.crossed is True
    assert [str(leg.pair) for leg in c.legs] == ["EUR/USD", "USD/JPY"]


def test_a_cross_is_as_stale_as_its_worst_leg():
    r = FxRates({EURUSD: series([(0, "1.10")]),
                 USDJPY: series([(HOUR, "150")])})
    c = r.convert(D("1"), "EUR", "JPY", HOUR, max_age_ns=2 * HOUR, via="USD")
    assert c.legs[0].age_ns == HOUR and c.legs[1].age_ns == 0
    assert c.age_ns == HOUR


def test_a_vehicle_that_is_an_endpoint_is_rejected():
    with pytest.raises(ValueError):
        cross_rates().convert(D("1"), "EUR", "JPY", 0, max_age_ns=HOUR,
                              via="EUR")


def test_an_unreachable_cross_says_which_leg_is_missing():
    r = FxRates({EURUSD: series([(0, "1.10")])})
    with pytest.raises(ValueError) as exc:
        r.convert(D("1"), "EUR", "JPY", 0, max_age_ns=HOUR, via="USD")
    assert "one of the two legs" in str(exc.value)


def test_a_stale_leg_gives_none_rather_than_an_error():
    r = FxRates({EURUSD: series([(0, "1.10")]),
                 USDJPY: series([(0, "150")])})
    assert r.convert(D("1"), "EUR", "JPY", DAY, max_age_ns=HOUR,
                     via="USD") is None


def test_a_present_pair_that_is_stale_gives_none_not_an_error():
    assert rates().convert(D("1"), "EUR", "USD", 10 * DAY,
                           max_age_ns=MINUTE) is None


def test_an_absent_pair_with_no_vehicle_is_an_error():
    with pytest.raises(ValueError):
        rates().convert(D("1"), "GBP", "CHF", 0, max_age_ns=HOUR)


# -- series and bars --------------------------------------------------------


def test_each_observation_is_converted_at_its_own_time():
    prices = series([(0, "100"), (HOUR, "100")])
    out, dropped = convert_series(prices, rates(), base="EUR", to="USD",
                                  max_age_ns=DAY)
    assert dropped == 0
    assert out.at(0)[0] == D("110.00")
    assert out.at(HOUR)[0] == D("120.00")      # the rate moved, not the price


def test_points_with_no_usable_rate_are_dropped_and_counted():
    prices = series([(0, "100"), (10 * DAY, "100")])
    out, dropped = convert_series(prices, rates(), base="EUR", to="USD",
                                  max_age_ns=HOUR)
    assert dropped == 1
    assert len(out) == 1


def bars(n=2):
    return [Bar(start_ns=i * HOUR, interval_ns=HOUR, open=D("10"),
                high=D("12"), low=D("9"), close=D("11"), volume=D("100"),
                trades=5) for i in range(n)]


def test_bar_prices_share_one_rate_so_ohlc_survives():
    """The bar takes the rate as of its end, which is where it is knowable."""
    out, dropped = convert_bars(bars(1), rates(), base="EUR", to="USD",
                                max_age_ns=DAY)
    b = out[0]
    assert dropped == 0
    assert b.open == D("12.00") and b.close == D("13.20")   # 1.20, not 1.10
    assert b.low <= b.open <= b.high and b.low <= b.close <= b.high


def test_volume_is_not_a_currency_amount():
    out, _ = convert_bars(bars(1), rates(), base="EUR", to="USD",
                          max_age_ns=DAY)
    assert out[0].volume == D("100")
    assert out[0].trades == 5


# -- returns ----------------------------------------------------------------


def test_the_identity_holds_exactly():
    d = decompose_return(D("100"), D("110"), D("1.00"), D("1.10"))
    assert d.asset_return == D("0.1")
    assert d.fx_return == D("0.1")
    assert d.total_return == D("0.21")            # not 0.20
    assert d.cross_term == D("0.01")


def test_the_additive_shorthand_understates_a_gain_in_both():
    d = decompose_return(D("100"), D("110"), D("1.00"), D("1.10"))
    assert d.additive == D("0.2")
    assert d.approximation_error == D("0.01")
    assert d.total_return > d.additive


def test_the_shorthand_also_misses_when_the_two_move_apart():
    d = decompose_return(D("100"), D("110"), D("1.00"), D("0.90"))
    assert d.asset_return == D("0.1") and d.fx_return == D("-0.1")
    assert d.total_return == D("-0.01")           # not zero
    assert d.additive == 0


def test_the_cross_term_is_negligible_on_a_small_move():
    d = decompose_return(D("100"), D("100.01"), D("1.0000"), D("1.0001"))
    assert abs(d.approximation_error) < D("0.0000001")


def test_non_positive_inputs_are_rejected():
    for args in ((D("0"), D("1"), D("1"), D("1")),
                 (D("1"), D("0"), D("1"), D("1")),
                 (D("1"), D("1"), D("0"), D("1")),
                 (D("1"), D("1"), D("1"), D("-1"))):
        with pytest.raises(ValueError):
            decompose_return(*args)


# -- construction and CSV ---------------------------------------------------


def test_an_empty_series_is_rejected_at_construction():
    with pytest.raises(ValueError):
        FxRates({EURUSD: AsOfSeries([])})


def test_keys_must_be_pairs():
    with pytest.raises(TypeError):
        FxRates({"EUR/USD": series([(0, "1.1")])})


def test_currencies_are_listed():
    assert cross_rates().currencies() == ("EUR", "JPY", "USD")


def test_a_non_positive_rate_in_a_series_is_an_error():
    r = FxRates({EURUSD: series([(0, "-1.10")])})
    with pytest.raises(ValueError):
        r.quote(EURUSD, 0, max_age_ns=HOUR)


def test_read_fx_csv(tmp_path):
    p = tmp_path / "fx.csv"
    p.write_text("pair,ts_ns,rate\n"
                 "EUR/USD,0,1.10\n"
                 "EUR/USD,3600000000000,1.20\n"
                 "USD/JPY,0,150\n")
    r = read_fx_csv(str(p))
    assert set(map(str, r.pairs)) == {"EUR/USD", "USD/JPY"}
    assert r.quote(EURUSD, HOUR, max_age_ns=HOUR).rate == D("1.20")


def test_read_fx_csv_rejects_a_six_letter_pair(tmp_path):
    p = tmp_path / "fx.csv"
    p.write_text("pair,ts_ns,rate\nEURUSD,0,1.10\n")
    with pytest.raises(ValueError):
        read_fx_csv(str(p))


def test_read_fx_csv_rejects_a_non_positive_rate(tmp_path):
    p = tmp_path / "fx.csv"
    p.write_text("pair,ts_ns,rate\nEUR/USD,0,0\n")
    with pytest.raises(ValueError):
        read_fx_csv(str(p))


def test_read_fx_csv_rejects_an_empty_file(tmp_path):
    p = tmp_path / "fx.csv"
    p.write_text("pair,ts_ns,rate\n")
    with pytest.raises(ValueError):
        read_fx_csv(str(p))


def test_repr_is_informative():
    assert "FxRates" in repr(rates())


def test_the_result_types_stand_on_their_own():
    q = Quote(EURUSD, D("1.10"), 0, 0)
    assert q.applied == D("1.10")                 # not inverted
    c = Conversion(D("110"), "EUR", "USD", (q,))
    assert c.rate == D("1.10") and c.crossed is False and c.age_ns == 0
    d = ReturnDecomposition(D("0"), D("0.5"), D("0.5"))
    assert d.cross_term == 0 and d.approximation_error == 0
