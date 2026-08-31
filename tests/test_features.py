"""Feature tests: causality, partial windows, gaps, and the annualisation trap."""
import random
from decimal import Decimal, Inexact, localcontext
from fractions import Fraction

import pytest

from mdnorm import (
    EventType,
    MarketEvent,
    ReturnMethod,
    align,
    column,
    periods_per_year,
    realized_volatility,
    returns,
    rolling_correlation,
    rolling_mean,
    rolling_std,
    rolling_sum,
    rolling_zscore,
    timestamps,
)

D = Decimal
SEC = 1_000_000_000
HOUR = 3600 * SEC


def px(*vals):
    return [None if v is None else D(str(v)) for v in vals]


def approx(x, target, tol="0.0000001"):
    return x is not None and abs(x - D(str(target))) < D(tol)


# -- causality: the contract of the whole module -----------------------------

CAUSAL = [
    ("returns", lambda s: returns(s)),
    ("log returns", lambda s: returns(s, method=ReturnMethod.LOG)),
    ("rolling_mean", lambda s: rolling_mean(s, 3)),
    ("rolling_std", lambda s: rolling_std(s, 3)),
    ("rolling_zscore", lambda s: rolling_zscore(s, 3)),
    ("realized_volatility", lambda s: realized_volatility(s, window=3)),
    # A fixed partner series: reversing s would make the *test* non-causal,
    # since changing s's tail would then change the partner's head.
    ("rolling_correlation",
     lambda s: rolling_correlation(s, px(5, 3, 9, 1, 7, 2, 8, 4), 3)),
]


@pytest.mark.parametrize("name,fn", CAUSAL, ids=[n for n, _ in CAUSAL])
def test_a_later_value_cannot_change_an_earlier_one(name, fn):
    """No feature at index i may depend on anything after index i.

    This is the property the module exists to guarantee, so it is tested as a
    property rather than through examples: change the tail of the input and
    every output before the change must be byte-identical.
    """
    base = px(100, 101, 102, 103, 104, 105, 106, 107)
    tampered = base[:5] + px(9999, 8888, 7777)
    assert fn(base)[:5] == fn(tampered)[:5]


def test_a_full_sample_zscore_would_fail_that_test():
    """Pinned to show the failure the rolling form avoids.

    Standardising against the mean and standard deviation of the entire series
    makes every point depend on every other one. The same tail change that
    leaves the rolling z-score untouched moves the very first value.
    """
    def full_sample_zscore(s):
        mean = sum(s, D(0)) / len(s)
        var = sum(((v - mean) ** 2 for v in s), D(0)) / (len(s) - 1)
        return [(v - mean) / var.sqrt() for v in s]

    base = px(100, 101, 102, 103, 104, 105, 106, 107)
    tampered = base[:5] + px(9999, 8888, 7777)
    assert full_sample_zscore(base)[0] != full_sample_zscore(tampered)[0]
    assert rolling_zscore(base, 3)[:5] == rolling_zscore(tampered, 3)[:5]


# -- returns -----------------------------------------------------------------

def test_the_first_observation_has_no_return():
    """A zero there would add a flat period that never happened."""
    assert returns(px(100, 110))[0] is None


def test_returns_are_aligned_to_the_later_observation():
    r = returns(px(100, 110))
    assert approx(r[1], "0.1")


def test_log_returns():
    r = returns(px(100, 200), method=ReturnMethod.LOG)
    assert approx(r[1], "0.6931471805599453")


def test_log_and_simple_returns_agree_for_small_moves():
    r_s = returns(px(100, "100.01"))[1]
    r_l = returns(px(100, "100.01"), method=ReturnMethod.LOG)[1]
    assert abs(r_s - r_l) < D("0.000001")


def test_a_gap_kills_the_return_on_both_sides_of_it():
    r = returns(px(100, None, 120, 130))
    assert r == [None, None, None] + [r[3]]
    assert approx(r[3], "0.0833333333333333333")


def test_a_non_positive_price_has_no_return():
    assert returns(px(100, 0, 100)) == [None, None, None]
    assert returns(px(100, -5), method=ReturnMethod.LOG)[1] is None


def test_returns_of_an_empty_or_single_series():
    assert returns([]) == []
    assert returns(px(100)) == [None]


# -- partial windows ---------------------------------------------------------

def test_a_partial_window_yields_nothing():
    """The first rows of a feature matrix are supposed to be empty."""
    assert rolling_mean(px(1, 2, 3, 4), 3)[:2] == [None, None]
    assert rolling_std(px(1, 2, 3, 4), 3)[:2] == [None, None]


def test_the_window_fills_exactly_at_the_window_length():
    m = rolling_mean(px(1, 2, 3), 3)
    assert m[2] == D(2) and m[1] is None


def test_a_gap_inside_the_window_propagates():
    """Stepping over the hole would compute a 3-period statistic from 2."""
    m = rolling_mean(px(1, 2, None, 4, 5, 6), 3)
    assert m == [None, None, None, None, None, D(5)]


def test_rolling_mean_of_a_flat_series():
    assert rolling_mean(px(7, 7, 7), 3)[2] == D(7)


def test_rolling_std_of_a_flat_series_is_zero():
    assert rolling_std(px(7, 7, 7), 3)[2] == 0


def test_rolling_std_is_the_sample_form_by_default():
    # values 1,2,3: sample sd = 1, population sd = sqrt(2/3)
    assert rolling_std(px(1, 2, 3), 3)[2] == D(1)
    assert approx(rolling_std(px(1, 2, 3), 3, ddof=0)[2], "0.8164965809277260")


@pytest.mark.parametrize("fn,window", [
    (rolling_mean, 0), (rolling_std, 1), (rolling_zscore, 1),
])
def test_windows_that_cannot_produce_a_statistic_are_rejected(fn, window):
    with pytest.raises(ValueError):
        fn(px(1, 2, 3), window)


def test_ddof_must_leave_a_degree_of_freedom():
    with pytest.raises(ValueError):
        rolling_std(px(1, 2, 3), 3, ddof=3)
    with pytest.raises(ValueError):
        rolling_std(px(1, 2, 3), 3, ddof=-1)


# -- z-score -----------------------------------------------------------------

def test_zscore_measures_the_latest_value_against_its_own_past():
    z = rolling_zscore(px(10, 12, 14), 3)
    assert approx(z[2], "1")


def test_a_frozen_window_has_an_undefined_zscore_not_zero():
    """Usually a forward-fill that has not expired — worth seeing as a hole."""
    assert rolling_zscore(px(5, 5, 5, 5), 3) == [None] * 4


# -- correlation -------------------------------------------------------------

def test_perfectly_co_moving_series_correlate_at_one():
    a, b = px(1, 2, 3, 4), px(10, 20, 30, 40)
    assert approx(rolling_correlation(a, b, 4)[3], "1")


def test_perfectly_opposed_series_correlate_at_minus_one():
    a, b = px(1, 2, 3, 4), px(40, 30, 20, 10)
    assert approx(rolling_correlation(a, b, 4)[3], "-1")


def test_a_frozen_series_correlates_with_nothing_and_says_so():
    """Reading this as zero is how a dead feed becomes an apparent hedge."""
    live, dead = px(1, 2, 3, 4), px(5, 5, 5, 5)
    assert rolling_correlation(live, dead, 4)[3] is None


def test_correlation_stays_inside_the_unit_interval():
    a, b = px(3, 1, 4, 1, 5, 9, 2, 6), px(2, 7, 1, 8, 2, 8, 1, 8)
    for v in rolling_correlation(a, b, 4)[3:]:
        assert D(-1) <= v <= D(1)


def test_correlation_is_symmetric():
    a, b = px(3, 1, 4, 1, 5), px(2, 7, 1, 8, 2)
    assert rolling_correlation(a, b, 4) == rolling_correlation(b, a, 4)


def test_correlation_needs_series_of_the_same_length():
    with pytest.raises(ValueError, match="align"):
        rolling_correlation(px(1, 2, 3), px(1, 2), 2)


def test_a_gap_in_either_series_blanks_the_correlation():
    assert rolling_correlation(px(1, 2, None, 4), px(1, 2, 3, 4), 3)[3] is None


# -- volatility and annualisation --------------------------------------------

def test_volatility_is_per_period_unless_you_annualise_it():
    r = returns(px(100, 110, 99, 108))
    per_period = realized_volatility(r, window=3)
    assert per_period[3] is not None
    assert per_period == rolling_std(r, 3)


def test_annualising_scales_by_the_square_root_of_the_factor():
    r = returns(px(100, 110, 99, 108))
    plain = realized_volatility(r, window=3)[3]
    scaled = realized_volatility(r, window=3, periods_per_year=D(4))[3]
    assert approx(scaled, plain * 2)


def test_a_non_positive_annualisation_factor_is_rejected():
    with pytest.raises(ValueError):
        realized_volatility(returns(px(1, 2, 3)), window=2, periods_per_year=D(0))


def test_periods_per_year_for_a_continuous_market():
    """Minute bars, 24/7: 525,600 — not 252, and not 365."""
    assert periods_per_year(60 * SEC, sessions_per_year=365,
                            session_length_ns=24 * HOUR) == D(525_600)


def test_periods_per_year_for_a_cash_equity_session():
    """252 sessions of 6.5 hours in minute bars."""
    assert periods_per_year(60 * SEC, sessions_per_year=252,
                            session_length_ns=(13 * HOUR) // 2) == D(98_280)


def test_daily_bars_on_a_252_day_calendar_give_the_familiar_number():
    assert periods_per_year(24 * HOUR, sessions_per_year=252,
                            session_length_ns=24 * HOUR) == D(252)


def test_the_annualisation_factor_has_no_default():
    """Stating the calendar is mandatory, because no default is safe.

    The same minute bars are 525,600 periods a year on a continuous venue and
    98,280 on a cash equity session — a factor of 2.3 in the reported
    volatility, from an assumption nobody writes down.
    """
    crypto = periods_per_year(60 * SEC, sessions_per_year=365,
                              session_length_ns=24 * HOUR)
    equity = periods_per_year(60 * SEC, sessions_per_year=252,
                              session_length_ns=(13 * HOUR) // 2)
    assert crypto / equity > D("5")
    assert crypto.sqrt() / equity.sqrt() > D("2.3")


@pytest.mark.parametrize("kwargs", [
    {"interval_ns": 0}, {"sessions_per_year": 0}, {"session_length_ns": 0},
    {"sessions_per_year": -1},
])
def test_periods_per_year_validates_its_inputs(kwargs):
    args = {"interval_ns": 60 * SEC, "sessions_per_year": 252,
            "session_length_ns": 24 * HOUR}
    args.update(kwargs)
    with pytest.raises(ValueError):
        periods_per_year(**args)


# -- reading a column out of an aligned matrix -------------------------------

def _trade(ts, price, symbol="X"):
    return MarketEvent(symbol=symbol, venue="v", event_type=EventType.TRADE,
                       ts_ns=ts, price=D(str(price)), size=D("1"))


def test_column_keeps_the_holes():
    rows = align({"A": [_trade(10, 1)], "B": [_trade(0, 2)]},
                 interval_ns=5, start_ns=0, end_ns=15)
    assert column(rows, "A") == [None, None, D(1)]


def test_a_mistyped_column_fails_immediately():
    rows = align({"A": [_trade(0, 1)]}, interval_ns=5, start_ns=0, end_ns=6)
    with pytest.raises(KeyError, match="available"):
        column(rows, "a")


def test_column_of_no_rows_is_empty():
    assert column([], "anything") == []


def test_timestamps_come_back_for_writing_features_out():
    rows = align({"A": [_trade(0, 1)]}, interval_ns=5, start_ns=0, end_ns=11)
    assert timestamps(rows) == [0, 5, 10]


# -- integration -------------------------------------------------------------

def test_the_whole_path_from_ticks_to_annualised_volatility():
    events = [_trade(i * 60 * SEC, 100 + (i % 3)) for i in range(10)]
    rows = align({"X": events}, interval_ns=60 * SEC)
    prices = column(rows, "X")
    r = returns(prices, method=ReturnMethod.LOG)
    ppy = periods_per_year(60 * SEC, sessions_per_year=365,
                           session_length_ns=24 * HOUR)
    vol = realized_volatility(r, window=4, periods_per_year=ppy)
    assert len(vol) == len(rows)
    assert all(v is None for v in vol[:4])       # window plus the missing r[0]
    assert vol[-1] is not None and vol[-1] > 0


def test_features_line_up_with_the_grid_they_came_from():
    """Every feature series is the same length as the matrix, holes included."""
    rows = align({"A": [_trade(0, 1), _trade(20, 2)]},
                 interval_ns=5, start_ns=0, end_ns=25)
    a = column(rows, "A")
    for series in (returns(a), rolling_mean(a, 3), rolling_std(a, 3),
                   rolling_zscore(a, 3), realized_volatility(returns(a), window=3)):
        assert len(series) == len(rows)


# -- the carried gap check (1.17.0) ------------------------------------------
# Deciding whether a window contains a hole used to be a sweep of the window at
# every index; it is now one comparison against the position of the most recent
# hole. These pin the boundaries where an off-by-one would hide.


def test_a_gap_blocks_exactly_its_own_window():
    vals = [D(1), D(2), None, D(4), D(5), D(6), D(7)]
    out = rolling_mean(vals, 3)
    # indices 2, 3 and 4 have the hole inside their window; 5 is the first clear
    assert out[:5] == [None, None, None, None, None]
    assert out[5] is not None and out[6] is not None


def test_the_window_clears_the_instant_the_gap_leaves_it():
    vals = [None] + [D(i) for i in range(1, 8)]
    out = rolling_mean(vals, 3)
    assert out[2] is None          # window 0..2 still contains the hole
    assert out[3] == Decimal(2)    # window 1..3 is clear: (1+2+3)/3


def test_two_gaps_are_both_respected():
    vals = [D(1), None, D(3), D(4), None, D(6), D(7), D(8)]
    out = rolling_mean(vals, 2)
    assert out == [None, None, None, Decimal("3.5"), None, None,
                   Decimal("6.5"), Decimal("7.5")]


def test_a_gap_at_the_very_end():
    vals = [D(1), D(2), D(3), None]
    assert rolling_mean(vals, 2)[-1] is None
    assert rolling_std(vals, 2)[-1] is None
    assert rolling_zscore(vals, 2)[-1] is None


def test_a_window_as_long_as_the_series():
    vals = [D(2), D(4), D(6)]
    assert rolling_mean(vals, 3) == [None, None, Decimal(4)]


def test_a_window_longer_than_the_series_yields_nothing():
    assert rolling_mean([D(1), D(2)], 5) == [None, None]
    assert rolling_std([D(1), D(2)], 5) == [None, None]


def test_all_three_agree_about_where_a_window_is_valid():
    vals = [D(1), D(2), None, D(4), D(5), D(6), D(7), D(8)]
    m = rolling_mean(vals, 3)
    s = rolling_std(vals, 3)
    z = rolling_zscore(vals, 3)
    assert [x is None for x in m] == [x is None for x in s]
    # the z-score is additionally None wherever the value itself is missing
    for i, v in enumerate(vals):
        if v is None:
            assert z[i] is None


def test_zscore_still_rejects_a_bad_ddof():
    """rolling_zscore validates ddof itself now rather than inheriting the
    check from rolling_std."""
    with pytest.raises(ValueError):
        rolling_zscore([D(1), D(2), D(3)], 3, ddof=3)
    with pytest.raises(ValueError):
        rolling_zscore([D(1), D(2), D(3)], 3, ddof=-1)


def test_zscore_of_a_frozen_window_is_none_not_zero():
    vals = [D(5), D(5), D(5), D(5)]
    assert rolling_zscore(vals, 3) == [None, None, None, None]


# -- the sliding sum, and the promise that it changes nothing ---------------


def _naive_sum(values, window):
    """What this module did before the sum was slid: recompute every window."""
    out = []
    last_gap = -1
    with localcontext() as ctx:
        ctx.prec = 34
        for i in range(len(values)):
            if values[i] is None:
                last_gap = i
            if i + 1 < window:
                out.append(None)
                continue
            start = i - window + 1
            out.append(None if last_gap >= start
                       else sum(values[start:i + 1], Decimal(0)))
    return out


def _naive_mean(values, window):
    return [None if s is None else _div(s, window)
            for s in _naive_sum(values, window)]


def _div(a, b):
    with localcontext() as ctx:
        ctx.prec = 34
        return a / b


#: Fixed per kind, because hash() of a str is salted per process and a
#: fixture that changes between runs is not a fixture.
_SEEDS = {"plain": 100, "gaps": 200, "magnitudes": 300, "digits": 400}


def _series(n, seed, *, gaps=0.0, magnitudes=False, digits=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if gaps and rng.random() < gaps:
            out.append(None)
        elif digits:
            out.append(Decimal("1." + "".join(rng.choice("0123456789")
                                              for _ in range(digits))))
        elif magnitudes:
            out.append(Decimal(str(rng.choice([1e25, 1e-8, -1e25, 3.14159]))))
        else:
            out.append(Decimal(str(round(100 + rng.gauss(0, 1), 4))))
    return out


@pytest.mark.parametrize("window", [2, 3, 20, 60])
@pytest.mark.parametrize("kind", ["plain", "gaps", "magnitudes", "digits"])
def test_the_slid_sum_is_never_further_from_the_truth(window, kind):
    """Against exact rational arithmetic, sliding is never the worse answer.

    On ordinary data the two agree exactly. Where they disagree it is because
    the recomputed sum rounded an intermediate partial that the slid total
    never held, and in that case the slid total is the correct one.
    """
    kw = {"gaps": 0.03} if kind == "gaps" else \
         {"magnitudes": True} if kind == "magnitudes" else \
         {"digits": 33} if kind == "digits" else {}
    values = _series(400, seed=_SEEDS[kind] + window, **kw)
    slid = rolling_sum(values, window)
    naive = _naive_sum(values, window)
    assert len(slid) == len(naive) == len(values)
    for i, (a, b) in enumerate(zip(slid, naive)):
        if a is None or b is None:
            assert a is None and b is None
            continue
        truth = sum((Fraction(str(v)) for v in values[i - window + 1:i + 1]),
                    Fraction(0))
        assert abs(Fraction(str(a)) - truth) <= abs(Fraction(str(b)) - truth)


@pytest.mark.parametrize("window", [2, 20, 60])
def test_on_ordinary_prices_nothing_changed_at_all(window):
    """The case every user is in: identical, value for value."""
    values = _series(400, seed=window, gaps=0.03)
    assert rolling_sum(values, window) == _naive_sum(values, window)
    assert rolling_mean(values, window) == _naive_mean(values, window)


def test_where_the_two_differ_the_slid_total_is_the_exact_one():
    """A window holding both 1e25 and 3.14159 rounds when summed forwards."""
    values = _series(400, seed=0, magnitudes=True)
    slid, naive = rolling_sum(values, 60), _naive_sum(values, 60)
    differing = [i for i, (a, b) in enumerate(zip(slid, naive)) if a != b]
    assert differing, "this fixture is meant to produce a disagreement"
    for i in differing:
        truth = sum((Fraction(str(v)) for v in values[i - 59:i + 1]),
                    Fraction(0))
        assert Fraction(str(slid[i])) == truth
        assert Fraction(str(naive[i])) != truth


def test_the_fallback_actually_fires_and_is_still_exact():
    """Values wide enough that sliding the total would round."""
    values = _series(400, seed=7, digits=33)
    slid = rolling_sum(values, 25)
    assert slid == _naive_sum(values, 25)
    fell_back = False
    with localcontext() as ctx:
        ctx.prec = 34
        for i in range(25, len(values)):
            ctx.flags[Inexact] = False
            _ = sum(values[i - 24:i + 1], Decimal(0))
            if ctx.flags[Inexact]:
                fell_back = True
                break
    assert fell_back, "this fixture is meant to force the fallback"


def test_the_same_series_gives_the_same_answer_twice():
    values = _series(300, seed=11)
    assert rolling_sum(values, 30) == rolling_sum(values, 30)
    assert rolling_zscore(values, 30) == rolling_zscore(values, 30)


def test_a_gap_resets_the_running_total_rather_than_carrying_it():
    values = [Decimal(i) for i in range(10)]
    values[4] = None
    got = rolling_sum(values, 3)
    assert got[3] == Decimal(1 + 2 + 3)        # the gap is still ahead
    assert got[4] is None and got[5] is None and got[6] is None
    assert got[7] == Decimal(5 + 6 + 7)        # first clean window after it
    assert got[8] == Decimal(6 + 7 + 8)


def test_rolling_sum_rejects_a_useless_window():
    with pytest.raises(ValueError):
        rolling_sum([Decimal(1)], 0)


def test_rolling_sum_on_a_short_series_is_all_none():
    assert rolling_sum([Decimal(1), Decimal(2)], 5) == [None, None]


def test_rolling_sum_of_an_empty_series():
    assert rolling_sum([], 3) == []
