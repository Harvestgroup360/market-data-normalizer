"""Hurdle tests: a return has to beat something, and that is a decision."""
import random
from decimal import Decimal

import pytest

from mdnorm.hurdle import (
    ACT_360,
    ACT_365,
    THIRTY_360,
    ActiveReport,
    DayCount,
    ExcessReport,
    HurdleComparison,
    active_report,
    active_returns,
    excess_report,
    excess_returns,
    hurdle_comparison,
    information_ratio,
    per_period_rate,
    rebase,
    tracking_error,
)

D = Decimal


def dec(*values):
    return [D(str(v)) for v in values]


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


def annual_path(t):
    """A synthetic cash rate: near zero, then up to five per cent, then easing."""
    if t < 132:
        return 0.005
    if t < 152:
        return 0.005 + (t - 132) * 0.00225
    if t < 200:
        return 0.050
    return 0.050 - (t - 200) * 0.000375


def series(n=240, seed=20260918):
    """A cash-plus book: it earns the rate it is financed at, plus a spread."""
    rng = random.Random(seed)
    rets, rates = [], []
    for t in range(n):
        m = annual_path(t) / 12
        rates.append(D(str(round(m, 12))))
        rets.append(D(str(round(m + rng.gauss(0.0025, 0.0085), 10))))
    return rets, rates


# -- turning a quote into a per-period rate --------------------------------

def test_dividing_and_compounding_are_different_numbers():
    """Small per period, one direction, for the whole sample."""
    divided = per_period_rate(D("0.05"), periods_per_year=D(252),
                              compound=False)
    compounded = per_period_rate(D("0.05"), periods_per_year=D(252),
                                 compound=True)
    assert divided > compounded
    assert about(divided, "0.000198413", "0.000000001")
    assert about(compounded, "0.000193631", "0.000000001")


def test_compounding_a_per_period_rate_returns_the_quote():
    r = per_period_rate(D("0.05"), periods_per_year=D(12), compound=True)
    assert about((1 + r) ** 12 - 1, "0.05", "0.0000000001")


def test_a_zero_quote_is_zero_either_way():
    for compound in (True, False):
        assert per_period_rate(D(0), periods_per_year=D(252),
                               compound=compound) == 0


def test_a_negative_quote_still_works_both_ways():
    """Policy rates have been below zero; the module is not surprised by it."""
    assert per_period_rate(D("-0.005"), periods_per_year=D(12),
                           compound=False) < 0
    assert per_period_rate(D("-0.005"), periods_per_year=D(12),
                           compound=True) < 0


def test_a_quote_at_minus_one_is_refused_when_compounding():
    with pytest.raises(ArithmeticError, match="not defined"):
        per_period_rate(D("-1"), periods_per_year=D(12), compound=True)


def test_the_refusal_names_the_alternative():
    with pytest.raises(ArithmeticError, match="compound=False"):
        per_period_rate(D("-1.2"), periods_per_year=D(12), compound=True)


def test_a_non_positive_calendar_is_refused():
    for n in (D(0), D(-12)):
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            per_period_rate(D("0.05"), periods_per_year=n, compound=True)


# -- day-count bases -------------------------------------------------------

def test_a_money_market_quote_is_larger_on_a_365_day_year():
    """The 1.39 per cent of itself that a 360-day quote is missing."""
    r = rebase(D("0.05"), quoted=ACT_360, target=ACT_365)
    assert about(r, "0.050694444", "0.000000001")
    assert r > D("0.05")


def test_rebasing_back_returns_the_quote():
    r = rebase(D("0.05"), quoted=ACT_360, target=ACT_365)
    assert about(rebase(r, quoted=ACT_365, target=ACT_360), "0.05", "1e-30")


def test_the_same_basis_changes_nothing():
    assert rebase(D("0.05"), quoted=ACT_360, target=THIRTY_360) == D("0.05")


def test_a_day_count_needs_a_positive_year():
    with pytest.raises(ValueError, match="positive year"):
        DayCount("nonsense", 0)


# -- subtracting a series rather than a constant ---------------------------

def test_excess_returns_are_taken_period_by_period():
    assert excess_returns(dec("0.01", "0.02"), dec("0.001", "0.004")) == \
        dec("0.009", "0.016")


def test_mismatched_lengths_are_refused_with_both_counts():
    with pytest.raises(ValueError, match="has 3 observations and rates has 2"):
        excess_returns(dec(1, 2, 3), dec(1, 2))


def test_the_refusal_points_at_alignment_rather_than_truncation():
    with pytest.raises(ValueError, match="mdnorm.align"):
        excess_returns(dec(1, 2, 3), dec(1, 2))


def test_an_empty_series_is_refused():
    with pytest.raises(ValueError, match="at least one observation"):
        excess_returns([], [])


def test_a_constant_rate_leaves_the_volatility_alone():
    rets, _ = series(n=60)
    flat = [D("0.001")] * 60
    rep = excess_report(rets, flat, ddof=1)
    assert about(rep.volatility_return, rep.volatility_excess, "1e-30")
    assert not rep.rate_moved


def test_a_moving_rate_does_not():
    rets, rates = series()
    rep = excess_report(rets, rates, ddof=1)
    assert rep.rate_moved
    assert rep.volatility_return != rep.volatility_excess


def test_the_share_credited_to_cash_is_none_on_a_losing_book():
    losing = dec("-0.01", "-0.02", "-0.005")
    rep = excess_report(losing, dec("0.001", "0.001", "0.001"), ddof=1)
    assert rep.share_credited_to_cash is None


# -- the three hurdles -----------------------------------------------------

def test_no_hurdle_is_always_the_flattering_one():
    """A non-negative rate can only make the raw figure the larger."""
    rets, rates = series()
    c = hurdle_comparison(rets, rates, ddof=1)
    assert all(f >= 0 for f in rates)
    assert c.zero_hurdle_gap > 0
    assert c.sharpe_raw > c.sharpe_series


def test_a_constant_rate_makes_the_two_hurdles_agree():
    rets, _ = series(n=120)
    flat = [D("0.002")] * 120
    c = hurdle_comparison(rets, flat, ddof=1)
    assert about(c.sharpe_constant, c.sharpe_series, "1e-30")
    assert about(c.constant_series_gap, 0, "1e-30")


def test_the_constant_series_gap_does_not_have_a_promised_direction():
    """The module refuses to name a direction, and this is why.

    Two samples, one construction, opposite signs. Where the rate is barely
    correlated with the returns the excess series is the more volatile of the
    two and the constant-rate figure comes out higher; where the returns are
    mostly the rate, subtracting it removes variance and the series figure is
    higher instead.
    """
    rets, rates = series()
    weak = hurdle_comparison(rets, rates, ddof=1)

    rng = random.Random(4)
    strong_rets = [r * 3 + D(str(round(rng.gauss(0, 0.0004), 10)))
                   for r in rates]
    strong = hurdle_comparison(strong_rets, rates, ddof=1)

    assert weak.constant_series_gap > 0
    assert strong.constant_series_gap < 0


def test_the_correlation_is_reported_beside_the_gap():
    rets, rates = series()
    c = hurdle_comparison(rets, rates, ddof=1)
    assert c.rate_correlation is not None
    assert -1 <= c.rate_correlation <= 1


def test_a_constant_rate_has_no_correlation_rather_than_zero():
    rets, _ = series(n=60)
    c = hurdle_comparison(rets, [D("0.001")] * 60, ddof=1)
    assert c.rate_correlation is None
    assert c.rate_volatility is None or c.rate_volatility == 0


def test_a_series_too_short_for_ddof_reports_none_rather_than_raising():
    c = hurdle_comparison(dec("0.01"), dec("0.001"), ddof=1)
    assert c.sharpe_raw is None
    assert c.sharpe_series is None
    assert c.zero_hurdle_gap is None
    assert c.constant_series_gap is None


def test_it_agrees_with_the_metrics_module_on_a_constant_rate():
    """Two implementations of the same subtraction.

    Not bit-identical: this module works at forty significant digits and
    `metrics` at thirty-four. Thirty digits of agreement is past anything a
    return series carries.
    """
    from mdnorm.metrics import sharpe_ratio

    rets, _ = series(n=180)
    flat = [D("0.0015")] * 180
    c = hurdle_comparison(rets, flat, ddof=1)
    assert about(c.sharpe_series, sharpe_ratio(rets, risk_free=D("0.0015")),
                 "1e-30")


# -- a benchmark is a hurdle too -------------------------------------------

def test_active_returns_are_taken_period_by_period():
    assert active_returns(dec("0.02", "0.01"), dec("0.015", "0.012")) == \
        dec("0.005", "-0.002")


def test_a_perfect_tracker_has_no_information_ratio_rather_than_an_infinite_one():
    rets, _ = series(n=60)
    assert tracking_error(rets, rets, ddof=1) is None
    assert information_ratio(rets, rets, ddof=1) is None


def test_beating_the_benchmark_gives_a_positive_information_ratio():
    rets, _ = series(n=120)
    bench = [r - D("0.001") for r in rets[:60]] + [r + D("0.0005")
                                                   for r in rets[60:]]
    rep = active_report(rets, bench, ddof=1)
    assert rep.mean_active > 0
    assert rep.information_ratio > 0


def test_the_tracking_error_moves_with_ddof_on_a_short_sample():
    rets, _ = series(n=24)
    bench = [r * D("0.6") for r in rets]
    a = tracking_error(rets, bench, ddof=0)
    b = tracking_error(rets, bench, ddof=1)
    assert b > a


def test_both_legs_are_reported_so_the_benchmark_stays_visible():
    rets, _ = series(n=120)
    bench = [r * D("0.5") for r in rets]
    rep = active_report(rets, bench, ddof=1)
    assert rep.mean_return != rep.mean_benchmark
    assert about(rep.mean_active, rep.mean_return - rep.mean_benchmark, "1e-30")


def test_the_benchmark_share_is_none_on_a_losing_book():
    rep = active_report(dec("-0.01", "-0.02"), dec("0.001", "0.002"), ddof=1)
    assert rep.benchmark_share is None


def test_a_benchmark_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="benchmark has 2"):
        active_returns(dec(1, 2, 3), dec(1, 2))


# -- the worked example ----------------------------------------------------

def test_the_worked_example():
    rets, rates = series()
    c = hurdle_comparison(rets, rates, ddof=1)
    assert c.observations == 240
    assert about(c.sharpe_raw, "0.486326")
    assert about(c.sharpe_constant, "0.260610")
    assert about(c.sharpe_series, "0.257868")
    assert about(c.zero_hurdle_gap, "0.228458")
    assert about(c.constant_series_gap, "0.002742")
    assert about(c.rate_correlation, "0.051507")


def test_the_worked_example_annualises_to_the_headline():
    """1.68 with no hurdle, 0.89 against the rate the book was financed at."""
    from mdnorm.metrics import annualise_sharpe

    rets, rates = series()
    c = hurdle_comparison(rets, rates, ddof=1)
    assert about(annualise_sharpe(c.sharpe_raw, D(12)), "1.6847", "0.0001")
    assert about(annualise_sharpe(c.sharpe_series, D(12)), "0.8933", "0.0001")


def test_almost_half_the_gross_return_was_the_hurdle():
    rets, rates = series()
    rep = excess_report(rets, rates, ddof=1)
    assert about(rep.share_credited_to_cash, "0.464125")
    assert about(rep.mean_return, "0.0039613", "0.0000001")
    assert about(rep.mean_excess, "0.00212276", "0.0000001")


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    rets, rates = series(n=60)
    for obj in (hurdle_comparison(rets, rates, ddof=1),
                excess_report(rets, rates, ddof=1),
                active_report(rets, rates, ddof=1),
                ACT_360):
        with pytest.raises(Exception):
            obj.observations = 9  # type: ignore[misc]


def test_the_reports_are_the_documented_types():
    rets, rates = series(n=60)
    assert isinstance(hurdle_comparison(rets, rates, ddof=1), HurdleComparison)
    assert isinstance(excess_report(rets, rates, ddof=1), ExcessReport)
    assert isinstance(active_report(rets, rates, ddof=1), ActiveReport)
