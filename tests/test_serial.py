"""Serial correlation tests: what the square root of time assumes."""
import random
from decimal import Decimal

import pytest

from mdnorm.serial import (
    ScalingFactor,
    SerialReport,
    annualise_sharpe,
    long_run_variance,
    scaling_factor,
    serial_report,
    variance_ratio,
)

D = Decimal


def ar1(rho, n=2000, seed=7, sd=0.01, mu=0.0):
    """A series whose lag-k autocorrelation is rho**k."""
    rng = random.Random(seed)
    out = []
    prev = 0.0
    for _ in range(n):
        prev = rho * prev + rng.gauss(0, sd)
        out.append(D(str(round(mu + prev, 10))))
    return out


def geometric(rho, lags=11):
    return [D(str(rho)) ** k for k in range(1, lags + 1)]


def about(value, target, tol="0.0001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the factor ------------------------------------------------------------

def test_no_autocorrelation_is_the_square_root_of_time():
    """The familiar factor is the special case, and it has to come back out."""
    f = scaling_factor([D(0)] * 11, periods=12)
    assert about(f.naive, f.corrected, "1e-30")
    assert about(f.corrected, "3.464102")
    assert about(f.ratio, 1, "1e-30")


def test_a_horizon_of_one_needs_no_autocorrelations():
    f = scaling_factor([], periods=1)
    assert f.naive == 1 and f.corrected == 1
    assert f.lags_needed == 0 and not f.truncated


def test_positive_autocorrelation_shrinks_the_factor():
    f = scaling_factor(geometric("0.3"), periods=12)
    assert about(f.corrected, "2.614806")
    assert f.naive_overstates
    assert about(f.ratio, "0.754829")


def test_negative_autocorrelation_grows_it():
    """The square root of time understates a mean reverting series."""
    f = scaling_factor([D("-0.2")] + [D(0)] * 10, periods=12)
    assert f.corrected > f.naive
    assert not f.naive_overstates


def test_the_factor_falls_as_the_autocorrelation_rises():
    got = [scaling_factor(geometric(r), periods=12).corrected
           for r in ("0", "0.1", "0.3", "0.5", "0.7")]
    assert got == sorted(got, reverse=True)


def test_only_the_lags_the_horizon_weights_are_read():
    """Lag twelve carries a weight of zero at a horizon of twelve."""
    short = scaling_factor(geometric("0.4", lags=11), periods=12)
    padded = scaling_factor(geometric("0.4", lags=40), periods=12)
    assert short.corrected == padded.corrected
    assert padded.lags_used == 11


# -- truncation ------------------------------------------------------------

def test_truncation_is_reported_not_hidden():
    f = scaling_factor([D("0.3")], periods=12)
    assert f.lags_used == 1 and f.lags_needed == 11
    assert f.truncated


def test_truncation_biases_toward_the_naive_answer():
    """The direction matters: truncating flatters a correlated series."""
    full = scaling_factor(geometric("0.3"), periods=12)
    cut = scaling_factor(geometric("0.3", lags=1), periods=12)
    assert full.corrected < cut.corrected < full.naive


def test_a_complete_set_is_not_flagged():
    assert not scaling_factor(geometric("0.3"), periods=12).truncated


# -- refusals --------------------------------------------------------------

def test_a_variance_driven_non_positive_is_refused():
    """Strong negative autocorrelation can break the formula, and then there
    is no factor rather than a small one."""
    with pytest.raises(ArithmeticError, match="not a variance"):
        scaling_factor([D("-0.9")] * 11, periods=12)


def test_an_impossible_autocorrelation_is_refused_by_lag():
    with pytest.raises(ValueError, match="at lag 2 is 1.4"):
        scaling_factor([D("0.1"), D("1.4")], periods=12)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(ValueError, match="periods must be positive"):
        scaling_factor([], periods=0)


# -- annualising a Sharpe --------------------------------------------------

def test_annualising_matches_the_library_factor_under_independence():
    from mdnorm.metrics import annualise_sharpe as plain

    naive = plain(D("0.35"), Decimal(12))
    honest = annualise_sharpe(D("0.35"), periods_per_year=12,
                              autocorrelations=[D(0)] * 11)
    assert about(naive, honest, "1e-10")


def test_a_smoothed_series_loses_a_quarter_of_its_annualised_sharpe():
    """The worked example, and the reason the module exists."""
    naive = D("0.35") * scaling_factor([D(0)] * 11, periods=12).naive
    honest = annualise_sharpe(D("0.35"), periods_per_year=12,
                              autocorrelations=geometric("0.3"))
    assert about(naive, "1.212436")
    assert about(honest, "0.915182")
    assert honest < naive * D("0.76")


def test_the_correction_scales_the_ratio_and_nothing_else():
    a = annualise_sharpe(D("0.35"), periods_per_year=12,
                         autocorrelations=geometric("0.3"))
    b = annualise_sharpe(D("0.70"), periods_per_year=12,
                         autocorrelations=geometric("0.3"))
    assert about(b, a * 2, "1e-20")


# -- the variance ratio ----------------------------------------------------

def test_a_horizon_of_one_is_one_by_definition():
    assert variance_ratio(ar1(0.0), horizon=1) == 1


def test_an_independent_series_lands_near_one():
    assert about(variance_ratio(ar1(0.0, n=4000, seed=1), horizon=12),
                 1, "0.15")


def test_a_trending_series_is_above_one_and_a_reverting_one_below():
    assert variance_ratio(ar1(0.3, n=4000, seed=1), horizon=12) > D("1.3")
    assert variance_ratio(ar1(-0.3, n=4000, seed=1), horizon=12) < D("0.8")


def test_the_ratio_is_biased_toward_one_at_long_horizons():
    """Documented rather than corrected: the bias flatters a dependent series.

    The asymptotic ratio for this process at twelve periods is 1.857. The
    estimator returns materially less, because a twelve-period window fits
    n-11 times and the windows share observations.
    """
    got = variance_ratio(ar1(0.3, n=4000, seed=1), horizon=12)
    assert got < D("1.857")


def test_a_constant_series_raises_rather_than_dividing_by_zero():
    with pytest.raises(ArithmeticError, match="does not vary"):
        variance_ratio([D(1)] * 50, horizon=5)


def test_a_horizon_longer_than_the_series_is_refused():
    with pytest.raises(ValueError, match="at least 51 observations"):
        variance_ratio([D(str(i)) for i in range(20)], horizon=50)


def test_a_non_positive_horizon_is_refused():
    with pytest.raises(ValueError, match="horizon must be positive"):
        variance_ratio(ar1(0.0, n=100), horizon=0)


# -- the long-run variance -------------------------------------------------

def test_zero_lags_is_the_ordinary_variance():
    x = ar1(0.0, n=500, seed=3)
    mu = sum(x, D(0)) / len(x)
    plain = sum(((v - mu) ** 2 for v in x), D(0)) / len(x)
    assert about(long_run_variance(x, max_lag=0), plain, "1e-20")


def test_positive_dependence_raises_the_long_run_variance():
    x = ar1(0.4, n=3000, seed=5)
    assert long_run_variance(x, max_lag=20) > long_run_variance(x, max_lag=0)


def test_negative_dependence_lowers_it():
    x = ar1(-0.4, n=3000, seed=5)
    assert long_run_variance(x, max_lag=20) < long_run_variance(x, max_lag=0)


def test_the_bartlett_weights_keep_it_positive():
    """A raw autocovariance sum can go negative, which is not a variance."""
    for seed in range(8):
        x = ar1(-0.45, n=600, seed=seed)
        assert long_run_variance(x, max_lag=15) > 0


def test_a_negative_lag_count_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        long_run_variance(ar1(0.0, n=100), max_lag=-1)


# -- the report ------------------------------------------------------------

def test_the_report_carries_both_annualised_figures():
    rep = serial_report(ar1(0.35, n=600, seed=11), periods_per_year=12,
                        max_lag=11)
    assert rep.observations == 600
    assert rep.sharpe is not None
    assert rep.naive_annualised is not None
    assert rep.factor.naive_overstates
    assert rep.corrected_annualised < rep.naive_annualised
    assert rep.difference > 0


def test_the_difference_goes_negative_on_a_reverting_series():
    """The name promises no direction, and this is why."""
    rep = serial_report(ar1(-0.35, n=600, seed=11), periods_per_year=12,
                        max_lag=11)
    assert not rep.factor.naive_overstates
    assert rep.difference < 0


def test_the_report_recovers_the_planted_first_order_autocorrelation():
    rep = serial_report(ar1(0.4, n=4000, seed=2), periods_per_year=12,
                        max_lag=11)
    assert about(rep.first_order, "0.4", "0.05")
    assert len(rep.autocorrelations) == 11


def test_a_thin_deepest_lag_is_flagged_not_corrected():
    rep = serial_report(ar1(0.2, n=40, seed=4), periods_per_year=12,
                        max_lag=11)
    assert rep.thin_lags
    assert not serial_report(ar1(0.2, n=600, seed=4), periods_per_year=12,
                             max_lag=11).thin_lags


def test_a_constant_series_raises_rather_than_reporting_zero_correlation():
    """A flat series has no autocorrelation to estimate, and saying it has
    none would read as evidence of independence rather than absence of data."""
    with pytest.raises(ValueError, match="constant series has no"):
        serial_report([D("0.01")] * 200, periods_per_year=12, max_lag=5)


def test_a_report_with_no_sharpe_carries_no_annualised_figures():
    rep = SerialReport(observations=100, periods_per_year=12,
                       autocorrelations=(D(0),),
                       factor=scaling_factor([D(0)] * 11, periods=12),
                       sharpe=None, variance_ratio=None)
    assert rep.naive_annualised is None
    assert rep.corrected_annualised is None
    assert rep.difference is None


def test_the_variance_ratio_is_none_when_the_series_is_too_short_for_it():
    rep = serial_report(ar1(0.05, n=15, seed=6), periods_per_year=20,
                        max_lag=2)
    assert rep.variance_ratio is None
    assert rep.factor.truncated


def test_annualising_a_short_series_to_a_daily_calendar_can_refuse_outright():
    """Twenty observations carry no information about 251 lags, and the
    weighted sum of five noisy ones can make the formula meaningless. It
    raises instead of returning a plausible-looking factor."""
    with pytest.raises(ArithmeticError, match="not a variance"):
        serial_report(ar1(0.2, n=20, seed=6), periods_per_year=252,
                      max_lag=5)


def test_the_report_refuses_a_non_positive_calendar():
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        serial_report(ar1(0.0, n=100), periods_per_year=0, max_lag=5)


def test_the_report_refuses_a_non_positive_lag_count():
    with pytest.raises(ValueError, match="max_lag must be positive"):
        serial_report(ar1(0.0, n=100), periods_per_year=12, max_lag=0)


def test_a_series_too_short_for_the_lags_is_refused():
    with pytest.raises(ValueError, match="at least 13 observations"):
        serial_report([D(str(i)) for i in range(10)], periods_per_year=12,
                      max_lag=11)


# -- composition -----------------------------------------------------------

def test_it_agrees_with_the_effective_sample_size_on_direction():
    """Two modules, one fact: dependence costs you information.

    `independence` counts observations, this counts what a square root is
    worth. A series that loses observations to autocorrelation must also lose
    scaling factor, and the signs have to move together.
    """
    from mdnorm.independence import effective_sample_size_series

    x = ar1(0.4, n=2000, seed=9)
    sample = effective_sample_size_series(x, max_lag=11)
    rep = serial_report(x, periods_per_year=12, max_lag=11)
    assert sample.effective < sample.nominal
    assert rep.factor.naive_overstates


def test_the_first_order_agrees_with_the_smoothing_model():
    """A smoothed mark is the concrete case this module is built for."""
    from mdnorm.staleness import smoothing_bias

    x = ar1(0.3, n=3000, seed=12)
    rep = serial_report(x, periods_per_year=12, max_lag=11)
    bias = smoothing_bias(x)
    assert bias.autocorrelation > 0
    assert rep.factor.naive_overstates


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    f = scaling_factor([D(0)] * 11, periods=12)
    rep = serial_report(ar1(0.1, n=200, seed=1), periods_per_year=12,
                        max_lag=11)
    for obj in (f, rep):
        with pytest.raises(Exception):
            obj.periods = 99  # type: ignore[misc]
