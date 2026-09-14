"""Exposure tests: what survives once the known factors are subtracted."""
import random
from decimal import Decimal

import pytest

from mdnorm.exposure import (
    ExposureReport,
    FactorLoading,
    alpha_stream,
    dominant_factor,
    factor_regression,
    residuals,
)

D = Decimal


def dec(values):
    return [D(str(round(v, 10))) for v in values]


def panel(n=600, seed=21, betas=(("market", 0.55, 0.010),
                                 ("momentum", 0.80, 0.006)),
          alpha=0.0, noise=0.0035, means=None):
    """A strategy built from known loadings, plus the factors it was built on."""
    rng = random.Random(seed)
    means = means or {}
    factors = {}
    for name, _, sd in betas:
        mu = means.get(name, 0.0)
        factors[name] = [rng.gauss(mu, sd) for _ in range(n)]
    series = []
    for i in range(n):
        value = alpha + rng.gauss(0, noise)
        for name, beta, _ in betas:
            value += beta * factors[name][i]
        series.append(value)
    return dec(series), {k: dec(v) for k, v in factors.items()}


def about(value, target, tol="0.02"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the fit ---------------------------------------------------------------

def test_the_planted_loadings_come_back():
    """Tolerance is two standard errors of the estimate, not a round number.

    With this noise and this many observations a loading lands within about
    0.006 of its planted value two times in three; 0.02 is the band that
    makes the test about the estimator rather than about the seed.
    """
    strategy, factors = panel(n=4000)
    rep = factor_regression(strategy, factors)
    assert about(rep.loading_map["market"], 0.55, "0.02")
    assert about(rep.loading_map["momentum"], 0.80, "0.02")


def test_a_strategy_that_is_one_factor_has_all_of_it_explained():
    rng = random.Random(3)
    mkt = dec([rng.gauss(0.0003, 0.01) for _ in range(400)])
    rep = factor_regression(mkt, {"market": mkt})
    assert about(rep.loading_map["market"], 1, "0.000001")
    assert about(rep.r_squared, 1, "0.000001")
    assert about(rep.alpha, 0, "0.000001")


def test_an_unrelated_factor_gets_a_beta_near_zero():
    strategy, factors = panel()
    rng = random.Random(99)
    factors["noise"] = dec([rng.gauss(0, 0.008) for _ in range(600)])
    rep = factor_regression(strategy, factors)
    assert about(rep.loading_map["noise"], 0, "0.06")


def test_r_squared_is_between_zero_and_one_on_a_real_fit():
    strategy, factors = panel()
    rep = factor_regression(strategy, factors)
    assert D("0.5") < rep.r_squared < D(1)


def test_the_observation_and_factor_counts_are_reported():
    strategy, factors = panel(n=300)
    rep = factor_regression(strategy, factors)
    assert rep.observations == 300
    assert rep.factors == 2
    assert rep.degrees_of_freedom == 297


# -- alpha -----------------------------------------------------------------

def test_a_planted_alpha_is_recovered():
    strategy, factors = panel(n=2000, alpha=0.0004, noise=0.002)
    rep = factor_regression(strategy, factors)
    assert about(rep.alpha, 0.0004, "0.0001")
    assert rep.alpha_t_stat is not None and rep.alpha_t_stat > 3


def test_a_strategy_with_no_alpha_reports_an_insignificant_one():
    strategy, factors = panel(n=2000, alpha=0.0)
    rep = factor_regression(strategy, factors)
    assert rep.alpha_t_stat is not None
    assert abs(rep.alpha_t_stat) < 3


def test_the_alpha_share_is_the_intercept_over_the_mean():
    rep = ExposureReport(observations=100, loadings=(),
                         alpha=D("0.25"), alpha_standard_error=D("0.1"),
                         mean_return=D(1), r_squared=D("0.5"),
                         residual_variance=D("0.01"))
    assert rep.alpha_share == D("0.25")
    assert rep.explained_share == D("0.75")
    assert rep.alpha_t_stat == D("2.5")


def test_a_zero_mean_return_has_no_share():
    """A share of nothing would invite a reader to divide by it."""
    rep = ExposureReport(observations=100, loadings=(), alpha=D("0.25"),
                         alpha_standard_error=None, mean_return=D(0),
                         r_squared=D(0), residual_variance=D(0))
    assert rep.alpha_share is None
    assert rep.explained_share is None
    assert rep.alpha_t_stat is None


# -- the trap in the arithmetic --------------------------------------------

def test_the_residual_mean_is_zero_by_construction():
    """An intercept absorbs exactly the mean, always."""
    strategy, factors = panel(n=800, alpha=0.0006)
    res = residuals(strategy, factors)
    assert len(res) == 800
    assert abs(sum(res, D(0)) / len(res)) < D("1e-20")


def test_the_alpha_stream_carries_the_alpha():
    strategy, factors = panel(n=800, alpha=0.0006)
    rep = factor_regression(strategy, factors)
    stream = alpha_stream(strategy, factors)
    mean = sum(stream, D(0)) / len(stream)
    assert abs(mean - rep.alpha) < D("1e-20")


def test_the_two_series_differ_by_exactly_the_alpha():
    strategy, factors = panel(n=300, alpha=0.0006)
    rep = factor_regression(strategy, factors)
    for a, b in zip(alpha_stream(strategy, factors),
                    residuals(strategy, factors)):
        assert abs((a - b) - rep.alpha) < D("1e-20")


def test_a_sharpe_on_the_alpha_stream_is_not_a_sharpe_on_the_strategy():
    """The whole point: the exposure carries the ratio, not the edge."""
    from mdnorm.metrics import sharpe_ratio

    strategy, factors = panel(n=1500, alpha=0.0, betas=(("market", 1.0, 0.01),),
                              noise=0.002, means={"market": 0.003})
    raw = sharpe_ratio(strategy)
    net = sharpe_ratio(alpha_stream(strategy, factors))
    assert raw is not None and net is not None
    assert raw > D("0.2")              # the exposure looks like a strategy
    assert abs(net) < D("0.06")        # almost none of it survives
    assert abs(net) < raw / 4


# -- contributions and ranking ---------------------------------------------

def test_the_contribution_is_the_beta_times_the_factor_mean():
    strategy, factors = panel(n=900, means={"market": 0.0005})
    rep = factor_regression(strategy, factors)
    market = next(l for l in rep.loadings if l.name == "market")
    mean_factor = sum(factors["market"], D(0)) / 900
    assert abs(market.contribution - market.beta * mean_factor) < D("1e-20")


def test_the_dominant_factor_is_ranked_by_contribution_not_by_beta():
    """A large loading on a factor that went nowhere carries no return."""
    rep = ExposureReport(
        observations=100,
        loadings=(FactorLoading("big beta, flat factor", D(9), D(1), D("0.001")),
                  FactorLoading("small beta, live factor", D("0.2"), D(1),
                                D("0.050"))),
        alpha=D(0), alpha_standard_error=D(1), mean_return=D(1),
        r_squared=D("0.5"), residual_variance=D(0))
    assert dominant_factor(rep).name == "small beta, live factor"


def test_a_negative_contribution_can_dominate():
    rep = ExposureReport(
        observations=100,
        loadings=(FactorLoading("a", D(1), D(1), D("0.01")),
                  FactorLoading("b", D(1), D(1), D("-0.09"))),
        alpha=D(0), alpha_standard_error=D(1), mean_return=D(1),
        r_squared=D("0.5"), residual_variance=D(0))
    assert dominant_factor(rep).name == "b"


def test_no_dominant_factor_when_nothing_contributed():
    rep = ExposureReport(observations=10, loadings=(
        FactorLoading("a", D(1), D(1), D(0)),), alpha=D(0),
        alpha_standard_error=D(1), mean_return=D(1), r_squared=D(0),
        residual_variance=D(0))
    assert dominant_factor(rep) is None


def test_no_factors_means_no_dominant_factor():
    rep = ExposureReport(observations=10, loadings=(), alpha=D(0),
                         alpha_standard_error=None, mean_return=D(1),
                         r_squared=D(0), residual_variance=D(0))
    assert dominant_factor(rep) is None


# -- refusals --------------------------------------------------------------

def test_two_identical_factors_are_refused_by_name():
    """'Singular matrix' tells a caller nothing they can act on."""
    strategy, factors = panel(n=200)
    factors["market_again"] = list(factors["market"])
    with pytest.raises(ValueError, match="singular at"):
        factor_regression(strategy, factors)


def test_a_factor_that_is_the_sum_of_two_others_is_refused():
    strategy, factors = panel(n=200)
    factors["both"] = [a + b for a, b in
                       zip(factors["market"], factors["momentum"])]
    with pytest.raises(ValueError, match="singular at"):
        factor_regression(strategy, factors)


def test_a_constant_factor_is_the_intercept_under_another_name():
    strategy, factors = panel(n=200)
    factors["flat"] = [D("0.001")] * 200
    with pytest.raises(ValueError, match="constant"):
        factor_regression(strategy, factors)


def test_ragged_series_raise_and_name_the_offender():
    strategy, factors = panel(n=200)
    factors["short"] = factors["market"][:100]
    with pytest.raises(ValueError, match="short has 100"):
        factor_regression(strategy, factors)


def test_no_factors_is_refused():
    with pytest.raises(ValueError, match="at least one factor"):
        factor_regression([D(1), D(2), D(3)], {})


def test_too_few_observations_are_refused():
    """A fit with no room left over says nothing about anything."""
    with pytest.raises(ValueError, match="cannot support"):
        factor_regression([D(1), D(2), D(3)],
                          {"a": [D(1), D(3), D(2)], "b": [D(2), D(1), D(5)]})


def test_an_empty_strategy_is_refused():
    with pytest.raises(ValueError, match="no observations"):
        factor_regression([], {"a": []})


# -- the conditions worth reporting ----------------------------------------

def test_a_crowded_regression_is_flagged():
    rep = ExposureReport(observations=40, loadings=tuple(
        FactorLoading(f"f{i}", D(1), D(1), D(0)) for i in range(4)),
        alpha=D(0), alpha_standard_error=D(1), mean_return=D(1),
        r_squared=D("0.9"), residual_variance=D(0))
    assert rep.crowded


def test_a_roomy_regression_is_not_flagged():
    strategy, factors = panel(n=600)
    assert not factor_regression(strategy, factors).crowded


# -- composition -----------------------------------------------------------

def test_the_alpha_stream_goes_back_through_the_rest_of_the_library():
    """The residual is what the claim of alpha is actually about."""
    from mdnorm.metrics import sharpe_ratio
    from mdnorm.windows import WindowKind, sweep, trimmed_starts

    strategy, factors = panel(n=1000, alpha=0.0003)
    stream = alpha_stream(strategy, factors)
    rep = sweep(stream, trimmed_starts(1000, step=50, count=8), sharpe_ratio,
                kind=WindowKind.TRIMMED_START)
    assert rep.trials == 9
    assert rep.full is not None


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (FactorLoading("a", D(1), D(1), D(0)),
                ExposureReport(1, (), D(0), None, D(0), D(0), D(0))):
        with pytest.raises(Exception):
            obj.name = "z"  # type: ignore[misc]
