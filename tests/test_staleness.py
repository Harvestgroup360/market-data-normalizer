"""Staleness tests: flat stretches, and what smoothing does to a Sharpe."""
import random
from decimal import Decimal

import pytest

from mdnorm.staleness import (
    Run,
    SmoothingBias,
    StalenessReport,
    runs,
    smoothing_bias,
    staleness_report,
)

D = Decimal


def series(text):
    return [D(x) for x in text.split()]


def smoothed(a, n=6000, seed=17):
    """``a * true[t] + (1-a) * true[t-1]`` — a partly stale mark."""
    rng = random.Random(seed)
    true = [rng.gauss(0, 1) for _ in range(n)]
    b = 1 - a
    return [D(str(round(a * true[i] + b * true[i - 1], 10)))
            for i in range(1, n)]


# -- runs ------------------------------------------------------------------

def test_runs_finds_every_maximal_flat_stretch():
    got = list(runs(series("1 1 1 2 3 3 4 4 4 4 5"), min_run=2))
    assert [(r.start, r.length, r.value) for r in got] == [
        (0, 3, D(1)), (4, 2, D(3)), (6, 4, D(4))]


def test_min_run_selects_which_stretches_are_worth_reporting():
    values = series("1 1 1 2 3 3 4 4 4 4 5")
    assert len(list(runs(values, min_run=3))) == 2
    assert len(list(runs(values, min_run=4))) == 1
    assert len(list(runs(values, min_run=5))) == 0


def test_a_min_run_of_one_reports_every_value_including_singletons():
    got = list(runs(series("1 2 2 3"), min_run=1))
    assert [r.length for r in got] == [1, 2, 1]


def test_repeats_counts_the_observations_that_carried_no_news():
    assert Run(0, 4, D(1)).repeats == 3
    assert Run(0, 1, D(1)).repeats == 0


def test_a_run_covers_at_least_one_observation():
    with pytest.raises(ValueError, match="at least one"):
        Run(0, 0, D(1))


def test_min_run_is_validated():
    with pytest.raises(ValueError, match="at least 1"):
        list(runs(series("1 1"), min_run=0))


def test_an_empty_series_has_no_runs():
    assert list(runs([], min_run=2)) == []


def test_a_series_that_never_repeats_has_no_runs():
    assert list(runs(series("1 2 3 4 5"), min_run=2)) == []


def test_a_series_that_never_moves_is_one_long_run():
    got = list(runs([D(7)] * 50, min_run=2))
    assert len(got) == 1 and got[0].length == 50


# -- the report ------------------------------------------------------------

def test_the_report_counts_transitions_not_observations():
    r = staleness_report(series("1 1 1 2 3 3 4 4 4 4 5"), min_run=3)
    assert r.observations == 11
    assert r.unchanged == 6              # ten transitions, six of them flat
    assert r.unchanged_share == D(6) / 10


def test_runs_and_in_runs_follow_min_run():
    values = series("1 1 1 2 3 3 4 4 4 4 5")
    loose = staleness_report(values, min_run=2)
    strict = staleness_report(values, min_run=4)
    assert loose.runs == 3 and loose.in_runs == 9
    assert strict.runs == 1 and strict.in_runs == 4
    assert loose.unchanged == strict.unchanged     # unaffected by min_run


def test_the_longest_run_is_reported():
    assert staleness_report(series("1 1 2 2 2 2 3"), min_run=2).longest_run == 4


def test_a_moving_series_reports_nothing_stale():
    r = staleness_report(series("1 2 3 4 5"), min_run=2)
    assert r.unchanged == 0
    assert r.unchanged_share == 0
    assert r.runs == 0 and r.longest_run == 0
    assert r.in_runs_share == 0


def test_a_frozen_series_is_entirely_stale():
    r = staleness_report([D(7)] * 20, min_run=2)
    assert r.unchanged_share == 1
    assert r.in_runs_share == 1
    assert r.longest_run == 20


def test_one_observation_has_no_transition_to_report():
    r = staleness_report([D(1)], min_run=2)
    assert r.unchanged_share is None
    assert r.in_runs_share == 0


def test_an_empty_series_reports_no_shares_rather_than_zero():
    r = staleness_report([], min_run=2)
    assert r.observations == 0
    assert r.unchanged_share is None
    assert r.in_runs_share is None


# -- smoothing -------------------------------------------------------------

def test_an_unsmoothed_series_shows_no_bias():
    rng = random.Random(5)
    values = [D(str(round(rng.gauss(0, 1), 10))) for _ in range(4000)]
    bias = smoothing_bias(values)
    assert bias.fits
    assert abs(bias.variance_ratio - 1) < D("0.02")
    assert abs(bias.sharpe_inflation - 1) < D("0.02")


@pytest.mark.parametrize("a", [D("0.9"), D("0.8"), D("0.7"), D("0.6")])
def test_the_weights_are_recovered_from_the_autocorrelation(a):
    bias = smoothing_bias(smoothed(float(a)))
    assert bias.fits
    assert abs(bias.weight_current - a) < D("0.05")
    assert bias.weight_current + bias.weight_previous == 1


def test_more_smoothing_hides_more_volatility():
    light = smoothing_bias(smoothed(0.9))
    heavy = smoothing_bias(smoothed(0.6))
    assert heavy.variance_ratio < light.variance_ratio
    assert heavy.sharpe_inflation > light.sharpe_inflation > 1


def test_the_recovered_variance_ratio_matches_the_real_one():
    """Checked against the variance the smoothing actually removed."""
    import statistics
    rng = random.Random(23)
    true = [rng.gauss(0, 1) for _ in range(8000)]
    a, b = 0.7, 0.3
    rep = [a * true[i] + b * true[i - 1] for i in range(1, len(true))]
    bias = smoothing_bias([D(str(round(x, 10))) for x in rep])
    actual = statistics.pvariance(rep) / statistics.pvariance(true[1:])
    assert abs(float(bias.variance_ratio) - actual) < 0.03


def test_the_model_is_self_consistent():
    """The weights it returns reproduce the autocorrelation it was given."""
    bias = smoothing_bias(smoothed(0.75))
    a, b = bias.weight_current, bias.weight_previous
    assert abs(a * b / (a * a + b * b) - bias.autocorrelation) < D("1e-12")
    assert bias.variance_ratio == a * a + b * b


def test_a_negative_autocorrelation_is_not_smoothing():
    """Bid-ask bounce and mean reversion are a different story entirely."""
    values = [D(1) if i % 2 == 0 else D(-1) for i in range(200)]
    bias = smoothing_bias(values)
    assert bias.autocorrelation < 0
    assert bias.weight_current == 1 and bias.weight_previous == 0
    assert bias.variance_ratio == 1
    assert bias.sharpe_inflation == 1


def test_too_much_autocorrelation_for_a_two_period_average_refuses():
    """A trending series cannot be explained by partial staleness."""
    rng = random.Random(31)
    x, values = 0.0, []
    for _ in range(4000):
        x = 0.85 * x + rng.gauss(0, 1)
        values.append(D(str(round(x, 10))))
    bias = smoothing_bias(values)
    assert bias.autocorrelation > D("0.5")
    assert not bias.fits
    assert bias.variance_ratio is None
    assert bias.volatility_understated is None
    assert bias.sharpe_inflation is None


def test_the_inflation_is_the_reciprocal_of_the_understatement():
    bias = smoothing_bias(smoothed(0.7))
    assert bias.sharpe_inflation * bias.volatility_understated == \
        pytest.approx(D(1), abs=D("1e-20"))


def test_a_smoothing_result_is_always_marked_as_modelled():
    """The run counts are arithmetic; this one rests on an assumption."""
    assert smoothing_bias(smoothed(0.8)).modelled is True
    assert not hasattr(staleness_report(series("1 1 2"), min_run=2),
                       "modelled")


def test_a_series_that_never_moves_is_sent_to_the_other_function():
    with pytest.raises(ValueError, match="staleness_report"):
        smoothing_bias([D(0)] * 50)


def test_too_short_a_series_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        smoothing_bias([D(1)])


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Run(0, 1, D(1)), StalenessReport(0, 0, 0, 0, 0, 2),
                SmoothingBias(0, D(0), None, None, None)):
        with pytest.raises(Exception):
            obj.start = 5  # type: ignore[misc]
