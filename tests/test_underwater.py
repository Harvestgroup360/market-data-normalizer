"""Underwater tests: a maximum drawdown is a maximum."""
import random
from decimal import Decimal

import pytest

from mdnorm.underwater import (
    ResampledDrawdowns,
    UnderwaterReport,
    depth_quantile,
    longest_underwater,
    pain_index,
    resampled_max_drawdown,
    time_under_water,
    ulcer_index,
    underwater_curve,
    underwater_report,
    underwater_share,
)

D = Decimal


def curve(*values):
    return [D(str(v)) for v in values]


def walk(n=1260, seed=20260921, drift=0.00044, sd=0.0100, rho=0.12):
    """Returns and the equity curve they compound into."""
    rng = random.Random(seed)
    prev = 0.0
    rets = []
    for _ in range(n):
        prev = rho * prev + rng.gauss(drift, sd)
        rets.append(D(str(round(prev, 10))))
    eq = [D(1)]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return rets, eq


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the underwater curve --------------------------------------------------

def test_a_rising_curve_is_never_under_water():
    assert underwater_curve(curve(1, 2, 3, 4)) == [0, 0, 0, 0]
    assert time_under_water(curve(1, 2, 3, 4)) == 0
    assert underwater_share(curve(1, 2, 3, 4)) == 0


def test_the_depth_is_measured_from_the_running_peak_not_the_start():
    """Halving from a peak of two is fifty per cent, whatever the start was."""
    assert underwater_curve(curve(1, 2, 1)) == [0, 0, D("0.5")]


def test_a_new_high_resets_the_depth_to_zero():
    d = underwater_curve(curve(10, 8, 12, 9))
    assert d[1] == D("0.2")
    assert d[2] == 0
    assert d[3] == D("0.25")


def test_a_single_observation_has_no_drawdown():
    assert underwater_curve(curve(7)) == [0]


def test_an_empty_curve_is_refused():
    with pytest.raises(ValueError, match="at least one observation"):
        underwater_curve([])


def test_a_non_positive_peak_is_refused_by_index():
    """Returns passed where an equity curve belongs is the error this catches."""
    with pytest.raises(ArithmeticError, match="observation 0 is 0"):
        underwater_curve(curve(0, 1, 2))
    with pytest.raises(ArithmeticError, match="peak at observation"):
        underwater_curve(curve(-1, "0.5"))


# -- how long, not just how deep -------------------------------------------

def test_the_longest_stretch_is_not_the_deepest_one():
    """Eight per cent for one period, two per cent for four."""
    c = curve(100, 92, 100, 98, 98, 98, 98, 101)
    assert longest_underwater(c) == 4
    assert about(max(underwater_curve(c)), "0.08")


def test_a_stretch_still_open_at_the_end_is_counted_at_its_length_so_far():
    c = curve(10, 9, 9, 9)
    assert longest_underwater(c) == 3
    assert underwater_report(c).open_at_end


def test_a_recovered_curve_is_not_open_at_the_end():
    assert not underwater_report(curve(10, 8, 11)).open_at_end


def test_the_share_counts_observations_below_a_peak():
    c = curve(1, 2, "1.5", "1.8", 3)
    assert time_under_water(c) == 2
    assert underwater_share(c) == D(2) / D(5)


# -- the robust alternatives -----------------------------------------------

def test_a_flat_curve_has_no_pain_and_no_ulcer():
    flat = curve(5, 5, 5, 5)
    assert pain_index(flat) == 0
    assert ulcer_index(flat) == 0


def test_the_ulcer_index_never_exceeds_the_deepest_drawdown():
    """It is a root mean square of the same numbers the maximum is over."""
    for seed in range(6):
        _, eq = walk(n=400, seed=seed)
        rep = underwater_report(eq)
        assert rep.ulcer <= rep.deepest


def test_the_pain_index_never_exceeds_the_ulcer_index():
    """Mean is at most root mean square, for any set of non-negative depths."""
    for seed in range(6):
        _, eq = walk(n=400, seed=seed)
        rep = underwater_report(eq)
        assert rep.pain <= rep.ulcer


def test_one_bad_day_moves_the_maximum_and_barely_moves_the_ulcer():
    """The whole argument for reporting more than a maximum."""
    calm = curve(*([100] * 200 + [101]))
    spike = curve(*([100] * 100 + [70] + [100] * 99 + [101]))
    a, b = underwater_report(calm), underwater_report(spike)
    assert a.deepest == 0
    assert about(b.deepest, "0.3", "0.001")
    assert b.ulcer < D("0.03")          # one observation in two hundred
    assert b.concentration < D("0.1")   # the maximum knows nothing about the rest


def test_concentration_is_high_when_the_curve_stayed_near_its_worst():
    sunk = curve(*([100] + [80] * 199))
    rep = underwater_report(sunk)
    assert about(rep.deepest, "0.2", "0.001")
    assert rep.concentration > D("0.99")


def test_a_curve_that_never_fell_has_no_concentration_rather_than_zero():
    rep = underwater_report(curve(1, 2, 3))
    assert rep.never_fell
    assert rep.concentration is None


# -- the quantile ----------------------------------------------------------

def test_the_median_depth_is_below_the_deepest():
    _, eq = walk()
    assert depth_quantile(eq, level=D("0.5")) < underwater_report(eq).deepest


def test_the_top_quantile_is_the_maximum():
    _, eq = walk(n=300)
    assert depth_quantile(eq, level=D(1)) == underwater_report(eq).deepest


def test_the_quantile_rises_with_the_level():
    _, eq = walk(n=500)
    got = [depth_quantile(eq, level=D(str(l)))
           for l in ("0.1", "0.25", "0.5", "0.75", "0.95")]
    assert got == sorted(got)


def test_a_level_outside_the_unit_interval_is_refused():
    _, eq = walk(n=50)
    with pytest.raises(ValueError, match=r"lies in \[0, 1\]"):
        depth_quantile(eq, level=D("1.5"))


# -- the length effect, which is the point ---------------------------------

def test_the_worst_drawdown_grows_with_the_horizon():
    """Same returns, same process, longer look: a deeper worst decline.

    This is the module's whole argument. Nothing about the strategy differs
    between these four rows; only the number of chances it had to have a bad
    run.
    """
    rets, _ = walk()
    medians = [resampled_max_drawdown(rets, periods=p, paths=300,
                                      seed=7).median
               for p in (252, 756, 1260, 2520)]
    assert medians == sorted(medians)
    assert medians[-1] > medians[0] * D("1.8")


def test_the_resample_brackets_the_observed_drawdown():
    """A sanity check, not a claim: the realised figure sits inside the
    distribution its own returns generate at the same horizon."""
    rets, eq = walk()
    observed = underwater_report(eq).deepest
    r = resampled_max_drawdown(rets, periods=len(rets), paths=500, seed=7)
    assert r.quantile(D("0.05")) < observed < r.quantile(D("0.95"))


def test_the_same_seed_reproduces_the_distribution():
    rets, _ = walk(n=300)
    a = resampled_max_drawdown(rets, periods=300, paths=50, seed=3)
    b = resampled_max_drawdown(rets, periods=300, paths=50, seed=3)
    assert a.depths == b.depths


def test_a_different_seed_does_not():
    rets, _ = walk(n=300)
    a = resampled_max_drawdown(rets, periods=300, paths=50, seed=3)
    b = resampled_max_drawdown(rets, periods=300, paths=50, seed=4)
    assert a.depths != b.depths


def test_every_simulated_depth_is_a_fraction():
    rets, _ = walk(n=300)
    r = resampled_max_drawdown(rets, periods=200, paths=60, seed=1)
    assert all(0 <= d < 1 for d in r.depths)
    assert r.paths == 60 and r.periods == 200


def test_the_depths_come_back_sorted():
    rets, _ = walk(n=300)
    r = resampled_max_drawdown(rets, periods=200, paths=60, seed=1)
    assert list(r.depths) == sorted(r.depths)
    assert r.worst == r.depths[-1]


def test_a_single_path_is_its_own_quantile():
    r = ResampledDrawdowns(paths=1, periods=10, depths=(D("0.2"),))
    assert r.quantile(D("0.5")) == D("0.2")
    assert r.median == D("0.2") == r.worst


def test_a_return_that_empties_the_account_is_refused_by_index():
    with pytest.raises(ValueError, match="return 1 is -1"):
        resampled_max_drawdown([D("0.01"), D("-1")], periods=10, paths=5,
                               seed=1)


def test_the_message_points_at_the_log_return_conversion():
    with pytest.raises(ValueError, match="mdnorm.to_simple"):
        resampled_max_drawdown([D("-1.4")], periods=10, paths=5, seed=1)


def test_non_positive_paths_or_periods_are_refused():
    rets, _ = walk(n=50)
    with pytest.raises(ValueError, match="periods must be positive"):
        resampled_max_drawdown(rets, periods=0, paths=5, seed=1)
    with pytest.raises(ValueError, match="paths must be positive"):
        resampled_max_drawdown(rets, periods=5, paths=0, seed=1)


# -- the report ------------------------------------------------------------

def test_the_report_agrees_with_the_standalone_functions():
    _, eq = walk(n=600)
    rep = underwater_report(eq)
    assert rep.deepest == max(underwater_curve(eq))
    assert rep.pain == pain_index(eq)
    assert rep.ulcer == ulcer_index(eq)
    assert rep.periods_under_water == time_under_water(eq)
    assert rep.longest_underwater == longest_underwater(eq)
    assert rep.underwater_share == underwater_share(eq)


def test_it_agrees_with_the_metrics_module_on_the_deepest():
    """Two implementations, one number.

    Not bit-identical: this module works at forty significant digits and
    `metrics` at thirty-four, which is a deliberate per-module choice. Thirty
    digits of agreement is far beyond what any real price series carries, and
    a disagreement above that would mean one of the two is wrong.
    """
    from mdnorm.metrics import max_drawdown

    for seed in range(8):
        _, eq = walk(n=400, seed=seed)
        assert about(underwater_report(eq).deepest, max_drawdown(eq).depth,
                     "1e-30")


def test_the_worked_example():
    rets, eq = walk()
    rep = underwater_report(eq)
    assert rep.observations == 1261
    assert about(rep.deepest, "0.208839")
    assert about(rep.ulcer, "0.088358")
    assert about(rep.pain, "0.071522")
    assert about(rep.concentration, "0.423089")
    assert rep.periods_under_water == 1172
    assert rep.longest_underwater == 281
    assert rep.open_at_end


def test_the_worked_example_reports_more_time_under_water_than_anyone_expects():
    """Ninety-three per cent of five years below a previous high, on a
    strategy whose annualised Sharpe ratio is 0.71."""
    _, eq = walk()
    rep = underwater_report(eq)
    assert rep.underwater_share > D("0.92")
    assert rep.longest_underwater > 250


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    _, eq = walk(n=100)
    for obj in (underwater_report(eq),
                ResampledDrawdowns(1, 1, (D(0),))):
        with pytest.raises(Exception):
            obj.paths = 9  # type: ignore[misc]
