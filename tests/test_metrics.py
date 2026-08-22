"""Tests for mdnorm.metrics."""
import math
from decimal import Decimal

import pytest

from mdnorm.metrics import (
    Drawdown,
    annualise_sharpe,
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdowns,
    equity_curve,
    expected_max_sharpe,
    hit_rate,
    max_drawdown,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    profit_factor,
    sharpe_ratio,
    sharpe_report,
    sortino_ratio,
    trial_variance,
    turnover,
)
from mdnorm.metrics import _phi, _phi_inv


D = Decimal


def approx(value, expected, tol=1e-9):
    assert abs(float(value) - expected) < tol, f"{value} != {expected}"


# -- the normal distribution -------------------------------------------------


def test_phi_at_zero_is_a_half():
    approx(_phi(0.0), 0.5, 1e-15)


def test_phi_is_symmetric():
    for x in (0.3, 1.0, 2.5, 4.0):
        approx(_phi(x) + _phi(-x), 1.0, 1e-14)


def test_phi_known_quantiles():
    approx(_phi(1.959963984540054), 0.975, 1e-12)
    approx(_phi(-1.2815515655446004), 0.1, 1e-12)


def test_phi_inv_round_trips():
    for p in (0.001, 0.01, 0.02, 0.2, 0.5, 0.8, 0.975, 0.999, 0.99999):
        approx(_phi(_phi_inv(p)), p, 1e-13)


def test_phi_inv_known_values():
    approx(_phi_inv(0.95), 1.6448536269514722, 1e-11)
    approx(_phi_inv(0.5), 0.0, 1e-13)


def test_phi_inv_rejects_bounds():
    with pytest.raises(ValueError):
        _phi_inv(0.0)
    with pytest.raises(ValueError):
        _phi_inv(1.0)


# -- moments -----------------------------------------------------------------


def test_moments_of_a_known_series():
    m = moments([D(2), D(4), D(4), D(4), D(5), D(5), D(7), D(9)], ddof=0)
    assert m.observations == 8
    assert m.mean == 5
    assert m.stdev == 2  # population standard deviation of the classic example
    # devs cubed: -27 -1 -1 -1 +0 +0 +8 +64 = 42, over 8 observations, over 2**3
    approx(m.skewness, 0.65625, 1e-15)


def test_moments_kurtosis_is_not_excess():
    # A symmetric two-point distribution has kurtosis 1, excess -2.
    m = moments([D(-1), D(1), D(-1), D(1)])
    assert m.kurtosis == 1
    assert m.excess_kurtosis == -2


def test_moments_ddof_changes_only_the_stdev():
    vals = [D(1), D(2), D(3), D(4)]
    a = moments(vals, ddof=0)
    b = moments(vals, ddof=1)
    assert a.mean == b.mean
    assert a.skewness == b.skewness
    assert a.kurtosis == b.kurtosis
    assert b.stdev > a.stdev


def test_moments_counts_missing_separately():
    m = moments([None, D(1), None, D(3)])
    assert m.observations == 2
    assert m.skipped == 2
    assert m.mean == 2


def test_moments_of_empty_series():
    m = moments([])
    assert m.observations == 0
    assert m.mean is None and m.stdev is None


def test_moments_of_all_missing():
    m = moments([None, None])
    assert m.observations == 0
    assert m.skipped == 2


def test_moments_constant_series_has_no_shape():
    m = moments([D(3)] * 5)
    assert m.stdev == 0
    assert m.skewness is None
    assert m.kurtosis is None
    assert m.excess_kurtosis is None


def test_moments_single_observation():
    m = moments([D(5)])
    assert m.observations == 1
    assert m.mean == 5
    assert m.stdev is None


def test_moments_rejects_negative_ddof():
    with pytest.raises(ValueError):
        moments([D(1)], ddof=-1)


# -- equity curve ------------------------------------------------------------


def test_equity_curve_is_one_longer_than_its_input():
    c = equity_curve([D("0.1"), D("0.1")])
    assert len(c) == 3
    assert c[0] == 1


def test_equity_curve_compounds():
    c = equity_curve([D("0.1"), D("0.1")])
    assert c[-1] == Decimal("1.21")


def test_equity_curve_simple_addition():
    c = equity_curve([D("0.1"), D("0.1")], compound=False)
    assert c[-1] == Decimal("1.2")


def test_equity_curve_treats_a_hole_as_no_change():
    c = equity_curve([D("0.1"), None, D("0.1")])
    assert len(c) == 4
    assert c[1] == c[2]
    assert c[-1] == Decimal("1.21")


def test_equity_curve_honours_initial_capital():
    c = equity_curve([D("0.5")], initial=D(200))
    assert c[0] == 200
    assert c[-1] == 300


def test_equity_curve_rejects_non_positive_capital():
    with pytest.raises(ValueError):
        equity_curve([D(0)], initial=D(0))


# -- drawdowns ---------------------------------------------------------------


def test_no_drawdown_in_a_rising_curve():
    assert drawdowns([D(1), D(2), D(3)]) == []
    assert max_drawdown([D(1), D(2), D(3)]) is None


def test_a_single_recovered_drawdown():
    dds = drawdowns([D(100), D(80), D(90), D(100), D(110)])
    assert len(dds) == 1
    dd = dds[0]
    assert (dd.peak_index, dd.trough_index, dd.recovery_index) == (0, 1, 3)
    assert dd.depth == Decimal("0.2")
    assert dd.recovered is True
    assert dd.length == 1
    assert dd.recovery_length == 2


def test_an_unrecovered_drawdown_stays_open():
    dds = drawdowns([D(100), D(120), D(90)])
    assert len(dds) == 1
    assert dds[0].peak_index == 1
    assert dds[0].recovery_index is None
    assert dds[0].recovered is False
    assert dds[0].recovery_length is None


def test_the_deepest_point_is_the_trough_not_the_first_dip():
    dd = drawdowns([D(100), D(90), D(70), D(85), D(100)])[0]
    assert dd.trough_index == 2
    assert dd.depth == Decimal("0.3")


def test_several_drawdowns_in_time_order():
    dds = drawdowns([D(100), D(90), D(100), D(60), D(100)])
    assert len(dds) == 2
    assert [d.trough_index for d in dds] == [1, 3]
    assert max_drawdown([D(100), D(90), D(100), D(60), D(100)]).depth == Decimal("0.4")


def test_a_new_high_closes_the_previous_decline():
    dds = drawdowns([D(100), D(80), D(105), D(95)])
    assert len(dds) == 2
    assert dds[0].recovered is True
    assert dds[1].recovered is False
    assert dds[1].peak_index == 2


def test_equal_to_the_old_high_counts_as_recovery():
    dds = drawdowns([D(100), D(80), D(100)])
    assert dds[0].recovery_index == 2


def test_drawdowns_of_a_short_curve():
    assert drawdowns([]) == []
    assert drawdowns([D(1)]) == []


def test_drawdown_indices_line_up_with_the_equity_curve():
    rets = [D("0.0"), D("-0.5"), D("1.0")]
    curve = equity_curve(rets)
    dd = max_drawdown(curve)
    assert curve[dd.peak_index] == 1
    assert curve[dd.trough_index] == Decimal("0.5")
    assert dd.depth == Decimal("0.5")


# -- ratios ------------------------------------------------------------------


def test_sharpe_is_mean_over_stdev():
    vals = [D(1), D(2), D(3), D(4)]
    m = moments(vals)
    approx(sharpe_ratio(vals), float(m.mean) / float(m.stdev), 1e-15)


def test_sharpe_subtracts_a_per_period_risk_free_rate():
    vals = [D(1), D(2), D(3), D(4)]
    plain = sharpe_ratio(vals)
    net = sharpe_ratio(vals, risk_free=D(1))
    assert net < plain


def test_sharpe_of_a_flat_series_is_none():
    assert sharpe_ratio([D(1)] * 5) is None


def test_sharpe_of_too_short_a_series_is_none():
    assert sharpe_ratio([D(1)]) is None
    assert sharpe_ratio([]) is None


def test_sharpe_is_not_annualised_by_default():
    vals = [D("0.01"), D("0.02"), D("-0.01"), D("0.03")]
    sr = sharpe_ratio(vals)
    assert sr == sharpe_ratio(vals)
    assert annualise_sharpe(sr, D(252)) > sr * 15


def test_annualise_scales_by_the_square_root():
    assert annualise_sharpe(D(1), D(4)) == 2


def test_annualise_rejects_a_non_positive_calendar():
    with pytest.raises(ValueError):
        annualise_sharpe(D(1), D(0))


def test_sortino_only_counts_the_downside():
    # Same mean, but one series has its dispersion entirely on the upside.
    upside = [D(0), D(0), D(0), D(4)]
    twosided = [D(-2), D(2), D(-2), D(6)]
    assert sortino_ratio(upside) is None  # nothing fell below zero
    assert sortino_ratio(twosided) is not None


def test_sortino_divides_by_the_whole_sample():
    # Two losses of the same size in a longer sample give a smaller downside
    # deviation, so a better ratio, than in a short one.
    short = [D(-1), D(-1), D(3), D(3)]
    longer = short + [D(1)] * 8
    assert sortino_ratio(longer) > sortino_ratio(short)


def test_sortino_uses_the_target_for_both_sides():
    vals = [D(1), D(2), D(3)]
    assert sortino_ratio(vals) is None
    assert sortino_ratio(vals, target=D(2)) is not None


def test_sortino_of_too_short_a_series():
    assert sortino_ratio([D(-1)], ddof=1) is None
    assert sortino_ratio([]) is None


def test_sortino_rejects_negative_ddof():
    with pytest.raises(ValueError):
        sortino_ratio([D(1)], ddof=-1)


def test_calmar_is_annual_return_over_worst_drawdown():
    rets = [D("0.1"), D("-0.2"), D("0.25")]
    c = calmar_ratio(rets, periods_per_year=D(3))
    curve = equity_curve(rets)
    worst = max_drawdown(curve)
    # One year of data, so the annualised return is just the total return.
    total = curve[-1] / curve[0] - 1
    approx(c, float(total / worst.depth), 1e-12)


def test_calmar_is_none_without_a_drawdown():
    assert calmar_ratio([D("0.1"), D("0.1")], periods_per_year=D(252)) is None


def test_calmar_is_none_for_an_empty_series():
    assert calmar_ratio([], periods_per_year=D(252)) is None


def test_calmar_requires_a_calendar():
    with pytest.raises(ValueError):
        calmar_ratio([D("0.1")], periods_per_year=D(0))


def test_calmar_survives_a_wipeout():
    assert calmar_ratio([D("-1")], periods_per_year=D(252)) is None


def test_hit_rate_excludes_flat_periods():
    assert hit_rate([D(1), D(-1), D(0), D(0)]) == Decimal("0.5")


def test_hit_rate_of_a_flat_series_is_none():
    assert hit_rate([D(0), D(0)]) is None
    assert hit_rate([]) is None


def test_profit_factor():
    assert profit_factor([D(3), D(-1)]) == 3


def test_profit_factor_without_a_loss_is_none():
    assert profit_factor([D(1), D(2)]) is None


def test_turnover_has_no_first_value():
    t = turnover([{"A": D(1)}, {"A": D(1)}])
    assert t[0] is None
    assert t[1] == 0


def test_turnover_is_one_sided():
    t = turnover([{"A": D(1)}, {"B": D(1)}])
    # A fully out, B fully in: two units of trading, one unit of turnover.
    assert t[1] == 1


def test_turnover_treats_an_absent_name_as_zero():
    t = turnover([{"A": D("0.5"), "B": D("0.5")}, {"A": D("0.5")}])
    assert t[1] == Decimal("0.25")


def test_turnover_of_an_empty_schedule():
    assert turnover([]) == []


# -- selection ---------------------------------------------------------------


def test_psr_of_the_benchmark_itself_is_a_half():
    p = probabilistic_sharpe_ratio(D(0), observations=100)
    approx(p, 0.5, 1e-12)


def test_psr_rises_with_the_sample():
    short = probabilistic_sharpe_ratio(D("0.2"), observations=30)
    longer = probabilistic_sharpe_ratio(D("0.2"), observations=3000)
    assert longer > short
    assert short < Decimal("0.99")


def test_psr_falls_with_negative_skew():
    plain = probabilistic_sharpe_ratio(D("0.2"), observations=250)
    skewed = probabilistic_sharpe_ratio(
        D("0.2"), observations=250, skewness=D("-1.5")
    )
    assert skewed < plain


def test_psr_falls_with_fat_tails():
    plain = probabilistic_sharpe_ratio(D("0.2"), observations=250)
    fat = probabilistic_sharpe_ratio(
        D("0.2"), observations=250, kurtosis=D(12)
    )
    assert fat < plain


def test_psr_matches_the_closed_form():
    sr, n, g3, g4 = 0.15, 500, -0.4, 5.0
    var = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr * sr
    expected = 0.5 * math.erfc(
        -(sr * math.sqrt(n - 1) / math.sqrt(var)) / math.sqrt(2.0)
    )
    got = probabilistic_sharpe_ratio(
        D(str(sr)), observations=n, skewness=D(str(g3)), kurtosis=D(str(g4))
    )
    approx(got, expected, 1e-12)


def test_psr_needs_two_observations():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(D(1), observations=1)


def test_psr_rejects_an_impossible_variance():
    with pytest.raises(ValueError) as exc:
        probabilistic_sharpe_ratio(
            D(1), observations=100, skewness=D(10), kurtosis=D(1)
        )
    assert "non-excess" in str(exc.value)


def test_min_track_record_shrinks_as_the_ratio_grows():
    weak = min_track_record_length(D("0.05"))
    strong = min_track_record_length(D("0.5"))
    assert weak > strong


def test_min_track_record_is_none_below_the_benchmark():
    assert min_track_record_length(D("-0.1")) is None
    assert min_track_record_length(D(0)) is None


def test_min_track_record_grows_with_confidence():
    a = min_track_record_length(D("0.2"), confidence=D("0.90"))
    b = min_track_record_length(D("0.2"), confidence=D("0.99"))
    assert b > a


def test_min_track_record_matches_the_closed_form():
    sr = 0.1
    z = _phi_inv(0.95)
    # Under a normal assumption the variance term is 1 + (3 - 1) / 4 * sr**2,
    # not 1: the estimator is noisier for a higher ratio, so the required
    # track record is slightly longer than the naive form suggests.
    var = 1.0 + 0.5 * sr * sr
    expected = 1.0 + var * (z / sr) ** 2
    approx(min_track_record_length(D(str(sr))), expected, 1e-9)


def test_min_track_record_agrees_with_psr():
    # At exactly the minimum track record length, the PSR should equal the
    # confidence level it was solved for.
    sr = D("0.12")
    n = min_track_record_length(sr, confidence=D("0.95"))
    p = probabilistic_sharpe_ratio(sr, observations=int(round(float(n))))
    approx(p, 0.95, 1e-3)


def test_min_track_record_rejects_bad_confidence():
    with pytest.raises(ValueError):
        min_track_record_length(D("0.2"), confidence=D(1))


def test_expected_max_sharpe_grows_with_the_search():
    small = expected_max_sharpe(10, D("0.04"))
    large = expected_max_sharpe(10000, D("0.04"))
    assert large > small > 0


def test_expected_max_sharpe_scales_with_dispersion():
    a = expected_max_sharpe(100, D("0.01"))
    b = expected_max_sharpe(100, D("0.04"))
    approx(b / a, 2.0, 1e-9)


def test_a_single_trial_selects_nothing():
    assert expected_max_sharpe(1, D("0.04")) == 0


def test_no_dispersion_selects_nothing():
    assert expected_max_sharpe(500, D(0)) == 0


def test_expected_max_sharpe_rejects_bad_input():
    with pytest.raises(ValueError):
        expected_max_sharpe(0, D("0.04"))
    with pytest.raises(ValueError):
        expected_max_sharpe(10, D("-1"))


def test_trial_variance_is_the_square_of_the_stdev():
    sharpes = [D("0.1"), D("0.2"), D("0.3"), D("0.4")]
    m = moments(sharpes)
    approx(trial_variance(sharpes), float(m.stdev * m.stdev), 1e-20)


def test_trial_variance_of_one_trial_is_none():
    assert trial_variance([D("0.1")]) is None


def test_deflation_lowers_the_probability():
    plain = probabilistic_sharpe_ratio(D("0.25"), observations=250)
    deflated = deflated_sharpe_ratio(
        D("0.25"), observations=250, trials=500, variance=D("0.01")
    )
    assert deflated < plain


def test_a_wide_search_can_erase_a_good_looking_result():
    # The same ratio, once against nothing and once against a large search.
    honest = deflated_sharpe_ratio(
        D("0.2"), observations=250, trials=1, variance=D("0.02")
    )
    searched = deflated_sharpe_ratio(
        D("0.2"), observations=250, trials=100000, variance=D("0.02")
    )
    assert honest > Decimal("0.99")
    assert searched < Decimal("0.5")


def test_deflated_equals_psr_for_a_single_trial():
    a = deflated_sharpe_ratio(
        D("0.3"), observations=100, trials=1, variance=D("0.05")
    )
    b = probabilistic_sharpe_ratio(D("0.3"), observations=100)
    assert a == b


# -- the report --------------------------------------------------------------


RETURNS = [
    D("0.004"), D("-0.002"), D("0.006"), D("0.001"), D("-0.005"),
    D("0.003"), D("0.002"), D("-0.001"), D("0.005"), D("0.000"),
    D("0.002"), D("-0.003"), D("0.004"), D("0.001"), D("0.002"),
]


def test_report_carries_the_plain_ratio():
    rep = sharpe_report(RETURNS)
    assert rep.observations == 15
    assert rep.sharpe == sharpe_ratio(RETURNS)


def test_report_warns_when_not_annualised():
    rep = sharpe_report(RETURNS)
    assert rep.sharpe_annualised is None
    assert any("per period" in w for w in rep.warnings)


def test_report_annualises_when_told_the_calendar():
    rep = sharpe_report(RETURNS, periods_per_year=D(252))
    assert rep.sharpe_annualised == annualise_sharpe(rep.sharpe, D(252))
    assert not any("per period" in w for w in rep.warnings)


def test_report_counts_missing_observations():
    rep = sharpe_report([None] + RETURNS)
    assert rep.skipped == 1
    assert rep.observations == 15
    assert any("missing" in w for w in rep.warnings)


def test_report_warns_when_the_sample_is_too_short():
    rep = sharpe_report(RETURNS)
    assert rep.demonstrated is False
    assert any("minimum track record" in w for w in rep.warnings)


def test_report_warns_that_no_trial_count_was_given():
    rep = sharpe_report(RETURNS)
    assert rep.deflated is None
    assert any("configurations were tried" in w for w in rep.warnings)


def test_report_deflates_when_given_a_search():
    rep = sharpe_report(
        RETURNS, trials=200, trial_sharpe_variance=D("0.02")
    )
    assert rep.deflated is not None
    assert rep.deflated < rep.probabilistic
    assert not any("configurations were tried" in w for w in rep.warnings)


def test_report_flags_a_result_the_search_explains():
    rep = sharpe_report(
        RETURNS, trials=100000, trial_sharpe_variance=D("0.05")
    )
    assert any("not clearly better" in w for w in rep.warnings)


def test_report_rejects_half_a_search():
    with pytest.raises(ValueError):
        sharpe_report(RETURNS, trials=10)
    with pytest.raises(ValueError):
        sharpe_report(RETURNS, trial_sharpe_variance=D("0.01"))


def test_report_of_a_flat_series():
    rep = sharpe_report([D(1)] * 10)
    assert rep.sharpe is None
    assert rep.probabilistic is None
    assert rep.demonstrated is None
    assert any("no dispersion" in w for w in rep.warnings)


def test_report_of_an_empty_series():
    rep = sharpe_report([])
    assert rep.observations == 0
    assert rep.sharpe is None
    assert any("no observations" in w for w in rep.warnings)


def test_report_demonstrated_is_true_for_a_long_enough_sample():
    strong = [D("0.01"), D("0.011"), D("0.009"), D("0.0105"), D("0.0095")] * 40
    rep = sharpe_report(strong)
    assert rep.demonstrated is True
    assert not any("minimum track record" in w for w in rep.warnings)


def test_report_min_track_record_is_none_for_a_losing_series():
    rep = sharpe_report([D("-0.01"), D("0.002"), D("-0.008"), D("0.001")])
    assert rep.min_track_record is None
    assert rep.demonstrated is None


def test_report_is_frozen():
    rep = sharpe_report(RETURNS)
    with pytest.raises(Exception):
        rep.sharpe = D(1)


# -- causality ---------------------------------------------------------------


def test_drawdowns_before_a_change_are_unaffected_by_the_tail():
    """The library's standing property: nothing before index i may depend on
    anything after it. Drawdowns already closed cannot be rewritten by later
    observations."""
    base = [D(100), D(80), D(100), D(120), D(110)]
    tampered = base[:3] + [D(9999), D(1)]
    a = [d for d in drawdowns(base) if d.recovered]
    b = [d for d in drawdowns(tampered) if d.recovered]
    assert a[0] == b[0]


def test_equity_curve_prefix_is_unaffected_by_the_tail():
    base = [D("0.01")] * 8
    tampered = base[:5] + [D("9"), D("-0.9"), D("3")]
    assert equity_curve(base)[:6] == equity_curve(tampered)[:6]


def test_drawdown_is_comparable_by_value():
    a = Drawdown(0, 1, 2, D(100), D(80), D("0.2"))
    b = Drawdown(0, 1, 2, D(100), D(80), D("0.2"))
    assert a == b
