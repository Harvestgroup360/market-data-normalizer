"""Flow tests: a strategy earns a return, an investor earns a rate."""
import random
from decimal import Decimal

import pytest

from mdnorm.flows import (
    FlowComparison,
    FlowReport,
    balances,
    compare_flows,
    flow_report,
    internal_rate_of_return,
    level_flows,
    modified_dietz,
    sign_changes,
    time_weighted,
)

D = Decimal


def market(n=60, seed=20260923, mu=0.008, sigma=0.045):
    """Five years of monthly returns for one unchanged strategy."""
    rng = random.Random(seed)
    return [D(str(round(rng.gauss(mu, sigma), 10))) for _ in range(n)]


def trailing(returns, i, k=12):
    acc = D(1)
    for x in returns[max(0, i - k):i]:
        acc *= D(1) + x
    return acc - D(1)


def chasing(returns, up=D(20), down=D(5), base=D(10), k=12):
    """Contribute more after a good year and less after a bad one."""
    return [base if i < k else (up if trailing(returns, i, k) > 0 else down)
            for i in range(len(returns))]


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the arguments a rate of return cannot do without ----------------------

def test_the_timing_convention_has_no_default():
    with pytest.raises(TypeError):
        flow_report([D("0.01")], [D(1)])  # type: ignore[call-arg]


def test_an_unknown_timing_convention_is_refused():
    with pytest.raises(ValueError, match="when is one of"):
        flow_report([D("0.01")], [D(1)], when="middle")


def test_flows_and_returns_must_describe_the_same_periods():
    with pytest.raises(ValueError, match="must describe the same periods"):
        flow_report([D("0.01")] * 3, [D(1)] * 2, when="start")


def test_an_empty_sample_is_refused():
    with pytest.raises(ValueError, match="at least one period"):
        flow_report([], [], when="start")


def test_a_return_at_minus_one_is_refused():
    with pytest.raises(ValueError, match="empties the account"):
        flow_report([D("0.01"), D(-1)], [D(1), D(0)], when="start")


def test_a_withdrawal_larger_than_the_balance_is_refused():
    with pytest.raises(ArithmeticError, match="takes the account to"):
        flow_report([D("0.01")] * 2, [D(100), D(-200)], when="start")


def test_annualising_needs_a_calendar():
    rep = flow_report(market(n=12), [D(10)] * 12, when="start")
    with pytest.raises(TypeError):
        rep.annualised(rep.money_weighted)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        rep.annualised(rep.money_weighted, periods_per_year=0)


# -- the timing convention changes the answer ------------------------------

def test_a_flow_at_the_start_earns_the_period_and_at_the_end_does_not():
    assert balances([D("0.10")], [D(100)], when="start") == (D(110),)
    assert balances([D("0.10")], [D(100)], when="end") == (D(100),)


def test_the_two_conventions_disagree_on_a_real_path():
    r = market()
    f = [D(10)] * 60
    start = flow_report(r, f, when="start")
    end = flow_report(r, f, when="end")
    assert start.money_weighted != end.money_weighted
    assert start.terminal_value > end.terminal_value


# -- the time-weighted return is blind to the flows ------------------------

def test_the_chain_linked_return_is_the_same_for_every_path():
    r = market()
    paths = {"level": [D(10)] * 60, "chasing": chasing(r),
             "contrarian": chasing(r, up=D(5), down=D(20))}
    cmp = compare_flows(r, paths, when="start")
    assert len({rep.time_weighted for rep in cmp.reports}) == 1
    assert about(cmp.time_weighted, "0.543659")


def test_the_money_weighted_return_is_not():
    r = market()
    paths = {"level": [D(10)] * 60, "chasing": chasing(r),
             "contrarian": chasing(r, up=D(5), down=D(20))}
    cmp = compare_flows(r, paths, when="start")
    assert about(cmp.row("level").money_weighted_total, "0.629243")
    assert about(cmp.row("chasing").money_weighted_total, "0.610070")
    assert about(cmp.row("contrarian").money_weighted_total, "0.666081")
    assert about(cmp.money_weighted_spread, "0.056011")


def test_chasing_earned_least_on_the_worked_example():
    r = market()
    cmp = compare_flows(r, {"level": [D(10)] * 60, "chasing": chasing(r)},
                        when="start")
    assert (cmp.row("chasing").money_weighted
            < cmp.row("level").money_weighted)


# -- the textbook case -----------------------------------------------------

def test_a_strategy_that_returned_nothing_can_lose_the_investor_money():
    rep = flow_report([D(1), D("-0.5")], [D(100), D(100)], when="start")
    assert rep.time_weighted == 0
    assert about(rep.money_weighted, "-0.177124")
    assert about(rep.money_weighted_total, "-0.322876")
    assert rep.terminal_value == D("150.0")
    assert rep.profit == D("-50.0")
    assert rep.flows_helped is False


def test_the_gap_has_the_sign_of_the_difference():
    rep = flow_report([D(1), D("-0.5")], [D(100), D(100)], when="start")
    assert rep.gap < 0
    mirrored = flow_report([D("-0.5"), D(1)], [D(100), D(100)], when="start")
    assert mirrored.gap > 0
    assert mirrored.flows_helped is True


# -- the internal rate of return -------------------------------------------

def test_a_single_contribution_returns_the_strategy_rate():
    r = [D("0.02")] * 4
    rep = flow_report(r, [D(100), D(0), D(0), D(0)], when="start")
    assert about(rep.money_weighted, "0.02", "0.0000000001")
    assert about(rep.gap, "0", "0.0000000001")


def test_a_constant_return_gives_that_rate_whatever_the_flows():
    r = [D("0.01")] * 24
    for f in ([D(10)] * 24, [D(100)] + [D(1)] * 23,
              [D(5)] * 12 + [D(50)] * 12):
        rep = flow_report(r, f, when="start")
        assert about(rep.money_weighted, "0.01", "0.0000000001")


def test_the_rate_reproduces_the_terminal_value():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    y = rep.money_weighted
    grown = sum(f * (D(1) + y) ** (60 - i) for i, f in enumerate(rep.flows))
    assert abs(grown - rep.terminal_value) < D("0.00000001")


def test_the_net_present_value_at_the_reported_rate_is_zero():
    rep = flow_report(market(), chasing(market()), when="start")
    assert abs(rep.npv(rep.money_weighted)) < D("1E-30")


def test_flows_of_one_sign_have_no_internal_rate():
    with pytest.raises(ArithmeticError, match="same sign"):
        internal_rate_of_return([D(-10), D(-10), D(-10)])


def test_an_internal_rate_needs_two_dates():
    with pytest.raises(ValueError, match="at least two dates"):
        internal_rate_of_return([D(-10)])


def test_sign_changes_ignores_zeros():
    assert sign_changes([D(-10), D(0), D(0), D(15)]) == 1
    assert sign_changes([D(-10), D(20), D(-5), D(30)]) == 3
    assert sign_changes([D(0), D(0)]) == 0


def test_a_path_with_one_sign_change_is_marked_unique():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    assert rep.root_unique is True


def test_a_path_that_pays_out_and_back_is_not():
    r = market(n=24)
    f = [D(100)] + [D(0)] * 10 + [D(-80)] + [D(50)] + [D(0)] * 11
    rep = flow_report(r, f, when="start")
    assert sign_changes(rep.cashflows) > 1
    assert rep.root_unique is False
    assert abs(rep.npv(rep.money_weighted)) < D("1E-25")


# -- modified Dietz --------------------------------------------------------

def test_dietz_is_exact_when_one_flow_opens_the_window():
    r = [D("0.01")] * 12
    rep = flow_report(r, [D(100)] + [D(0)] * 11, when="start")
    assert about(rep.dietz_error, "0", "0.0000000001")


def test_dietz_misses_a_constant_rate_when_the_flows_are_spread():
    """It charges simple interest on the weighted base and never compounds."""
    rep = flow_report([D("0.01")] * 12, [D(10)] * 12, when="start")
    assert about(rep.modified_dietz, "0.124512")
    assert about(rep.money_weighted_total, "0.126825")
    assert rep.dietz_error < 0


def test_dietz_misses_when_the_returns_move():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    assert about(rep.modified_dietz, "0.579104")
    assert about(rep.dietz_error, "-0.050139")


def test_dietz_refuses_a_negative_capital_base():
    with pytest.raises(ArithmeticError, match="weighted average capital"):
        modified_dietz([D(2), D(2), D("0.01")], [D(10), D(0), D(-40)],
                       when="start")


# -- what the money was doing ---------------------------------------------

def test_the_capital_series_follows_the_convention():
    r = [D("0.10")] * 3
    start = flow_report(r, [D(100)] * 3, when="start")
    assert start.capital == (D(100), D(210), D(331))
    end = flow_report(r, [D(100)] * 3, when="end")
    assert end.capital[0] == 0


def test_more_money_was_exposed_to_the_worst_period_than_to_a_typical_one():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    assert rep.worst_period == 33
    assert rep.best_period == 28
    assert about(rep.capital_at_worst_ratio, "1.177099", "0.00001")
    assert about(rep.capital_at_best_ratio, "0.921429", "0.00001")


def test_contributions_and_withdrawals_are_kept_apart():
    rep = flow_report(market(n=12), [D(1000)] + [D(-90)] * 11, when="start")
    assert rep.contributed == D(1000)
    assert rep.withdrawn == D(990)
    assert abs(rep.profit - (rep.terminal_value - D(10))) < D("1E-25")


# -- timing against the level counterfactual -------------------------------

def test_a_level_schedule_has_no_timing_effect_against_itself():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    assert abs(rep.timing_effect) < D("1E-25")


def test_level_flows_keep_the_net_amount():
    f = chasing(market())
    level = level_flows(f)
    assert sum(level) == sum(f)
    assert len(set(level)) == 1


def test_the_timing_effect_separates_behaviour_from_arithmetic():
    r = market()
    chase = flow_report(r, chasing(r), when="start")
    assert chase.timing_effect is not None
    assert chase.timing_effect < 0            # the timing cost something
    assert chase.gap > 0                      # the arithmetic gave more back
    assert about(chase.timing_effect, "-0.019173", "0.00001")


def test_the_counterfactual_is_refused_when_nothing_was_contributed():
    r = [D("0.10")] * 12
    kept = flow_report(r, [D(1000)] + [D(0)] * 11, when="start")
    assert kept.timing_effect is not None
    out = flow_report(r, [D(1000)] + [D(0)] * 10 + [D(-1600)], when="start")
    assert sum(out.flows) < 0
    assert out.timing_effect is None


def test_level_flows_needs_a_date():
    with pytest.raises(ValueError, match="at least one date"):
        level_flows([])


# -- the pieces line up ----------------------------------------------------

def test_the_chain_linked_return_matches_its_per_period_form():
    rep = flow_report(market(), [D(10)] * 60, when="start")
    acc = D(1)
    for _ in range(60):
        acc *= D(1) + rep.time_weighted_per_period
    assert abs(acc - D(1) - rep.time_weighted) < D("1E-25")


def test_time_weighted_refuses_what_it_cannot_compound():
    with pytest.raises(ValueError, match="at least one period"):
        time_weighted([])
    with pytest.raises(ValueError, match="empties the account"):
        time_weighted([D("0.01"), D("-1.5")])


def test_annualising_compounds_the_calendar():
    rep = flow_report(market(n=12), [D(10)] * 12, when="start")
    y = rep.money_weighted
    assert about(rep.annualised(y, periods_per_year=12),
                 (D(1) + y) ** 12 - D(1), "1E-25")


def test_the_balances_end_where_the_report_does():
    r = market()
    f = chasing(r)
    assert balances(r, f, when="start")[-1] == flow_report(
        r, f, when="start").terminal_value


# -- comparison ------------------------------------------------------------

def test_a_comparison_needs_a_path():
    with pytest.raises(ValueError, match="at least one flow path"):
        compare_flows(market(), {}, when="start")


def test_the_comparison_keeps_the_order_it_was_given():
    r = market()
    cmp = compare_flows(r, {"b": [D(10)] * 60, "a": chasing(r)}, when="start")
    assert cmp.labels == ("b", "a")
    assert cmp.row("a").flows == tuple(chasing(r))


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    rep = flow_report(market(n=12), [D(10)] * 12, when="start")
    cmp = compare_flows(market(n=12), {"a": [D(10)] * 12}, when="start")
    for obj in (rep, cmp):
        with pytest.raises(Exception):
            obj.periods = 9  # type: ignore[misc]


def test_the_results_are_the_documented_types():
    cmp = compare_flows(market(n=12), {"a": [D(10)] * 12}, when="start")
    assert isinstance(cmp, FlowComparison)
    assert isinstance(cmp.row("a"), FlowReport)
