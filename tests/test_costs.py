"""Tests for mdnorm.costs."""
from decimal import Decimal

import pytest

from mdnorm.costs import (
    CostModel,
    Fees,
    ImpactModel,
    Liquidity,
    apply_costs,
    breakeven_participation,
    capacity,
    cost_report,
    estimate,
)

D = Decimal


def approx(value, expected, tol=1e-9):
    assert abs(float(value) - expected) < tol, f"{value} != {expected}"


LIQ = Liquidity(adv=D(1_000_000), volatility=D("0.02"), spread_bps=D(4))


# -- Fees --------------------------------------------------------------------


def test_taker_fee_is_bps_of_notional():
    f = Fees(taker_bps=D(2))
    assert f.commission(D(100_000), D(1_000)) == 20


def test_maker_and_taker_are_separate():
    f = Fees(maker_bps=D(1), taker_bps=D(5))
    assert f.commission(D(100_000), D(1), maker=True) == 10
    assert f.commission(D(100_000), D(1), maker=False) == 50


def test_per_unit_commission_adds_to_the_notional_rate():
    f = Fees(taker_bps=D(1), per_unit=D("0.005"))
    # 10 on notional, 5 on 1000 shares
    assert f.commission(D(100_000), D(1_000)) == 15


def test_the_minimum_is_a_floor_not_an_addition():
    f = Fees(taker_bps=D(1), minimum=D(1))
    assert f.commission(D(100), D(1)) == 1        # 0.01 would be charged
    assert f.commission(D(1_000_000), D(1)) == 100  # well above the floor


def test_a_zero_order_costs_nothing_even_with_a_minimum():
    assert Fees(minimum=D(5)).commission(D(0), D(0)) == 0


def test_fees_reject_negative_rates():
    for kw in ("maker_bps", "taker_bps", "per_unit", "minimum"):
        with pytest.raises(ValueError):
            Fees(**{kw: D(-1)})


def test_commission_rejects_a_negative_trade():
    with pytest.raises(ValueError):
        Fees().commission(D(-1), D(1))


# -- Liquidity ---------------------------------------------------------------


def test_participation_is_quantity_over_volume():
    assert LIQ.participation(D(50_000)) == Decimal("0.05")


def test_liquidity_rejects_non_positive_volume():
    with pytest.raises(ValueError):
        Liquidity(adv=D(0), volatility=D("0.02"))


def test_liquidity_rejects_negative_volatility_and_spread():
    with pytest.raises(ValueError):
        Liquidity(adv=D(1), volatility=D("-0.01"))
    with pytest.raises(ValueError):
        Liquidity(adv=D(1), volatility=D("0.01"), spread_bps=D(-1))


def test_participation_rejects_a_negative_quantity():
    with pytest.raises(ValueError):
        LIQ.participation(D(-1))


# -- ImpactModel -------------------------------------------------------------


def test_impact_follows_the_square_root_by_default():
    m = ImpactModel(coefficient=D(1))
    # 1 * 0.02 * sqrt(0.01) = 0.002 -> 20 bps
    approx(m.cost_bps(D("0.01"), D("0.02")), 20.0, 1e-12)


def test_quadrupling_size_doubles_square_root_impact():
    m = ImpactModel(coefficient=D(1))
    a = m.cost_bps(D("0.01"), D("0.02"))
    b = m.cost_bps(D("0.04"), D("0.02"))
    approx(b / a, 2.0, 1e-12)


def test_impact_is_linear_in_volatility():
    m = ImpactModel(coefficient=D(1))
    a = m.cost_bps(D("0.01"), D("0.01"))
    b = m.cost_bps(D("0.01"), D("0.02"))
    approx(b / a, 2.0, 1e-12)


def test_impact_is_linear_in_the_coefficient():
    a = ImpactModel(coefficient=D(1)).cost_bps(D("0.01"), D("0.02"))
    b = ImpactModel(coefficient=D(3)).cost_bps(D("0.01"), D("0.02"))
    approx(b / a, 3.0, 1e-12)


def test_a_linear_exponent_makes_cost_proportional_to_size():
    m = ImpactModel(coefficient=D(1), exponent=D(1))
    a = m.cost_bps(D("0.01"), D("0.02"))
    b = m.cost_bps(D("0.02"), D("0.02"))
    approx(b / a, 2.0, 1e-12)


def test_a_general_exponent_is_computed_by_logarithm():
    m = ImpactModel(coefficient=D(1), exponent=D("0.6"))
    # 1 * 0.02 * 0.01**0.6 * 10000
    approx(m.cost_bps(D("0.01"), D("0.02")), 0.02 * (0.01 ** 0.6) * 10_000, 1e-9)


def test_zero_participation_has_zero_impact():
    assert ImpactModel(coefficient=D(1)).cost_bps(D(0), D("0.02")) == 0


def test_impact_model_has_no_default_coefficient():
    with pytest.raises(TypeError):
        ImpactModel()


def test_impact_model_rejects_bad_parameters():
    with pytest.raises(ValueError):
        ImpactModel(coefficient=D(-1))
    with pytest.raises(ValueError):
        ImpactModel(coefficient=D(1), exponent=D(0))


def test_impact_rejects_negative_participation():
    with pytest.raises(ValueError):
        ImpactModel(coefficient=D(1)).cost_bps(D("-0.01"), D("0.02"))


# -- estimate ----------------------------------------------------------------


def test_estimate_adds_the_three_components():
    model = CostModel(fees=Fees(taker_bps=D(1)),
                      impact=ImpactModel(coefficient=D(1)))
    b = estimate(model, notional=D(100_000), quantity=D(10_000), liquidity=LIQ)
    assert b.commission_bps == 1
    assert b.spread_bps == 2               # half of a 4 bps quoted spread
    approx(b.impact_bps, 0.02 * (0.01 ** 0.5) * 10_000, 1e-9)
    approx(b.total_bps, float(b.commission_bps + b.spread_bps + b.impact_bps), 1e-15)


def test_estimate_converts_the_total_back_to_currency():
    model = CostModel(fees=Fees(taker_bps=D(10)))
    b = estimate(model, notional=D(100_000), quantity=D(1), liquidity=LIQ)
    approx(b.total, float(b.total_bps) / 10_000 * 100_000, 1e-9)


def test_fixed_bps_excludes_impact():
    model = CostModel(fees=Fees(taker_bps=D(1)),
                      impact=ImpactModel(coefficient=D(1)))
    b = estimate(model, notional=D(100_000), quantity=D(10_000), liquidity=LIQ)
    assert b.fixed_bps == b.commission_bps + b.spread_bps
    assert b.fixed_bps < b.total_bps


def test_spread_fraction_controls_how_much_of_the_spread_is_paid():
    passive = CostModel(spread_fraction=D(0))
    crossing = CostModel(spread_fraction=D("0.5"))
    picked_off = CostModel(spread_fraction=D(1))
    q = dict(notional=D(100_000), quantity=D(100), liquidity=LIQ)
    assert estimate(passive, **q).spread_bps == 0
    assert estimate(crossing, **q).spread_bps == 2
    assert estimate(picked_off, **q).spread_bps == 4


def test_participation_is_reported():
    b = estimate(CostModel(), notional=D(1), quantity=D(250_000), liquidity=LIQ)
    assert b.participation == Decimal("0.25")


def test_estimate_warns_when_the_cost_ignores_size():
    b = estimate(CostModel(fees=Fees(taker_bps=D(1))),
                 notional=D(100_000), quantity=D(10_000), liquidity=LIQ)
    assert b.impact_bps == 0
    assert any("does not depend on trade size" in w for w in b.warnings)


def test_estimate_warns_beyond_the_calibrated_range():
    model = CostModel(impact=ImpactModel(coefficient=D(1)))
    b = estimate(model, notional=D(1), quantity=D(300_000), liquidity=LIQ)
    assert any("beyond the range" in w for w in b.warnings)


def test_estimate_is_quiet_at_a_small_participation():
    model = CostModel(impact=ImpactModel(coefficient=D(1)))
    b = estimate(model, notional=D(1), quantity=D(1_000), liquidity=LIQ)
    assert not any("beyond the range" in w for w in b.warnings)


def test_an_impact_model_without_liquidity_is_an_error():
    model = CostModel(impact=ImpactModel(coefficient=D(1)))
    with pytest.raises(ValueError) as exc:
        estimate(model, notional=D(100), quantity=D(1))
    assert "size-independent" in str(exc.value)


def test_without_liquidity_no_spread_is_charged_and_it_says_so():
    b = estimate(CostModel(fees=Fees(taker_bps=D(1))),
                 notional=D(100_000), quantity=D(10))
    assert b.spread_bps == 0
    assert b.participation is None
    assert any("midpoint" in w for w in b.warnings)


def test_estimate_rejects_a_negative_trade():
    with pytest.raises(ValueError):
        estimate(CostModel(), notional=D(-1), quantity=D(1), liquidity=LIQ)


def test_breakdown_is_frozen():
    b = estimate(CostModel(), notional=D(1), quantity=D(1), liquidity=LIQ)
    with pytest.raises(Exception):
        b.total_bps = D(0)


# -- apply_costs -------------------------------------------------------------


def test_costs_are_charged_on_twice_the_one_sided_turnover():
    net = apply_costs([D("0.01")], [D("0.5")], cost_bps=D(10))
    # 0.5 one-sided turnover = 1.0 of notional traded at 10 bps
    assert net[0] == Decimal("0.009")


def test_no_turnover_means_no_charge():
    assert apply_costs([D("0.01")], [D(0)], cost_bps=D(50))[0] == Decimal("0.01")


def test_a_hole_propagates_rather_than_becoming_free():
    net = apply_costs([D("0.01"), None, D("0.01")],
                      [None, D("0.5"), D("0.5")], cost_bps=D(10))
    assert net[0] is None and net[1] is None
    assert net[2] == Decimal("0.009")


def test_apply_costs_requires_matching_lengths():
    with pytest.raises(ValueError):
        apply_costs([D(1)], [D(1), D(1)], cost_bps=D(1))


def test_apply_costs_rejects_a_negative_cost():
    with pytest.raises(ValueError):
        apply_costs([D(1)], [D(1)], cost_bps=D(-1))


def test_apply_costs_of_an_empty_series():
    assert apply_costs([], [], cost_bps=D(10)) == []


# -- cost_report -------------------------------------------------------------


GROSS = [D("0.01")] * 10
TURN = [D("0.25")] * 10


def test_report_compounds_both_series():
    r = cost_report(GROSS, TURN, cost_bps=D(10))
    approx(r.gross_return, 1.01 ** 10 - 1, 1e-12)
    assert r.net_return < r.gross_return
    approx(r.cost, float(r.gross_return - r.net_return), 1e-15)


def test_report_states_the_share_of_gross_that_trading_took():
    r = cost_report(GROSS, TURN, cost_bps=D(10))
    approx(r.cost_fraction, float(r.cost / r.gross_return), 1e-15)
    assert 0 < r.cost_fraction < 1


def test_report_counts_turnover():
    r = cost_report(GROSS, TURN, cost_bps=D(10))
    assert r.periods == 10
    assert r.total_turnover == Decimal("2.5")
    assert r.average_turnover == Decimal("0.25")


def test_report_flags_a_strategy_that_only_works_before_costs():
    r = cost_report(GROSS, [D(1)] * 10, cost_bps=D(60))
    assert r.gross_return > 0
    assert r.net_return < 0
    assert any("not after them" in w for w in r.warnings)


def test_report_flags_costs_that_dominate():
    # 0.5 turnover at 60 bps is 0.6% a period against a 1% gross return
    r = cost_report(GROSS, [D("0.5")] * 10, cost_bps=D(60))
    assert r.net_return > 0
    assert any("consume" in w for w in r.warnings)


def test_report_is_quiet_when_costs_are_small():
    r = cost_report(GROSS, [D("0.01")] * 10, cost_bps=D(1))
    assert not any("consume" in w for w in r.warnings)
    assert not any("not after them" in w for w in r.warnings)


def test_report_says_when_the_model_was_never_exercised():
    r = cost_report(GROSS, [D(0)] * 10, cost_bps=D(100))
    assert r.net_return == r.gross_return
    assert any("never exercised" in w for w in r.warnings)


def test_report_counts_excluded_periods():
    r = cost_report([None] + GROSS, [D("0.25")] * 11, cost_bps=D(10))
    assert r.skipped == 1
    assert r.periods == 10
    assert any("excluded" in w for w in r.warnings)


def test_report_of_an_empty_series():
    r = cost_report([], [], cost_bps=D(10))
    assert r.periods == 0
    assert r.gross_return is None
    assert "no periods" in r.warnings


def test_report_handles_a_zero_gross_return():
    r = cost_report([D(0)] * 5, [D("0.1")] * 5, cost_bps=D(10))
    assert r.gross_return == 0
    assert r.cost_fraction is None
    assert r.net_return < 0


def test_compounded_cost_is_not_the_sum_of_the_charges():
    """The difference between two compounded series is not the sum of the
    per-period differences, and the gap grows with the sample."""
    r = cost_report(GROSS, TURN, cost_bps=D(10))
    naive = sum(2 * t * Decimal(10) / 10_000 for t in TURN)
    assert r.cost != naive


# -- breakeven and capacity --------------------------------------------------


MODEL = CostModel(fees=Fees(taker_bps=D(1)),
                  impact=ImpactModel(coefficient=D(1)))


def test_breakeven_solves_the_cost_equation():
    p = breakeven_participation(D(20), model=MODEL, liquidity=LIQ)
    b = estimate(MODEL, notional=D(1_000_000), quantity=p * LIQ.adv,
                 liquidity=LIQ)
    approx(b.total_bps, 20.0, 1e-9)


def test_a_bigger_edge_supports_a_bigger_trade():
    small = breakeven_participation(D(10), model=MODEL, liquidity=LIQ)
    large = breakeven_participation(D(40), model=MODEL, liquidity=LIQ)
    assert large > small


def test_a_more_liquid_name_supports_the_same_participation():
    """Participation is scale-free; capacity is not."""
    thin = Liquidity(adv=D(100_000), volatility=D("0.02"), spread_bps=D(4))
    deep = Liquidity(adv=D(100_000_000), volatility=D("0.02"), spread_bps=D(4))
    assert (breakeven_participation(D(20), model=MODEL, liquidity=thin)
            == breakeven_participation(D(20), model=MODEL, liquidity=deep))
    assert (capacity(D(20), model=MODEL, liquidity=deep)
            > capacity(D(20), model=MODEL, liquidity=thin))


def test_capacity_is_participation_times_volume():
    p = breakeven_participation(D(20), model=MODEL, liquidity=LIQ)
    approx(capacity(D(20), model=MODEL, liquidity=LIQ), float(p * LIQ.adv), 1e-9)


def test_a_more_volatile_name_has_less_capacity():
    calm = Liquidity(adv=D(1_000_000), volatility=D("0.01"), spread_bps=D(4))
    wild = Liquidity(adv=D(1_000_000), volatility=D("0.04"), spread_bps=D(4))
    assert (capacity(D(20), model=MODEL, liquidity=wild)
            < capacity(D(20), model=MODEL, liquidity=calm))


def test_an_edge_below_the_fixed_costs_has_no_breakeven():
    # 1 bps fee + 2 bps half-spread = 3 bps before a single unit is traded.
    assert breakeven_participation(D(2), model=MODEL, liquidity=LIQ) is None
    assert capacity(D(2), model=MODEL, liquidity=LIQ) is None


def test_an_edge_exactly_at_the_fixed_costs_has_no_breakeven():
    assert breakeven_participation(D(3), model=MODEL, liquidity=LIQ) is None


def test_a_model_without_impact_has_no_breakeven():
    flat = CostModel(fees=Fees(taker_bps=D(1)))
    assert breakeven_participation(D(50), model=flat, liquidity=LIQ) is None
    assert capacity(D(50), model=flat, liquidity=LIQ) is None


def test_breakeven_with_a_linear_impact_model():
    linear = CostModel(impact=ImpactModel(coefficient=D(1), exponent=D(1)))
    liq = Liquidity(adv=D(1_000_000), volatility=D("0.02"))
    # cost_bps = 0.02 * p * 10000 = 200p ; edge 20 -> p = 0.1
    approx(breakeven_participation(D(20), model=linear, liquidity=liq), 0.1, 1e-12)


def test_breakeven_rejects_a_negative_edge():
    with pytest.raises(ValueError):
        breakeven_participation(D(-1), model=MODEL, liquidity=LIQ)


def test_zero_impact_coefficient_has_no_breakeven():
    free = CostModel(impact=ImpactModel(coefficient=D(0)))
    assert breakeven_participation(D(20), model=free, liquidity=LIQ) is None


# -- the two modules together ------------------------------------------------


def test_costs_move_a_sharpe_ratio_in_one_direction():
    """The property that matters: applying a cost never improves the result."""
    from mdnorm import sharpe_ratio

    gross = [D("0.004"), D("-0.002"), D("0.006"), D("0.001"), D("-0.005"),
             D("0.003"), D("0.002"), D("-0.001"), D("0.005"), D("0.002")]
    turn = [D("0.3")] * 10
    net = apply_costs(gross, turn, cost_bps=D(5))
    assert sharpe_ratio(net) < sharpe_ratio(gross)
