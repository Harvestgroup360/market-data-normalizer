"""Compounding tests: the average, the rate that compounds, and the gap."""
import random
from decimal import Decimal

import pytest

from mdnorm.compounding import (
    Annualised,
    CompoundReport,
    Convention,
    annualise_return,
    approximate_drag,
    arithmetic_mean,
    compound,
    compound_report,
    geometric_mean,
    leverage_drag,
    naive_total,
    to_log,
    to_simple,
    variance_drag,
)

D = Decimal
SIMPLE = Convention.SIMPLE
LOG = Convention.LOG

MONTHLY = [D(s) for s in ("0.05 -0.03 0.04 0.02 -0.02 0.06 "
                          "-0.04 0.03 0.01 0.02 -0.01 -0.01").split()]


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the means -------------------------------------------------------------

def test_the_arithmetic_mean_is_the_plain_average():
    assert arithmetic_mean(MONTHLY) == D("0.01")


def test_the_geometric_mean_compounds_to_the_total():
    """The defining property, checked rather than assumed."""
    g = geometric_mean(MONTHLY, convention=SIMPLE)
    total = compound(MONTHLY, convention=SIMPLE)
    assert about((D(1) + g) ** 12 - 1, total, "1e-20")


def test_the_geometric_mean_never_exceeds_the_arithmetic_one():
    """An inequality, not a tendency: checked on a hundred random series."""
    rng = random.Random(17)
    for _ in range(100):
        series = [D(str(round(rng.gauss(0.002, 0.05), 10)))
                  for _ in range(24)]
        assert (geometric_mean(series, convention=SIMPLE)
                <= arithmetic_mean(series))


def test_a_constant_series_has_no_drag():
    flat = [D("0.01")] * 10
    assert variance_drag(flat, convention=SIMPLE) == 0
    assert about(geometric_mean(flat, convention=SIMPLE), "0.01", "1e-20")


def test_a_single_observation_is_its_own_geometric_mean():
    assert about(geometric_mean([D("0.07")], convention=SIMPLE), "0.07",
                 "1e-20")


def test_losing_everything_is_allowed_and_gives_a_rate_of_minus_one():
    """An account can reach zero; it cannot go past it."""
    assert geometric_mean([D("-1"), D("0.5")], convention=SIMPLE) == -1
    assert compound([D("-1"), D("0.5")], convention=SIMPLE) == -1


# -- log returns -----------------------------------------------------------

def test_log_returns_compound_by_addition():
    logs = to_log(MONTHLY)
    assert about(compound(logs, convention=LOG),
                 compound(MONTHLY, convention=SIMPLE), "1e-20")


def test_the_two_conventions_agree_on_the_geometric_mean():
    assert about(geometric_mean(to_log(MONTHLY), convention=LOG),
                 geometric_mean(MONTHLY, convention=SIMPLE), "1e-20")


def test_the_conversion_round_trips():
    for a, b in zip(to_simple(to_log(MONTHLY)), MONTHLY):
        assert about(a, b, "1e-20")


def test_a_log_return_can_be_arbitrarily_negative():
    """Minus one is a total loss in simple terms and ordinary in log terms."""
    assert compound([D("-2")], convention=LOG) < D("-0.8")


# -- the drag --------------------------------------------------------------

def test_the_drag_is_the_difference_between_the_means():
    rep = compound_report(MONTHLY, convention=SIMPLE)
    assert rep.drag == rep.arithmetic - rep.geometric
    assert rep.drag > 0


def test_the_drag_grows_with_volatility():
    """The flattery grows with the risk, which is the wrong way round."""
    got = []
    for vol in ("0.01", "0.05", "0.10", "0.20"):
        rng = random.Random(5)
        series = [D(str(round(rng.gauss(0.005, float(vol)), 10)))
                  for _ in range(2000)]
        got.append(variance_drag(series, convention=SIMPLE))
    assert got == sorted(got)


def test_the_approximation_is_close_on_small_returns_and_not_on_large():
    rng = random.Random(5)
    small = [D(str(round(rng.gauss(0.005, 0.01), 10))) for _ in range(2000)]
    rng = random.Random(5)
    large = [D(str(round(rng.gauss(0.005, 0.20), 10))) for _ in range(2000)]
    close = variance_drag(small, convention=SIMPLE) / approximate_drag(small)
    far = variance_drag(large, convention=SIMPLE) / approximate_drag(large)
    assert abs(close - 1) < D("0.01")
    assert abs(far - 1) > D("0.03")


def test_the_report_carries_the_approximation_beside_the_exact_figure():
    rep = compound_report(MONTHLY, convention=SIMPLE)
    assert about(rep.approximate_drag,
                 rep.volatility * rep.volatility / 2, "1e-20")


# -- leverage --------------------------------------------------------------

def test_leverage_multiplies_the_drag_by_roughly_its_square():
    rng = random.Random(5)
    series = [D(str(round(rng.gauss(0.0005, 0.012), 10))) for _ in range(1000)]
    one = leverage_drag(series, multiple=D(1), convention=SIMPLE)
    two = leverage_drag(series, multiple=D(2), convention=SIMPLE)
    three = leverage_drag(series, multiple=D(3), convention=SIMPLE)
    assert abs(two / one - 4) < D("0.1")
    assert abs(three / one - 9) < D("0.4")


def test_unit_leverage_is_the_plain_drag():
    assert (leverage_drag(MONTHLY, multiple=D(1), convention=SIMPLE)
            == variance_drag(MONTHLY, convention=SIMPLE))


def test_a_non_positive_multiple_is_refused():
    with pytest.raises(ValueError, match="multiple is positive"):
        leverage_drag(MONTHLY, multiple=D(0), convention=SIMPLE)


def test_leverage_that_wipes_the_account_out_is_refused():
    """Five times a twenty-five per cent loss is more than everything."""
    with pytest.raises(ValueError, match="more than everything"):
        leverage_drag([D("0.1"), D("-0.25")], multiple=D(5),
                      convention=SIMPLE)


# -- annualising -----------------------------------------------------------

def test_annualising_compounds_rather_than_multiplying():
    assert about(annualise_return(D("0.01"), periods_per_year=12),
                 "0.126825030131969720661201", "1e-15")


def test_the_naive_annual_never_undershoots_the_actual_one():
    rng = random.Random(23)
    for _ in range(50):
        series = [D(str(round(rng.gauss(0.004, 0.06), 10)))
                  for _ in range(12)]
        year = compound_report(series, convention=SIMPLE).annualised(
            periods_per_year=12)
        assert year.overstatement >= 0


def test_the_worked_example():
    rep = compound_report(MONTHLY, convention=SIMPLE)
    year = rep.annualised(periods_per_year=12)
    assert about(rep.arithmetic, "0.01", "1e-12")
    assert about(rep.geometric, "0.009529", "0.000001")
    assert about(year.naive, "0.126825", "0.000001")
    assert about(year.actual, "0.120539", "0.000001")
    assert about(year.overstatement, "0.006286", "0.000001")
    assert about(year.overstatement_share, "0.0522", "0.0001")


def test_the_actual_annual_equals_the_compounded_total_on_a_full_year():
    rep = compound_report(MONTHLY, convention=SIMPLE)
    year = rep.annualised(periods_per_year=12)
    assert about(year.actual, rep.actual_total, "1e-18")


def test_a_zero_actual_return_has_no_share():
    year = Annualised(periods_per_year=12, naive=D("0.05"), actual=D(0))
    assert year.overstatement == D("0.05")
    assert year.overstatement_share is None


def test_annualise_refuses_a_non_positive_calendar():
    with pytest.raises(ValueError, match="must be positive"):
        annualise_return(D("0.01"), periods_per_year=0)


def test_annualise_refuses_a_rate_that_empties_the_account():
    with pytest.raises(ValueError, match="already empty"):
        annualise_return(D("-1"), periods_per_year=12)


# -- the claim an earlier draft got wrong ----------------------------------

def test_the_sum_of_returns_can_land_either_side_of_the_truth():
    """Compounding adds the cross-products, so the sign is not fixed."""
    gaining = compound_report(MONTHLY, convention=SIMPLE)
    assert gaining.total_gap > 0          # the account beat the sum

    volatile = compound_report([D("0.5"), D("-0.4"), D("0.5"), D("-0.4")],
                               convention=SIMPLE)
    assert volatile.total_gap < 0         # and here it trailed it


def test_the_naive_total_is_the_sum():
    assert naive_total(MONTHLY) == sum(MONTHLY, D(0))


# -- refusals --------------------------------------------------------------

def test_a_simple_return_below_minus_one_is_refused_by_index():
    with pytest.raises(ValueError, match="observation 1 is -1.5"):
        compound([D("0.1"), D("-1.5")], convention=SIMPLE)


def test_exactly_minus_one_is_not_refused():
    assert compound([D("-1")], convention=SIMPLE) == -1


def test_the_message_suggests_the_other_convention():
    """A log series fed in as simple is the error this catches."""
    with pytest.raises(ValueError, match="Convention.LOG"):
        compound([D("-1.2")], convention=SIMPLE)


def test_an_empty_series_is_refused_everywhere():
    for call in (lambda: arithmetic_mean([]),
                 lambda: naive_total([]),
                 lambda: approximate_drag([]),
                 lambda: to_simple([]),
                 lambda: compound([], convention=SIMPLE),
                 lambda: geometric_mean([], convention=LOG)):
        with pytest.raises(ValueError, match="at least one observation"):
            call()


# -- the report ------------------------------------------------------------

def test_the_report_carries_the_convention_it_was_given():
    assert compound_report(MONTHLY, convention=SIMPLE).convention is SIMPLE
    assert compound_report(to_log(MONTHLY),
                           convention=LOG).convention is LOG


def test_the_two_conventions_produce_the_same_report_figures():
    a = compound_report(MONTHLY, convention=SIMPLE)
    b = compound_report(to_log(MONTHLY), convention=LOG)
    assert about(a.geometric, b.geometric, "1e-20")
    assert about(a.actual_total, b.actual_total, "1e-20")


def test_frozen_dataclasses():
    for obj in (compound_report(MONTHLY, convention=SIMPLE),
                Annualised(12, D(0), D(0))):
        with pytest.raises(Exception):
            obj.naive = D(9)  # type: ignore[misc]
