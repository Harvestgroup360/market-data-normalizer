"""Fund fee tests: a backtest reports what the strategy earned."""
import random
from decimal import Decimal

import pytest

from mdnorm.fundfees import (
    FeeComparison,
    FeeResult,
    FeeSchedule,
    apply_fees,
    compare_fees,
)

D = Decimal


def gross(n=120, seed=20260921):
    """Ten years of monthly gross returns, about eight per cent a year."""
    rng = random.Random(seed)
    return [D(str(round(rng.gauss(0.0095, 0.035), 10))) for _ in range(n)]


def sched(m="0.02", i="0.20", every=12, hwm=True, hurdle=None, ppy=12):
    return FeeSchedule(management=D(m), incentive=D(i), periods_per_year=ppy,
                       crystallise_every=every, high_water_mark=hwm,
                       hurdle=hurdle)


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the schedule refuses what it cannot mean ------------------------------

def test_a_negative_management_fee_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        sched(m="-0.01")


def test_an_incentive_share_of_one_or_more_is_refused():
    for i in ("1", "1.2", "-0.1"):
        with pytest.raises(ValueError, match=r"lies in \[0, 1\)"):
            sched(i=i)


def test_a_non_positive_calendar_is_refused():
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        sched(ppy=0)


def test_a_crystallisation_period_below_one_is_refused():
    with pytest.raises(ValueError, match="at least one period"):
        sched(every=0)


def test_a_hurdle_without_a_high_water_mark_is_refused():
    with pytest.raises(ValueError, match="has none"):
        sched(hwm=False, hurdle=(D("0.001"),))


def test_every_field_but_the_hurdle_is_required():
    with pytest.raises(TypeError):
        FeeSchedule(management=D("0.02"), incentive=D("0.2"))  # type: ignore[call-arg]


# -- arithmetic by hand ----------------------------------------------------

def test_no_fees_leaves_the_returns_alone():
    g = gross(n=36)
    res = apply_fees(g, sched(m="0", i="0"))
    for got, want in zip(res.net_returns, g):
        assert about(got, want, "1e-30")
    assert res.management_paid == 0 and res.incentive_paid == 0
    assert about(res.fee_share_of_profit, 0, "1e-30")


def test_the_management_fee_is_charged_on_start_of_period_value():
    """Twelve per cent a year is one per cent a month, on a flat month."""
    res = apply_fees([D(0)], sched(m="0.12", i="0"))
    assert res.net_returns[0] == D("-0.01")
    assert res.management_paid == D("0.01")


def test_the_incentive_fee_takes_its_share_of_the_gain():
    """Up ten per cent, twenty per cent of it goes: the investor keeps eight."""
    res = apply_fees([D("0.1")], sched(m="0", i="0.2", every=1))
    assert res.incentive_paid == D("0.02")
    assert res.net_returns[0] == D("0.08")


def test_a_loss_pays_no_incentive_fee():
    res = apply_fees([D("-0.05")], sched(m="0", i="0.2", every=1))
    assert res.incentive_paid == 0
    assert res.net_returns[0] == D("-0.05")


def test_the_high_water_mark_remembers_a_loss():
    """Down, back up to the start, then up again: only the last leg pays."""
    path = [D("0.25"), D("-0.2"), D("0.25")]  # 1.25, 1.0, 1.25 gross
    res = apply_fees(path, sched(m="0", i="0.2", every=1))
    # first month pays on 0.25, the mark is 1.2 after the fee. Month two falls
    # to 0.96; month three rises to 1.2, exactly the mark: nothing is due.
    assert about(res.incentive_paid, "0.05", "1e-30")
    assert res.charged == 1


def test_without_a_mark_the_recovery_is_paid_for_again():
    path = [D("0.25"), D("-0.2"), D("0.25")]
    with_mark = apply_fees(path, sched(m="0", i="0.2", every=1))
    without = apply_fees(path, sched(m="0", i="0.2", every=1, hwm=False))
    assert without.incentive_paid > with_mark.incentive_paid
    assert without.charged == 2


def test_the_final_observation_always_crystallises():
    """An accrued fee is not left off the net figure."""
    res = apply_fees([D("0.05")] * 5, sched(m="0", i="0.2", every=12))
    assert res.crystallisations == 1
    assert res.incentive_paid > 0


def test_a_hurdle_raises_the_mark_before_any_fee_is_due():
    g = [D("0.01")] * 12
    low = apply_fees(g, sched(m="0", i="0.2", every=12))
    hurdled = apply_fees(g, sched(m="0", i="0.2", every=12,
                                  hurdle=tuple([D("0.005")] * 12)))
    assert hurdled.incentive_paid < low.incentive_paid


def test_a_hurdle_above_the_return_means_no_incentive_fee():
    g = [D("0.01")] * 12
    res = apply_fees(g, sched(m="0", i="0.2", every=12,
                              hurdle=tuple([D("0.02")] * 12)))
    assert res.incentive_paid == 0


# -- refusals over plausible answers ---------------------------------------

def test_an_empty_series_is_refused():
    with pytest.raises(ValueError, match="at least one period"):
        apply_fees([], sched())


def test_a_hurdle_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="hurdle has 2 periods"):
        apply_fees(gross(n=5), sched(hurdle=(D(0), D(0))))


def test_a_return_that_empties_the_account_is_refused_by_index():
    with pytest.raises(ValueError, match="gross return 1 is -1"):
        apply_fees([D("0.01"), D("-1")], sched())


def test_an_annual_rate_passed_as_monthly_is_caught():
    """Twelve hundred per cent a year empties the account in a month."""
    with pytest.raises(ArithmeticError, match="per-period one does this"):
        apply_fees([D(0)] * 3, sched(m="12", i="0", ppy=1))


# -- the headline rate is not the share of profit --------------------------

def test_the_share_of_profit_exceeds_the_incentive_rate_even_with_no_management_fee():
    """No gain is returned after a crystallisation, so twenty takes more than twenty."""
    res = apply_fees(gross(), sched(m="0", i="0.2"))
    assert res.fee_share_of_profit > D("0.2")


def test_the_share_is_none_on_a_losing_strategy():
    res = apply_fees([D("-0.01")] * 24, sched())
    assert res.fee_share_of_profit is None
    assert res.management_paid > 0


def test_more_frequent_crystallisation_takes_more():
    g = gross()
    shares = [apply_fees(g, sched(every=e)).fee_share_of_profit
              for e in (12, 3, 1)]
    assert shares == sorted(shares)


def test_dropping_the_high_water_mark_takes_more():
    g = gross()
    assert apply_fees(g, sched(hwm=False)).net_total < \
        apply_fees(g, sched(hwm=True)).net_total


def test_the_incentive_fee_trims_volatility_so_the_sharpe_falls_less_than_the_return():
    res = apply_fees(gross(), sched(m="0", i="0.2", every=1))
    return_fall = 1 - res.net_total / res.gross_total
    sharpe_fall = 1 - res.sharpe(net=True, ddof=1) / res.sharpe(net=False, ddof=1)
    assert sharpe_fall < return_fall


def test_the_incentive_share_of_fees_is_none_when_nothing_was_paid():
    assert apply_fees(gross(n=12), sched(m="0", i="0")).incentive_share_of_fees is None


# -- comparing schedules --------------------------------------------------

def test_every_row_of_a_comparison_has_the_same_gross_total():
    c = compare_fees(gross(), {"a": sched(), "b": sched(every=1),
                               "c": sched(m="0.01", i="0.1")})
    totals = {r.gross_total for r in c.results}
    assert len(totals) == 1


def test_an_empty_comparison_is_refused():
    with pytest.raises(ValueError, match="at least one fee schedule"):
        compare_fees(gross(n=12), {})


# -- the worked example ---------------------------------------------------

def test_the_worked_example():
    g = gross()
    res = apply_fees(g, sched())
    assert res.periods == 120
    assert about(res.gross_total, "1.169200", "0.000001")
    assert about(res.net_total, "0.554361", "0.000001")
    assert about(res.fee_share_of_profit, "0.525863", "0.000001")
    assert res.crystallisations == 10
    assert res.charged == 7


def test_the_contract_moves_the_answer_on_identical_returns():
    g = gross()
    c = compare_fees(g, {
        "2/20 annual HWM": sched(),
        "2/20 monthly no HWM": sched(every=1, hwm=False),
        "0/20 annual HWM": sched(m="0"),
    })
    assert about(c.row("2/20 monthly no HWM").fee_share_of_profit,
                 "0.832859", "0.000001")
    assert about(c.row("0/20 annual HWM").fee_share_of_profit,
                 "0.284809", "0.000001")
    assert about(c.net_spread, "0.640780", "0.000001")


# -- types ----------------------------------------------------------------

def test_frozen_dataclasses():
    res = apply_fees(gross(n=12), sched())
    for obj in (sched(), res, compare_fees(gross(n=12), {"a": sched()})):
        with pytest.raises(Exception):
            obj.periods = 9  # type: ignore[misc]


def test_the_results_are_the_documented_types():
    c = compare_fees(gross(n=12), {"a": sched()})
    assert isinstance(c, FeeComparison)
    assert isinstance(c.row("a"), FeeResult)
