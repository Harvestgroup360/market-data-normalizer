"""Rebalance tests: how often a backtest trades is an assumption."""
import random
from decimal import Decimal

import pytest

from mdnorm.rebalance import (
    DriftReport,
    ScheduleComparison,
    ScheduleResult,
    band_rebalance,
    buy_and_hold,
    compare_schedules,
    drift_once,
    drift_path,
    drift_report,
    periodic_rebalance,
    turnover_between,
)

D = Decimal

NAMES = ["a", "b", "c", "d", "e"]
SPEC = {"a": (0.00055, 0.022), "b": (0.00030, 0.011), "c": (0.00028, 0.010),
        "d": (0.00026, 0.0095), "e": (0.00024, 0.0090)}


def make(n=1260, seed=20260919):
    """Five names, one of them markedly more volatile than the others."""
    rng = random.Random(seed)
    return [{k: D(str(round(rng.gauss(*SPEC[k]), 10))) for k in NAMES}
            for _ in range(n)]


def equal_target():
    return {k: D("0.2") for k in NAMES}


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- one period of drift ---------------------------------------------------

def test_drift_by_hand():
    """Half in a name that doubles, half in one that is flat."""
    w, pnl = drift_once({"x": D("0.5"), "y": D("0.5")},
                        {"x": D(1), "y": D(0)})
    assert pnl == D("0.5")
    assert about(w["x"], D(1) / D("1.5"), "1e-25")      # 0.6666...
    assert about(w["y"], D("0.5") / D("1.5"), "1e-25")  # 0.3333...
    assert about(w["x"] + w["y"], 1, "1e-25")


def test_a_flat_period_leaves_the_weights_alone():
    w, pnl = drift_once({"x": D("0.3"), "y": D("0.7")},
                        {"x": D(0), "y": D(0)})
    assert pnl == 0
    assert w == {"x": D("0.3"), "y": D("0.7")}


def test_the_residual_is_cash_and_earns_nothing():
    """Half invested in a name that gains ten per cent: the book gains five."""
    w, pnl = drift_once({"x": D("0.5")}, {"x": D("0.1")})
    assert pnl == D("0.05")
    assert about(w["x"], D("0.55") / D("1.05"), "1e-25")


def test_a_short_book_drifts_too():
    w, pnl = drift_once({"long": D(1), "short": D(-1)},
                        {"long": D("0.02"), "short": D("-0.01")})
    assert about(pnl, "0.03", "1e-30")
    assert w["short"] < 0


def test_a_period_that_wipes_the_book_out_is_refused():
    with pytest.raises(ArithmeticError, match="no weights after that"):
        drift_once({"x": D(1)}, {"x": D("-1.5")})


def test_the_refusal_points_at_the_log_return_conversion():
    with pytest.raises(ArithmeticError, match="mdnorm.to_simple"):
        drift_once({"x": D(1)}, {"x": D("-2")})


# -- refusals over filled gaps ---------------------------------------------

def test_a_missing_return_is_refused_rather_than_treated_as_zero():
    rets = [{"a": D("0.01"), "b": D("0.01")}, {"a": D("0.01")}]
    with pytest.raises(ValueError, match="period 1 has no return for b"):
        periodic_rebalance({"a": D("0.5"), "b": D("0.5")}, rets, every=1)


def test_the_refusal_says_why_a_gap_is_not_a_zero():
    rets = [{"a": D("0.01")}, {}]
    with pytest.raises(ValueError, match="not a zero return"):
        buy_and_hold({"a": D(1)}, rets)


def test_an_empty_portfolio_is_refused():
    with pytest.raises(ValueError, match="at least one instrument"):
        buy_and_hold({}, [{"a": D(0)}])


def test_an_empty_return_series_is_refused():
    with pytest.raises(ValueError, match="at least one period"):
        buy_and_hold({"a": D(1)}, [])


def test_a_frequency_below_one_is_refused():
    rets = make(n=10)
    for every in (0, -3):
        with pytest.raises(ValueError, match="at least one period"):
            periodic_rebalance(equal_target(), rets, every=every)


def test_a_non_positive_band_is_refused():
    rets = make(n=10)
    with pytest.raises(ValueError, match="band must be positive"):
        band_rebalance(equal_target(), rets, band=D(0))


# -- turnover --------------------------------------------------------------

def test_one_sided_turnover_is_half_the_two_sided_one():
    a = {"x": D("0.6"), "y": D("0.4")}
    b = {"x": D("0.4"), "y": D("0.6")}
    assert turnover_between(a, b, one_sided=True) == D("0.2")
    assert turnover_between(a, b, one_sided=False) == D("0.4")


def test_entering_and_leaving_the_book_both_register():
    assert turnover_between({"x": D(1)}, {"y": D(1)}, one_sided=True) == D(1)


def test_no_change_is_no_turnover():
    a = {"x": D("0.5"), "y": D("0.5")}
    assert turnover_between(a, dict(a), one_sided=True) == 0


def test_it_agrees_with_the_metrics_module_on_the_same_weight_path():
    """Two implementations of the same convention."""
    from mdnorm.metrics import turnover as metrics_turnover

    rets = make(n=200)
    res = periodic_rebalance(equal_target(), rets, every=1)
    theirs = metrics_turnover(list(res.weights))
    # metrics reads a path of held weights; ours is the trade back to target,
    # so compare the trades our path implies rather than the fields directly.
    ours = [None] + [turnover_between(res.weights[i - 1], res.weights[i],
                                      one_sided=True)
                     for i in range(1, len(res.weights))]
    assert theirs[0] is None and ours[0] is None
    for t, o in zip(theirs[1:], ours[1:]):
        assert about(t, o, "1e-30")


def test_the_first_period_is_never_counted_as_a_rebalance():
    res = periodic_rebalance(equal_target(), make(n=50), every=1)
    assert res.turnover[0] is None
    assert all(t is not None for t in res.turnover[1:])


# -- the schedules --------------------------------------------------------

def test_rebalancing_every_period_holds_the_target_exactly():
    rets = make(n=300)
    res = periodic_rebalance(equal_target(), rets, every=1)
    for row in res.weights:
        for k, v in equal_target().items():
            assert about(row[k], v, "1e-30")


def test_its_returns_are_the_weighted_average_every_period():
    rets = make(n=120)
    res = periodic_rebalance(equal_target(), rets, every=1)
    for period, got in zip(rets, res.returns):
        want = sum((D("0.2") * period[k] for k in NAMES), D(0))
        assert about(got, want, "1e-30")


def test_buy_and_hold_compounds_each_name_on_its_own():
    """A closed form the schedule machinery should reproduce exactly."""
    rets = make(n=400)
    res = buy_and_hold(equal_target(), rets)
    want = D(0)
    for k in NAMES:
        leg = D(1)
        for period in rets:
            leg *= D(1) + period[k]
        want += D("0.2") * leg
    assert about(res.total_return, want - 1, "1e-25")
    assert res.total_turnover == 0
    assert res.rebalances == 0


def test_a_frequency_longer_than_the_sample_never_rebalances():
    rets = make(n=50)
    res = periodic_rebalance(equal_target(), rets, every=500)
    assert res.rebalances == 0
    assert about(res.total_return, buy_and_hold(equal_target(), rets)
                 .total_return, "1e-30")


def test_more_often_means_more_turnover():
    rets = make()
    turns = [periodic_rebalance(equal_target(), rets, every=e).total_turnover
             for e in (1, 5, 21, 63, 252)]
    assert turns == sorted(turns, reverse=True)


def test_a_wider_band_trades_less():
    rets = make()
    tight = band_rebalance(equal_target(), rets, band=D("0.02"))
    wide = band_rebalance(equal_target(), rets, band=D("0.05"))
    assert wide.rebalances < tight.rebalances
    assert wide.total_turnover < tight.total_turnover


def test_a_band_wider_than_any_drift_never_fires():
    rets = make()
    res = band_rebalance(equal_target(), rets, band=D("0.9"))
    assert res.rebalances == 0


# -- drift ----------------------------------------------------------------

def test_an_untouched_book_wanders_and_a_rebalanced_one_does_not():
    rets = make()
    never = drift_report(equal_target(), buy_and_hold(equal_target(), rets)
                         .weights, band=D("0.02"))
    monthly = drift_report(equal_target(),
                           periodic_rebalance(equal_target(), rets, every=21)
                           .weights, band=D("0.02"))
    assert never.max_drift > monthly.max_drift
    assert never.share_outside_band > monthly.share_outside_band


def test_the_drift_report_names_the_instrument_that_led():
    rets = make()
    rep = drift_report(equal_target(),
                       buy_and_hold(equal_target(), rets).weights,
                       band=D("0.02"))
    assert rep.worst_name in NAMES


def test_a_drift_report_needs_a_band():
    rets = make(n=20)
    with pytest.raises(ValueError, match="band must be positive"):
        drift_report(equal_target(), buy_and_hold(equal_target(), rets).weights,
                     band=D("-0.01"))


def test_drift_path_starts_at_the_weights_supplied():
    rets = make(n=30)
    path = drift_path(equal_target(), rets)
    assert path[0] == equal_target()
    assert len(path) == 30


# -- comparing schedules --------------------------------------------------

def test_schedules_of_different_lengths_are_refused():
    short, long = make(n=100), make(n=200)
    with pytest.raises(ValueError, match="different numbers of periods"):
        compare_schedules({
            "a": periodic_rebalance(equal_target(), short, every=1),
            "b": periodic_rebalance(equal_target(), long, every=1),
        })


def test_an_empty_comparison_is_refused():
    with pytest.raises(ValueError, match="at least one schedule"):
        compare_schedules({})


def test_the_breakeven_is_none_when_two_schedules_traded_the_same():
    rets = make(n=100)
    a = buy_and_hold(equal_target(), rets, label="one")
    b = buy_and_hold(equal_target(), rets, label="two")
    assert compare_schedules({"one": a, "two": b}).breakeven_cost is None


def test_the_breakeven_is_negative_when_the_extra_trading_lost_money():
    """Reported with its sign, because it reads as a bargain without one."""
    rets = make()
    cmp = compare_schedules({
        "every 21": periodic_rebalance(equal_target(), rets, every=21),
        "never": buy_and_hold(equal_target(), rets),
    })
    assert cmp.most_active.label == "every 21"
    assert cmp.breakeven_cost < 0


def test_the_most_active_schedule_is_by_turnover_not_by_return():
    rets = make()
    cmp = compare_schedules({
        "every 1": periodic_rebalance(equal_target(), rets, every=1),
        "every 252": periodic_rebalance(equal_target(), rets, every=252),
        "never": buy_and_hold(equal_target(), rets),
    })
    assert cmp.most_active.label == "every 1"
    assert cmp.least_active.label == "never"


# -- the worked example ---------------------------------------------------

def test_the_worked_example():
    rets = make()
    tgt = equal_target()
    daily = periodic_rebalance(tgt, rets, every=1)
    monthly = periodic_rebalance(tgt, rets, every=21)
    annual = periodic_rebalance(tgt, rets, every=252)
    never = buy_and_hold(tgt, rets)

    assert about(daily.total_return, "0.11790065", "0.0000001")
    assert about(monthly.total_return, "0.10183339", "0.0000001")
    assert about(annual.total_return, "0.10949531", "0.0000001")
    assert about(never.total_return, "0.11346550", "0.0000001")

    assert about(daily.total_turnover, "5.62170439", "0.0000001")
    assert about(monthly.total_turnover, "1.09552663", "0.0000001")
    assert about(annual.total_turnover, "0.28684028", "0.0000001")
    assert never.total_turnover == 0

    assert daily.rebalances == 1259
    assert monthly.rebalances == 59
    assert annual.rebalances == 4


def test_the_daily_advantage_is_inside_the_cost_of_getting_it():
    """The whole argument, in one number.

    Rebalancing at every observation beats never rebalancing by 44 basis
    points over five years and turns the book 5.6 times to do it. At a cost
    of 7.89 basis points per unit of one-sided turnover the advantage is
    exactly gone, and above it the ranking reverses.
    """
    rets = make()
    tgt = equal_target()
    cmp = compare_schedules({
        "every 1": periodic_rebalance(tgt, rets, every=1),
        "never": buy_and_hold(tgt, rets),
    })
    assert about(cmp.return_spread, "0.00443514", "0.0000001")
    assert about(cmp.breakeven_cost_bps, "7.8893", "0.0001")


def test_the_frequency_moves_the_answer_without_a_pattern():
    """Five frequencies, no monotone ordering in the return.

    Turnover is monotone in the frequency and the return is not, which is the
    reason this module reports both instead of recommending one.
    """
    rets = make()
    tgt = equal_target()
    totals = [periodic_rebalance(tgt, rets, every=e).total_return
              for e in (1, 5, 21, 63, 252)]
    assert totals != sorted(totals)
    assert totals != sorted(totals, reverse=True)


def test_an_untouched_book_spends_the_sample_off_target():
    rets = make()
    rep = drift_report(equal_target(),
                       buy_and_hold(equal_target(), rets).weights,
                       band=D("0.02"))
    assert about(rep.max_drift, "0.15225272", "0.0000001")
    assert rep.worst_name == "c"
    assert rep.share_outside_band > D("0.97")


# -- types ----------------------------------------------------------------

def test_frozen_dataclasses():
    rets = make(n=40)
    res = periodic_rebalance(equal_target(), rets, every=5)
    for obj in (res, compare_schedules({"a": res}),
                drift_report(equal_target(), res.weights, band=D("0.02"))):
        with pytest.raises(Exception):
            obj.periods = 9  # type: ignore[misc]


def test_the_results_are_the_documented_types():
    rets = make(n=40)
    res = periodic_rebalance(equal_target(), rets, every=5)
    assert isinstance(res, ScheduleResult)
    assert isinstance(compare_schedules({"a": res}), ScheduleComparison)
    assert isinstance(drift_report(equal_target(), res.weights, band=D("0.02")),
                      DriftReport)
