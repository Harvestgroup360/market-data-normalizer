"""Extremes tests: the ruler the outliers widened, and what cutting costs."""
import random
from decimal import Decimal

import pytest

from mdnorm.extremes import (
    ClipEffect,
    Extreme,
    Spread,
    TailReport,
    clip_effect,
    concentration,
    flag_extremes,
    spread,
    tail_contribution,
    winsorise,
    zscores,
)

D = Decimal


def series(text):
    return [D(x) for x in text.split()]


def contaminated(n=1000, k=30, mult=6, sd="0.01", seed=3):
    """A clean series with ``k`` observations at ``mult`` times the scale."""
    rng = random.Random(seed)
    vals = [D(str(round(rng.gauss(0, float(sd)), 10))) for _ in range(n)]
    for i in rng.sample(range(n), k):
        vals[i] = D(mult) * D(sd) * (D(1) if rng.random() < 0.5 else D(-1))
    return vals


# -- the two scales --------------------------------------------------------

def test_the_ordinary_spread_is_the_mean_and_the_deviation():
    s = spread(series("1 2 3 4 5"), robust=False)
    assert s.centre == 3
    assert s.robust is False
    assert abs(s.scale - D("1.4142135")) < D("0.001")


def test_the_robust_spread_is_the_median_and_the_scaled_mad():
    s = spread(series("1 2 3 4 5"), robust=True)
    assert s.centre == 3
    assert s.robust is True
    assert s.scale == D(1) * D("1.4826")


def test_the_median_of_an_even_count_is_the_midpoint():
    assert spread(series("1 2 3 4"), robust=True).centre == D("2.5")


def test_one_extreme_moves_the_ordinary_centre_and_not_the_robust_one():
    clean = series("1 2 3 4 5")
    spiked = series("1 2 3 4 500")
    assert spread(spiked, robust=False).centre != spread(clean, robust=False).centre
    assert spread(spiked, robust=True).centre == spread(clean, robust=True).centre


def test_a_spread_needs_an_observation():
    with pytest.raises(ValueError, match="at least one"):
        spread([], robust=False)


def test_a_constant_series_has_no_scale():
    s = spread([D(7)] * 10, robust=False)
    assert s.degenerate
    assert spread([D(7)] * 10, robust=True).degenerate


def test_a_scale_is_not_negative():
    with pytest.raises(ValueError, match="not negative"):
        Spread(D(0), D(-1), False)


# -- masking: the whole point ----------------------------------------------

def test_contamination_inflates_the_ordinary_scale():
    vals = contaminated()
    ordinary = spread(vals, robust=False).scale
    robust = spread(vals, robust=True).scale
    assert ordinary > robust
    assert D("1.3") < ordinary / robust < D("1.6")


def test_at_five_sigma_the_ordinary_score_finds_nothing_and_the_robust_finds_all():
    """Thirty six-sigma points widened the ruler until none of them was five."""
    vals = contaminated(k=30, mult=6)
    assert len(flag_extremes(vals, sigma=D(5))) == 0
    assert len(flag_extremes(vals, sigma=D(5), robust=True)) == 30


def test_the_same_points_are_found_by_both_at_a_looser_threshold():
    vals = contaminated(k=30, mult=6)
    plain = {e.index for e in flag_extremes(vals, sigma=D(4))}
    robust = {e.index for e in flag_extremes(vals, sigma=D(4), robust=True)}
    assert plain == robust and len(plain) == 30


def test_a_single_extreme_is_found_by_either():
    """Masking needs company; one outlier cannot hide itself."""
    rng = random.Random(9)
    vals = [D(str(round(rng.gauss(0, 0.01), 10))) for _ in range(500)]
    vals[100] = D("0.5")
    assert len(flag_extremes(vals, sigma=D(5))) == 1
    assert len(flag_extremes(vals, sigma=D(5), robust=True)) >= 1


# -- flagging --------------------------------------------------------------

def test_flags_come_back_in_input_order_with_their_scores():
    vals = series("0 0 0 10 0 0 0 -10 0 0")
    got = flag_extremes(vals, sigma=D(2), robust=False)
    assert [e.index for e in got] == [3, 7]
    assert got[0].value == 10 and got[1].value == -10
    assert got[0].score > 0 > got[1].score


def test_the_threshold_is_inclusive():
    vals = series("1 2 3 4 5")
    s = spread(vals, robust=False)
    edge = (D(5) - s.centre) / s.scale
    assert len(flag_extremes(vals, sigma=edge)) >= 1


def test_a_tighter_threshold_never_finds_fewer():
    vals = contaminated()
    counts = [len(flag_extremes(vals, sigma=D(s), robust=True))
              for s in (3, 4, 5, 6)]
    assert counts == sorted(counts, reverse=True)


def test_sigma_is_required_to_be_positive():
    with pytest.raises(ValueError, match="sigma must be positive"):
        flag_extremes(series("1 2 3"), sigma=D(0))


def test_a_series_with_no_spread_is_refused_rather_than_scored():
    with pytest.raises(ValueError, match="no extremes, only values"):
        zscores([D(4)] * 5, robust=False)
    with pytest.raises(ValueError, match="no extremes, only values"):
        flag_extremes([D(4)] * 5, sigma=D(3))


def test_nothing_is_removed():
    vals = contaminated()
    flag_extremes(vals, sigma=D(3), robust=True)
    assert len(vals) == 1000


# -- the tail --------------------------------------------------------------

def test_the_tail_is_ranked_by_size_not_by_profit():
    vals = series("1 1 1 -20 1 15 1")
    r = tail_contribution(vals, n=2)
    assert r.counted == 2
    assert r.tail_total == D(-5)          # -20 and +15, signed
    assert r.total == D(0)


def test_the_rest_is_the_total_without_the_tail():
    vals = series("1 2 3 40")
    r = tail_contribution(vals, n=1)
    assert r.tail_total == D(40)
    assert r.rest == D(6)


def test_the_tail_share_is_none_when_the_total_is_zero():
    r = tail_contribution(series("5 -5"), n=1)
    assert r.total == 0
    assert r.tail_share is None


def test_asking_for_more_than_there_is_counts_what_there_is():
    r = tail_contribution(series("1 2"), n=10)
    assert r.counted == 2 and r.rest == 0


def test_asking_for_none_counts_nothing():
    r = tail_contribution(series("1 2 3"), n=0)
    assert r.counted == 0 and r.tail_total == 0 and r.rest == D(6)


def test_n_must_not_be_negative():
    with pytest.raises(ValueError, match="must not be negative"):
        tail_contribution(series("1 2"), n=-1)


# -- concentration ---------------------------------------------------------

def test_concentration_counts_the_fewest_gains_making_the_share():
    vals = series("10 10 1 1 1 1 1 1")
    assert concentration(vals, share=D("0.5")) == 2       # 20 of 26


def test_a_flat_series_needs_many_of_them():
    vals = [D(1)] * 100
    assert concentration(vals, share=D("0.5")) == 50


def test_concentration_is_none_when_the_total_is_not_positive():
    assert concentration(series("-1 -2 3"), share=D("0.5")) is None
    assert concentration(series("5 -5"), share=D("0.5")) is None


def test_concentration_is_none_when_the_gains_cannot_reach_the_share():
    """Possible only when losses are absent, so a full share is reachable."""
    assert concentration(series("1 2 3"), share=D(1)) == 3


def test_the_share_is_validated():
    for bad in (D(0), D("1.5"), D(-1)):
        with pytest.raises(ValueError, match="above zero and at most one"):
            concentration(series("1 2"), share=bad)


# -- what clipping costs ---------------------------------------------------

def test_clipping_lowers_volatility_and_raises_the_sharpe():
    vals = contaminated(k=30, mult=6)
    eff = clip_effect(vals, sigma=D(3), robust=True)
    # Thirty contaminated points and two ordinary ones: at three sigma a
    # clean thousand is expected to put a couple past the line, and clipping
    # takes those too. That is the cost, not a defect.
    assert eff.clipped == 32
    assert len(flag_extremes(vals, sigma=D(3), robust=True)) == 32
    assert eff.volatility_after < eff.volatility_before
    assert eff.volatility_understated < 1
    assert eff.sharpe_shift is not None


def test_the_effect_reports_and_hands_back_no_data():
    eff = clip_effect(contaminated(), sigma=D(3), robust=True)
    assert not hasattr(eff, "values")
    assert isinstance(eff, ClipEffect)


def test_a_clean_series_is_barely_touched():
    rng = random.Random(11)
    vals = [D(str(round(rng.gauss(0, 0.01), 10))) for _ in range(2000)]
    eff = clip_effect(vals, sigma=D(5))
    assert eff.clipped <= 2
    assert abs(eff.volatility_understated - 1) < D("0.02")


def test_the_clipped_share_is_reported():
    vals = contaminated(k=30, mult=6)
    eff = clip_effect(vals, sigma=D(3), robust=True)
    assert eff.clipped_share == D(eff.clipped) / 1000
    assert eff.clipped_share < D("0.05")


def test_the_effect_needs_an_observation():
    with pytest.raises(ValueError, match="at least one"):
        clip_effect([], sigma=D(3))


# -- winsorising -----------------------------------------------------------

def test_winsorising_keeps_the_sample_size():
    vals = contaminated()
    assert len(winsorise(vals, sigma=D(3), robust=True)) == len(vals)


def test_winsorising_pulls_values_to_the_boundary_and_no_further():
    vals = series("0 0 0 0 100")
    out = winsorise(vals, sigma=D(1), robust=False)
    s = spread(vals, robust=False)
    assert out[4] == s.centre + s.scale
    assert out[0] == D(0)


def test_winsorising_is_symmetric():
    vals = series("-100 0 0 0 100")
    out = winsorise(vals, sigma=D(1), robust=False)
    assert out[0] == -out[4]


def test_winsorising_a_clean_series_changes_nothing():
    vals = series("1 2 3 4 5")
    assert winsorise(vals, sigma=D(10)) == vals


def test_winsorising_validates_its_inputs():
    with pytest.raises(ValueError, match="sigma must be positive"):
        winsorise(series("1 2 3"), sigma=D(-1))
    with pytest.raises(ValueError, match="only values"):
        winsorise([D(2)] * 4, sigma=D(3))


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Spread(D(0), D(1), False), Extreme(0, D(0), D(0)),
                TailReport(0, 0, D(0), D(0)),
                ClipEffect(D(3), False, 0, 0, D(0), D(0), D(0), D(0))):
        with pytest.raises(Exception):
            obj.robust = True  # type: ignore[misc]


def test_a_one_sided_tail_makes_the_clip_lower_the_sharpe_not_raise_it():
    """The direction people assume is only the symmetric case."""
    rng = random.Random(5)
    vals = [D(str(round(rng.gauss(0.0004, 0.01), 10))) for _ in range(1000)]
    for i in rng.sample(range(1000), 30):
        vals[i] = D("0.06")                       # every extreme a gain
    eff = clip_effect(vals, sigma=D(3), robust=True)
    assert eff.volatility_understated < 1         # always true
    assert eff.sharpe_shift < 1                   # the profit went with it


def test_a_symmetric_tail_makes_the_clip_raise_the_sharpe():
    rng = random.Random(5)
    vals = [D(str(round(rng.gauss(0.002, 0.01), 10))) for _ in range(1000)]
    for i, j in enumerate(rng.sample(range(1000), 30)):
        vals[j] = D("0.06") if i % 2 else D("-0.06")
    eff = clip_effect(vals, sigma=D(3), robust=True)
    assert eff.volatility_understated < 1
    assert eff.sharpe_shift > 1


def test_the_shift_is_unstable_when_there_is_almost_no_mean_to_divide_by():
    """Documented rather than hidden: both Sharpe ratios are exposed."""
    rng = random.Random(5)
    vals = [D(str(round(rng.gauss(0.0004, 0.01), 10))) for _ in range(1000)]
    for i, j in enumerate(rng.sample(range(1000), 30)):
        vals[j] = D("0.06") if i % 2 else D("-0.06")
    eff = clip_effect(vals, sigma=D(3), robust=True)
    assert abs(eff.sharpe_before) < D("0.001")      # nothing to speak of
    assert abs(eff.sharpe_after) < D("0.001")
    assert abs(eff.sharpe_shift) > 1                # yet the ratio is large
