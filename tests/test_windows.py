"""Window tests: the sweep, and the trial count nobody was writing down."""
import random
from decimal import Decimal

import pytest

from mdnorm.metrics import sharpe_ratio
from mdnorm.windows import (
    WindowResult,
    SensitivityReport,
    Window,
    WindowKind,
    sweep,
    expanding_windows,
    rolling_windows,
    trimmed_ends,
    trimmed_starts,
)

D = Decimal


def series(text):
    return [D(x) for x in text.split()]


def drifting(n=1000, mu=0.0005, sd=0.01, seed=17):
    rng = random.Random(seed)
    return [D(str(round(rng.gauss(mu, sd), 10))) for _ in range(n)]


def total(values):
    return sum(values, D(0)) if values else None


# -- the window itself -----------------------------------------------------

def test_a_window_is_half_open():
    w = Window(2, 5)
    assert w.observations == 3
    assert list(w.slice(series("0 1 2 3 4 5 6"))) == series("2 3 4")


def test_a_window_ends_after_it_starts():
    with pytest.raises(ValueError, match="ends after it starts"):
        Window(5, 5)
    with pytest.raises(ValueError, match="ends after it starts"):
        Window(5, 2)


def test_a_window_does_not_start_before_the_beginning():
    with pytest.raises(ValueError, match="at or after the beginning"):
        Window(-1, 5)


# -- generating them -------------------------------------------------------

def test_rolling_windows_advance_by_the_step():
    got = rolling_windows(10, length=4, step=3)
    assert [(w.start, w.end) for w in got] == [(0, 4), (3, 7), (6, 10)]


def test_rolling_windows_never_reach_past_the_end():
    for w in rolling_windows(10, length=4, step=1):
        assert w.end <= 10


def test_a_length_longer_than_the_series_produces_nothing():
    assert rolling_windows(5, length=6, step=1) == []


def test_expanding_windows_all_start_at_zero_and_end_on_the_whole():
    got = expanding_windows(10, minimum=4, step=3)
    assert [(w.start, w.end) for w in got] == [(0, 4), (0, 7), (0, 10)]
    assert got[-1].end == 10


def test_an_expanding_sweep_always_includes_the_full_sample():
    """Even when the step does not land on it."""
    got = expanding_windows(10, minimum=4, step=4)
    assert [(w.start, w.end) for w in got] == [(0, 4), (0, 8), (0, 10)]


def test_trimmed_starts_drop_the_beginning_and_keep_the_end():
    got = trimmed_starts(10, step=3)
    assert [(w.start, w.end) for w in got] == [(3, 10), (6, 10), (9, 10)]


def test_trimmed_starts_can_be_limited():
    got = trimmed_starts(100, step=10, count=3)
    assert [w.start for w in got] == [10, 20, 30]


def test_trimmed_ends_keep_the_beginning_and_stop_earlier():
    got = trimmed_ends(10, step=3)
    assert [(w.start, w.end) for w in got] == [(0, 7), (0, 4), (0, 1)]


def test_trimmed_ends_can_be_limited():
    got = trimmed_ends(100, step=10, count=2)
    assert [w.end for w in got] == [90, 80]


def test_the_generators_validate_their_arguments():
    with pytest.raises(ValueError, match="must be positive"):
        rolling_windows(10, length=0, step=1)
    with pytest.raises(ValueError, match="must be positive"):
        expanding_windows(10, minimum=4, step=0)
    with pytest.raises(ValueError, match="step must be positive"):
        trimmed_starts(10, step=0)
    with pytest.raises(ValueError, match="must not be negative"):
        trimmed_ends(10, step=2, count=-1)
    with pytest.raises(ValueError, match="at least one observation"):
        rolling_windows(0, length=1, step=1)


# -- evaluating ------------------------------------------------------------

def test_the_metric_runs_on_every_window_and_on_the_whole():
    vals = series("1 2 3 4 5 6")
    rep = sweep(vals, rolling_windows(6, length=3, step=3), total)
    assert [s.value for s in rep.samples] == [D(6), D(15)]
    assert rep.full == D(21)


def test_a_window_the_metric_cannot_answer_for_becomes_none():
    """A Sharpe needs two observations; a window of one is not an error."""
    vals = drifting(n=10)
    rep = sweep(vals, [Window(0, 1), Window(0, 10)], sharpe_ratio)
    assert rep.samples[0].value is None
    assert rep.samples[1].value is not None


def test_none_values_are_left_out_of_the_summaries_not_counted_as_zero():
    rep = SensitivityReport(
        kind=WindowKind.EXPLICIT,
        samples=(WindowResult(Window(0, 1), None), WindowResult(Window(0, 2), D(4)),
                 WindowResult(Window(0, 3), D(6))),
        full=D(5))
    assert rep.values == (D(4), D(6))
    assert rep.lowest == D(4) and rep.highest == D(6)
    assert rep.median == D(5)


def test_a_window_past_the_end_raises_rather_than_being_shortened():
    with pytest.raises(ValueError, match="reaches past"):
        sweep(series("1 2 3"), [Window(0, 9)], total)


def test_the_kind_is_carried_through():
    rep = sweep(drifting(n=50), trimmed_starts(50, step=10),
                   sharpe_ratio, kind=WindowKind.TRIMMED_START)
    assert rep.kind is WindowKind.TRIMMED_START


# -- the summaries ---------------------------------------------------------

def test_the_spread_is_the_range_of_the_metric():
    rep = sweep(series("1 2 3 4 5 6"),
                   rolling_windows(6, length=2, step=2), total)
    assert [s.value for s in rep.samples] == [D(3), D(7), D(11)]
    assert rep.lowest == D(3) and rep.highest == D(11)
    assert rep.spread == D(8)
    assert rep.median == D(7)


def test_the_median_of_an_even_count_is_the_midpoint():
    rep = sweep(series("1 2 3 4"), rolling_windows(4, length=1, step=1),
                   total)
    assert rep.median == D("2.5")


def test_a_sign_change_is_reported():
    rep = sweep(series("5 -9 5 5"), rolling_windows(4, length=2, step=2),
                   total)
    assert [s.value for s in rep.samples] == [D(-4), D(10)]
    assert rep.changes_sign
    assert rep.positive == 1
    assert rep.share_positive == D("0.5")


def test_a_result_positive_everywhere_does_not_change_sign():
    rep = sweep(series("1 2 3 4"), rolling_windows(4, length=2, step=2),
                   total)
    assert not rep.changes_sign
    assert rep.share_positive == 1


def test_the_shortest_and_longest_windows_are_reported():
    rep = sweep(drifting(n=100), expanding_windows(100, minimum=20, step=40),
                   sharpe_ratio)
    assert rep.shortest == 20
    assert rep.longest == 100


def test_everything_is_none_when_no_window_produced_a_value():
    rep = sweep(drifting(n=10), [Window(0, 1), Window(1, 2)], sharpe_ratio)
    assert rep.values == ()
    assert rep.lowest is None and rep.highest is None
    assert rep.median is None and rep.spread is None
    assert rep.share_positive is None
    assert not rep.changes_sign


# -- the trial count, which is the point -----------------------------------

def test_the_trial_count_includes_the_full_sample():
    rep = sweep(drifting(n=200), trimmed_starts(200, step=20, count=5),
                   sharpe_ratio)
    assert len(rep.samples) == 5
    assert rep.trials == 6


def test_windows_that_answered_nothing_are_not_trials():
    rep = sweep(drifting(n=10), [Window(0, 1), Window(0, 10)], sharpe_ratio)
    assert len(rep.samples) == 2
    assert rep.trials == 2          # one window plus the full sample


def test_the_trial_count_feeds_a_deflated_sharpe():
    """The connection the module exists to make."""
    from mdnorm.metrics import deflated_sharpe_ratio, trial_variance

    vals = drifting(n=1000)
    rep = sweep(vals, trimmed_starts(1000, step=21, count=24),
                   sharpe_ratio, kind=WindowKind.TRIMMED_START)
    assert rep.trials == 25
    var = trial_variance(list(rep.values))
    best = rep.highest
    assert best is not None
    naive = deflated_sharpe_ratio(best, observations=1000, trials=1,
                                  variance=var)
    honest = deflated_sharpe_ratio(best, observations=1000,
                                   trials=rep.trials, variance=var)
    assert honest <= naive


# -- the behaviour that motivated the module -------------------------------

def test_moving_the_start_moves_the_answer():
    vals = drifting(n=1000, mu=0.0002)
    rep = sweep(vals, trimmed_starts(1000, step=21, count=24),
                   sharpe_ratio, kind=WindowKind.TRIMMED_START)
    assert rep.spread > 0
    assert rep.lowest < rep.highest


def test_a_few_large_observations_can_carry_the_whole_difference():
    """Trim past the spike and the metric collapses: that is the finding."""
    vals = [D("0.001")] * 500
    vals[10] = D("2")
    rep = sweep(vals, trimmed_starts(500, step=20, count=3), total)
    assert rep.full == D("2.499")
    assert [s.value for s in rep.samples] == [D("0.48"), D("0.46"), D("0.44")]
    assert rep.highest < rep.full / 5


def test_an_expanding_sweep_ends_at_the_headline_figure():
    vals = drifting(n=300)
    rep = sweep(vals, expanding_windows(300, minimum=50, step=50),
                   sharpe_ratio, kind=WindowKind.EXPANDING)
    assert rep.samples[-1].value == rep.full


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Window(0, 1), WindowResult(Window(0, 1), None),
                SensitivityReport(WindowKind.EXPLICIT, (), None)):
        with pytest.raises(Exception):
            obj.start = 5  # type: ignore[misc]
