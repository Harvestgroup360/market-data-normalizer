"""Selection tests: choosing the in-sample winner is a procedure."""
import random
from decimal import Decimal
from math import comb

import pytest

from mdnorm.selection import SelectionReport, SelectionSplit, block_splits, cscv

D = Decimal


def panel(n_var=20, n=1000, seed=20260922, skilled=None, mu=0.0002,
          edge=0.0009):
    """Twenty variants with the same true edge, optionally one with more."""
    rng = random.Random(seed)
    out = {}
    for v in range(n_var):
        m = edge if (skilled is not None and v == skilled) else mu
        out[f"v{v:02d}"] = [D(str(round(rng.gauss(m, 0.01), 10)))
                            for _ in range(n)]
    return out


def about(value, target, tol="0.000001"):
    return abs(D(value) - D(str(target))) < D(tol)


ANN = D(252).sqrt()


# -- the splits ------------------------------------------------------------

def test_there_are_n_choose_half_splits():
    for s in (2, 4, 6, 10):
        assert len(block_splits(s)) == comb(s, s // 2)


def test_each_split_partitions_the_blocks():
    for is_, oos in block_splits(6):
        assert sorted(is_ + oos) == list(range(6))
        assert len(is_) == len(oos) == 3


def test_every_split_has_its_complement():
    splits = {is_ for is_, _ in block_splits(8)}
    for is_, oos in block_splits(8):
        assert oos in splits


def test_an_odd_or_tiny_block_count_is_refused():
    for s in (0, 1, 3, 7):
        with pytest.raises(ValueError, match="even number of at least 2"):
            block_splits(s)


# -- refusals over plausible answers ---------------------------------------

def test_a_remainder_is_refused_rather_than_dropped():
    p = panel(n_var=3, n=1001)
    with pytest.raises(ValueError, match="1 would be left over"):
        cscv(p, blocks=10, metric="mean")


def test_the_refusal_says_to_trim_deliberately():
    p = panel(n_var=3, n=1003)
    with pytest.raises(ValueError, match="Trim the sample deliberately"):
        cscv(p, blocks=10, metric="mean")


def test_variants_of_different_lengths_are_refused():
    p = panel(n_var=3, n=100)
    p["v01"] = p["v01"][:90]
    with pytest.raises(ValueError, match="different numbers of periods"):
        cscv(p, blocks=10, metric="mean")


def test_one_variant_is_not_a_choice():
    with pytest.raises(ValueError, match="at least two variants"):
        cscv(panel(n_var=1, n=100), blocks=10, metric="mean")


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="metric is one of"):
        cscv(panel(n_var=3, n=100), blocks=10, metric="sortino")


def test_a_sharpe_ratio_needs_ddof():
    with pytest.raises(ValueError, match="needs ddof"):
        cscv(panel(n_var=3, n=100), blocks=10, metric="sharpe")


def test_a_flat_variant_has_no_sharpe_and_is_named():
    p = panel(n_var=3, n=100)
    p["v01"] = [D(0)] * 100
    with pytest.raises(ArithmeticError, match="'v01' has no sharpe"):
        cscv(p, blocks=10, metric="sharpe", ddof=1)


# -- the mechanics ---------------------------------------------------------

def test_a_dominant_variant_wins_every_split_and_ranks_first():
    p = panel(n_var=5, n=200)
    p["v02"] = [x + D("0.05") for x in p["v02"]]
    rep = cscv(p, blocks=10, metric="mean")
    assert rep.chosen == {"v02": len(rep.splits)}
    assert all(s.oos_rank == 5 for s in rep.splits)
    assert rep.pbo == 0


def test_a_dominated_winner_is_counted_as_overfit():
    """A variant that is best in one half and worst in the other."""
    base = [D("0.001"), D("-0.001")] * 50
    good_first = [D("0.01")] * 50 + [D("-0.01")] * 50
    rep = cscv({"a": base, "b": good_first}, blocks=2, metric="mean")
    first = rep.splits[0]          # in-sample block 0: b looks better
    assert first.winner == "b"
    assert first.oos_rank == 1
    assert first.logit < 0


def test_the_relative_rank_is_strictly_inside_the_unit_interval():
    rep = cscv(panel(n_var=4, n=100), blocks=4, metric="mean")
    for s in rep.splits:
        assert 0 < s.relative_rank < 1


def test_ties_never_flatter_the_winner():
    same = [D("0.01"), D("-0.005")] * 50
    rep = cscv({"a": list(same), "b": list(same)}, blocks=2, metric="mean")
    for s in rep.splits:
        assert s.winner == "a"           # first in sorted order
        assert s.oos_rank == 1           # nothing strictly below it
        assert s.logit < 0


def test_the_mean_metric_matches_a_direct_calculation():
    p = panel(n_var=3, n=100)
    rep = cscv(p, blocks=4, metric="mean")
    s = rep.splits[0]
    rows = [x for b in s.in_sample for x in p[s.winner][b * 25:(b + 1) * 25]]
    assert about(s.is_metric, sum(rows, D(0)) / len(rows), "1e-30")


def test_the_sharpe_metric_matches_a_direct_calculation():
    from mdnorm.metrics import sharpe_ratio

    p = panel(n_var=3, n=100)
    rep = cscv(p, blocks=4, metric="sharpe", ddof=1)
    s = rep.splits[0]
    oos = [b for b in range(4) if b not in s.in_sample]
    rows = [x for b in oos for x in p[s.winner][b * 25:(b + 1) * 25]]
    assert about(s.oos_metric, sharpe_ratio(rows, ddof=1), "1e-25")


# -- what the procedure does on noise and on skill -------------------------

def test_on_noise_the_winner_does_no_better_than_an_average_pick():
    rep = cscv(panel(), blocks=10, metric="sharpe", ddof=1)
    assert rep.pbo >= D("0.5")
    assert rep.mean_oos_of_is_best < rep.mean_oos_all
    assert rep.mean_is_best > 3 * rep.mean_oos_of_is_best


def test_on_noise_a_better_looking_winner_does_worse():
    rep = cscv(panel(), blocks=10, metric="sharpe", ddof=1)
    assert rep.degradation_slope < 0


def test_one_skilled_variant_brings_the_probability_down():
    noise = cscv(panel(), blocks=10, metric="sharpe", ddof=1)
    skill = cscv(panel(skilled=7), blocks=10, metric="sharpe", ddof=1)
    assert skill.pbo < noise.pbo
    assert list(skill.chosen)[0] == "v07"
    assert skill.mean_oos_of_is_best > skill.mean_oos_all


def test_the_worked_example_on_noise():
    rep = cscv(panel(), blocks=10, metric="sharpe", ddof=1)
    assert len(rep.splits) == 252
    assert about(rep.pbo, "0.753968")
    assert about(rep.mean_is_best * ANN, "1.7260", "0.0001")
    assert about(rep.mean_oos_of_is_best * ANN, "0.2128", "0.0001")
    assert about(rep.mean_oos_all * ANN, "0.4979", "0.0001")
    assert about(rep.degradation_slope, "-0.5604", "0.0001")


def test_the_worked_example_with_one_skilled_variant():
    rep = cscv(panel(skilled=7), blocks=10, metric="sharpe", ddof=1)
    assert about(rep.pbo, "0.388889")
    assert rep.chosen["v07"] == 130
    assert about(rep.mean_oos_of_is_best * ANN, "0.7707", "0.0001")


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    rep = cscv(panel(n_var=3, n=100), blocks=4, metric="mean")
    for obj in (rep, rep.splits[0]):
        with pytest.raises(Exception):
            obj.variants = 9  # type: ignore[misc]


def test_the_report_is_the_documented_type():
    rep = cscv(panel(n_var=3, n=100), blocks=4, metric="mean")
    assert isinstance(rep, SelectionReport)
    assert isinstance(rep.splits[0], SelectionSplit)
