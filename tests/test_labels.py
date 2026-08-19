"""Label and split tests: the overlap invariant, purging, embargo."""
from decimal import Decimal

import pytest

from mdnorm import (
    ReturnMethod,
    Split,
    forward_returns,
    purged_splits,
    purged_train_test,
)

D = Decimal


def px(*vals):
    return [None if v is None else D(str(v)) for v in vals]


def leaks(split, horizon):
    """Training samples whose label window reaches into the test block."""
    if not split.test:
        return []
    lo, hi = split.test[0], split.test[-1]
    return [i for i in split.train if i < lo <= i + horizon or lo <= i <= hi]


# -- the label ---------------------------------------------------------------

def test_a_forward_return_looks_exactly_horizon_ahead():
    y = forward_returns(px(100, 0.1, 0.1, 110), horizon=3)
    assert y[0] == D("0.1")


def test_the_last_rows_have_no_outcome_yet():
    """Real missingness: those futures have not happened."""
    y = forward_returns(px(1, 2, 3, 4, 5), horizon=2)
    assert y[-2:] == [None, None]
    assert len(y) == 5


def test_a_horizon_of_one_is_the_next_period_return():
    y = forward_returns(px(100, 110), horizon=1)
    assert y[0] == D("0.1") and y[1] is None


def test_log_forward_returns():
    y = forward_returns(px(100, 200), horizon=1, method=ReturnMethod.LOG)
    assert abs(y[0] - D("0.6931471805599453")) < D("0.0000001")


def test_a_gap_at_either_end_of_the_horizon_kills_the_label():
    assert forward_returns(px(100, None, 120), horizon=2)[0] == D("0.2")
    assert forward_returns(px(100, 110, None), horizon=2)[0] is None
    assert forward_returns(px(None, 110, 120), horizon=2)[0] is None


def test_a_non_positive_price_has_no_forward_return():
    assert forward_returns(px(100, 0), horizon=1)[0] is None
    assert forward_returns(px(0, 100), horizon=1)[0] is None


def test_the_label_series_lines_up_with_its_input():
    for n in (1, 5, 40):
        assert len(forward_returns(px(*range(1, n + 1)), horizon=3)) == n


def test_a_horizon_longer_than_the_series_labels_nothing():
    assert forward_returns(px(1, 2, 3), horizon=10) == [None, None, None]


def test_a_non_positive_horizon_is_rejected():
    with pytest.raises(ValueError):
        forward_returns(px(1, 2, 3), horizon=0)


def test_a_label_is_not_a_feature():
    """Pinned as documentation: this series depends on the future by design.

    Every function in mdnorm.features fails if a later value changes an
    earlier output. This one is supposed to.
    """
    base = px(100, 101, 102, 103, 104)
    tampered = base[:3] + px(9999, 8888)
    assert forward_returns(base, horizon=2)[0] == forward_returns(tampered, horizon=2)[0]
    assert forward_returns(base, horizon=2)[1] != forward_returns(tampered, horizon=2)[1]


# -- the invariant the splits exist for --------------------------------------

@pytest.mark.parametrize("n,k,h,e", [
    (100, 5, 1, 0), (100, 5, 10, 0), (100, 5, 10, 5),
    (37, 3, 4, 2), (20, 4, 2, 2), (10, 2, 3, 0),
])
def test_no_training_label_ever_reaches_the_test_block(n, k, h, e):
    """The property the module exists to guarantee, checked directly."""
    for split in purged_splits(n, n_splits=k, horizon=h, embargo=e):
        assert leaks(split, h) == []


def test_an_unpurged_split_would_fail_that_check():
    """Pinned to show the leak the purge removes.

    A plain contiguous split — train is simply everything outside the test
    block — puts rows whose label window covers the test period into training.
    """
    n, lo, hi, h = 40, 20, 29, 5
    naive = Split(train=tuple(i for i in range(n) if not lo <= i <= hi),
                  test=tuple(range(lo, hi + 1)))
    assert leaks(naive, h) == [15, 16, 17, 18, 19]

    purged = purged_splits(n, n_splits=4, horizon=h)[2]
    assert purged.test == naive.test
    assert leaks(purged, h) == []


def test_purging_removes_exactly_the_horizon_worth_of_rows():
    split = purged_splits(40, n_splits=4, horizon=5)[2]   # test block 20..29
    assert split.purged == 5
    assert max(i for i in split.train if i < 20) == 14


# -- the folds themselves ----------------------------------------------------

def test_every_sample_is_tested_exactly_once():
    tested = [i for s in purged_splits(50, n_splits=5, horizon=3) for i in s.test]
    assert sorted(tested) == list(range(50))


def test_test_blocks_are_contiguous_and_in_time_order():
    splits = purged_splits(50, n_splits=5, horizon=3)
    for s in splits:
        assert list(s.test) == list(range(s.test[0], s.test[-1] + 1))
    assert [s.test[0] for s in splits] == sorted(s.test[0] for s in splits)


def test_train_and_test_never_overlap():
    for s in purged_splits(50, n_splits=5, horizon=3, embargo=4):
        assert not set(s.train) & set(s.test)


def test_uneven_sample_counts_still_partition_cleanly():
    tested = [i for s in purged_splits(37, n_splits=5, horizon=2) for i in s.test]
    assert sorted(tested) == list(range(37))


# -- embargo -----------------------------------------------------------------

def test_the_embargo_drops_the_rows_right_after_the_test_block():
    without = purged_splits(40, n_splits=4, horizon=2, embargo=0)[1]
    with_e = purged_splits(40, n_splits=4, horizon=2, embargo=3)[1]
    dropped = set(without.train) - set(with_e.train)
    assert dropped == {20, 21, 22}          # block is 10..19
    assert with_e.embargoed == 3


def test_the_last_fold_has_nothing_to_embargo():
    last = purged_splits(40, n_splits=4, horizon=2, embargo=5)[-1]
    assert last.embargoed == 0


def test_the_first_fold_has_nothing_to_purge():
    first = purged_splits(40, n_splits=4, horizon=5)[0]
    assert first.purged == 0


def test_discarded_counts_both_reasons():
    s = purged_splits(40, n_splits=4, horizon=2, embargo=3)[1]
    assert s.discarded == s.purged + s.embargoed == 5


# -- the single chronological split ------------------------------------------

def test_a_single_split_puts_the_test_block_last():
    s = purged_train_test(100, test_fraction=0.2, horizon=1)
    assert s.test == tuple(range(80, 100))
    assert max(s.train) < 80


def test_the_single_split_purges_the_seam():
    s = purged_train_test(100, test_fraction=0.2, horizon=4)
    assert s.purged == 4
    assert max(s.train) == 75


def test_a_test_fraction_must_leave_something_to_train_on():
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            purged_train_test(100, test_fraction=bad, horizon=1)


def test_a_tiny_fraction_still_yields_one_test_sample():
    s = purged_train_test(100, test_fraction=0.001, horizon=1)
    assert len(s.test) == 1


# -- validation --------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"n_splits": 1}, {"n_splits": 0}, {"horizon": 0}, {"embargo": -1},
    {"n_splits": 101},
])
def test_split_arguments_are_validated(kwargs):
    args = {"n_splits": 5, "horizon": 3, "embargo": 0}
    args.update(kwargs)
    with pytest.raises(ValueError):
        purged_splits(100, **args)


def test_a_non_positive_sample_count_is_rejected():
    with pytest.raises(ValueError):
        purged_splits(0, n_splits=2, horizon=1)


def test_a_horizon_that_swallows_the_training_set_leaves_it_empty():
    """Not an error — an honest answer about a horizon that big."""
    s = purged_splits(10, n_splits=2, horizon=100)[1]
    assert s.train == () and len(s.test) == 5


# -- integration -------------------------------------------------------------

def test_features_and_labels_share_the_same_index():
    from mdnorm import returns, rolling_zscore
    prices = px(*[100 + (i % 7) for i in range(60)])
    x1 = returns(prices)
    x2 = rolling_zscore(prices, 10)
    y = forward_returns(prices, horizon=5)
    assert len(x1) == len(x2) == len(y) == len(prices)


def test_the_whole_path_from_a_grid_to_clean_folds():
    from decimal import Decimal as Dec

    from mdnorm import EventType, MarketEvent, align, column, rolling_zscore
    SEC = 1_000_000_000
    events = [MarketEvent(symbol="X", venue="v", event_type=EventType.TRADE,
                          ts_ns=i * 60 * SEC, price=Dec(100 + (i % 5)),
                          size=Dec(1)) for i in range(80)]
    rows = align({"X": events}, interval_ns=60 * SEC)
    prices = column(rows, "X")
    z = rolling_zscore(prices, 10)
    y = forward_returns(prices, horizon=5)

    splits = purged_splits(len(rows), n_splits=4, horizon=5, embargo=10)
    for s in splits:
        assert leaks(s, 5) == []
        # every index a fold hands back addresses a real row of the matrix
        for i in tuple(s.train) + tuple(s.test):
            assert 0 <= i < len(z) and 0 <= i < len(y)
