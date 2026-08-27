"""Reconciliation tests: two kinds of disagreement, and the clock offset."""
from decimal import Decimal

import pytest

from mdnorm import (
    AsOfSeries,
    Bar,
    BarField,
    Mismatch,
    MismatchKind,
    reconcile,
    reconcile_bars,
    suggest_shift,
)

D = Decimal
NS = 1_000_000_000
MS = 1_000_000


def series(pairs, name=""):
    return AsOfSeries([(t, D(str(v))) for t, v in pairs], name=name)


def identical():
    return series([(i * NS, 100 + i) for i in range(10)])


# -- agreement --------------------------------------------------------------


def test_identical_sources_agree_completely():
    rep, mm = reconcile(identical(), identical())
    assert rep.common_timestamps == 10
    assert rep.agreed == 10
    assert rep.value_mismatches == 0
    assert rep.coverage_difference == 0
    assert rep.agreement == 1
    assert mm == []
    assert rep.worst is None


def test_without_tolerance_any_difference_is_a_mismatch():
    a = series([(0, "100.00")])
    b = series([(0, "100.01")])
    rep, mm = reconcile(a, b)
    assert rep.agreed == 0 and rep.value_mismatches == 1
    assert mm[0].kind is MismatchKind.VALUE
    assert mm[0].difference == D("0.01")


def test_absolute_tolerance_admits_a_small_difference():
    a = series([(0, "100.00")])
    b = series([(0, "100.01")])
    rep, _ = reconcile(a, b, absolute_tolerance=D("0.01"))
    assert rep.agreed == 1 and rep.value_mismatches == 0


def test_relative_tolerance_scales_with_the_value():
    small = reconcile(series([(0, "1.00")]), series([(0, "1.02")]),
                      relative_tolerance=D("0.01"))[0]
    large = reconcile(series([(0, "100.00")]), series([(0, "100.50")]),
                      relative_tolerance=D("0.01"))[0]
    assert small.value_mismatches == 1     # 2% of 1.00 is outside 1%
    assert large.agreed == 1               # 0.5% of 100.00 is inside 1%


def test_either_tolerance_can_admit_a_pair():
    rep, _ = reconcile(series([(0, "100")]), series([(0, "100.5")]),
                       absolute_tolerance=D("1"),
                       relative_tolerance=D("0.0001"))
    assert rep.agreed == 1


def test_negative_tolerances_are_rejected():
    with pytest.raises(ValueError):
        reconcile(identical(), identical(), absolute_tolerance=D("-1"))
    with pytest.raises(ValueError):
        reconcile(identical(), identical(), relative_tolerance=D("-1"))
    with pytest.raises(ValueError):
        reconcile(identical(), identical(), limit=-1)


# -- the two kinds are kept apart -------------------------------------------


def test_coverage_and_content_are_counted_separately():
    a = series([(0, 100), (NS, 101), (2 * NS, 102)])
    b = series([(0, 100), (NS, 999)])            # one differs, one missing
    rep, mm = reconcile(a, b)
    assert rep.common_timestamps == 2
    assert rep.agreed == 1
    assert rep.value_mismatches == 1
    assert rep.only_left == 1
    assert rep.only_right == 0
    kinds = {m.kind for m in mm}
    assert kinds == {MismatchKind.VALUE, MismatchKind.ONLY_LEFT}


def test_agreement_is_computed_over_shared_timestamps_only():
    """A feed that simply carries less must not look like a feed that lies."""
    a = series([(i * NS, 100) for i in range(100)])
    b = series([(0, 100)])
    rep, _ = reconcile(a, b)
    assert rep.agreement == 1          # the one shared point matched
    assert rep.coverage_difference == 99


def test_agreement_is_none_without_a_shared_timestamp():
    rep, _ = reconcile(series([(0, 1)]), series([(NS, 1)]))
    assert rep.common_timestamps == 0
    assert rep.agreement is None
    assert rep.coverage_difference == 2


# -- worst case and limits --------------------------------------------------


def test_worst_is_the_largest_absolute_difference():
    a = series([(0, 100), (NS, 100), (2 * NS, 100)])
    b = series([(0, 101), (NS, 105), (2 * NS, 99)])
    rep, _ = reconcile(a, b)
    assert rep.max_absolute_difference == 5
    assert rep.worst.ts_ns == NS
    assert rep.max_relative_difference == D("0.05")


def test_limit_truncates_the_list_but_not_the_count():
    a = series([(i * NS, 100) for i in range(50)])
    b = series([(i * NS, 200) for i in range(50)])
    rep, mm = reconcile(a, b, limit=5)
    assert len(mm) == 5
    assert rep.value_mismatches == 50


def test_relative_difference_is_none_against_zero():
    m = Mismatch(0, MismatchKind.VALUE, D("0"), D("1"))
    assert m.difference == 1
    assert m.relative_difference is None


def test_difference_is_none_when_a_side_is_missing():
    m = Mismatch(0, MismatchKind.ONLY_LEFT, D("1"), None)
    assert m.difference is None
    assert m.relative_difference is None


# -- the clock offset -------------------------------------------------------


def offset_pair(offset_ns):
    """The same series, with the right one stamped late by a constant."""
    a = series([(i * NS, 100 + i) for i in range(20)])
    b = series([(i * NS + offset_ns, 100 + i) for i in range(20)])
    return a, b


def test_a_constant_offset_looks_like_total_disagreement():
    a, b = offset_pair(250 * MS)
    rep, _ = reconcile(a, b)
    assert rep.common_timestamps == 0
    assert rep.agreement is None          # not zero: nothing was compared


def test_suggest_shift_finds_the_offset():
    a, b = offset_pair(250 * MS)
    s = suggest_shift(a, b, max_shift_ns=NS)
    assert s.shift_ns == -250 * MS
    assert s.explains == 1


def test_the_suggested_shift_makes_the_sources_agree():
    a, b = offset_pair(250 * MS)
    s = suggest_shift(a, b, max_shift_ns=NS)
    rep, _ = reconcile(a, b, shift_right_ns=s.shift_ns)
    assert rep.common_timestamps == 20
    assert rep.agreed == 20


def test_suggest_shift_returns_none_when_nothing_is_close():
    a = series([(0, 1)])
    b = series([(10 * NS, 1)])
    assert suggest_shift(a, b, max_shift_ns=NS) is None


def test_suggest_shift_returns_none_on_an_empty_source():
    assert suggest_shift(series([]), identical(), max_shift_ns=NS) is None
    assert suggest_shift(identical(), series([]), max_shift_ns=NS) is None


def test_scattered_offsets_produce_a_weak_suggestion():
    """A low `explains` is noise, not a clock difference."""
    a = series([(i * NS, 1) for i in range(10)])
    b = series([(i * NS + (i % 5) * MS, 1) for i in range(10)])
    s = suggest_shift(a, b, max_shift_ns=NS)
    assert s.explains < D("0.5")


def test_suggest_shift_rejects_bad_arguments():
    with pytest.raises(ValueError):
        suggest_shift(identical(), identical(), max_shift_ns=-1)
    with pytest.raises(ValueError):
        suggest_shift(identical(), identical(), max_shift_ns=NS, sample=0)


def test_explains_is_none_when_nothing_was_considered():
    from mdnorm import ShiftSuggestion
    assert ShiftSuggestion(0, 0, 0).explains is None


# -- bars -------------------------------------------------------------------


def bars(values, *, label_offset=0):
    return [Bar(start_ns=i * NS + label_offset, interval_ns=NS, open=D("1"),
                high=D("2"), low=D("1"), close=D(str(v)), volume=D("10"),
                trades=1) for i, v in enumerate(values)]


def test_bars_are_compared_at_their_close_not_their_label():
    rep, _ = reconcile_bars(bars([10, 11, 12]), bars([10, 11, 12]))
    assert rep.common_timestamps == 3 and rep.agreed == 3


def test_bar_field_is_selectable():
    left = bars([10, 11])
    right = bars([99, 99])
    rep, _ = reconcile_bars(left, right, field=BarField.OPEN)
    assert rep.agreed == 2            # opens are equal, closes are not
    rep2, _ = reconcile_bars(left, right, field=BarField.CLOSE)
    assert rep2.value_mismatches == 2
