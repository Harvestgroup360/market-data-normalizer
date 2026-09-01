"""Arrival tests: two clocks, no default delay, and what the venue stamp buys."""
from decimal import Decimal
from fractions import Fraction

import pytest

from mdnorm import (
    Arrival,
    DelayReport,
    ViewGap,
    as_received,
    as_stamped,
    delay_report,
    read_arrivals_csv,
    view_gap,
)
from mdnorm.arrival import _nearest_rank

D = Decimal
MS = 1_000_000
S = 1_000_000_000


def feed(*pairs, start=1 * S, step=1 * S):
    """Arrivals at ``start``, ``start+step``, … each delayed by its pair."""
    out = []
    for i, delay in enumerate(pairs):
        venue = start + i * step
        out.append(Arrival(venue, venue + delay, D(100 + i)))
    return out


# -- Arrival ---------------------------------------------------------------

def test_delay_is_receipt_minus_venue():
    assert Arrival(1_000, 1_250).delay_ns == 250


def test_receipt_before_the_venue_stamp_is_representable():
    """Refusing to construct it would mean the clock skew never gets measured."""
    a = Arrival(2_000, 1_900)
    assert a.delay_ns == -100


def test_negative_timestamps_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        Arrival(-1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        Arrival(0, -1)


# -- delay_report ----------------------------------------------------------

def test_empty_input_reports_nothing_rather_than_zero():
    """No observations is not a delay of zero, and must not read as one."""
    rep = delay_report([])
    assert rep.observations == 0
    assert rep.median_ns is None and rep.p95_ns is None
    assert rep.clock_skew_share is None and rep.tail_ratio is None


def test_percentiles_are_values_that_actually_happened():
    delays = [10, 20, 30, 40, 50, 60, 70, 80, 90, 1_000]
    rep = delay_report(feed(*delays))
    assert rep.min_ns == 10 and rep.max_ns == 1_000
    assert rep.median_ns in delays
    assert rep.p95_ns in delays


def test_median_is_nearest_rank_not_the_average_of_two():
    """An even count does not average the middle pair into a new number."""
    rep = delay_report(feed(100, 200, 300, 400))
    assert rep.median_ns == 200


def test_nearest_rank_endpoints():
    xs = list(range(1, 101))
    assert _nearest_rank(xs, 50) == 50
    assert _nearest_rank(xs, 95) == 95
    assert _nearest_rank(xs, 100) == 100
    assert _nearest_rank([7], 95) == 7


def test_the_mean_is_not_reported_and_would_have_been_misleading():
    """The whole reason the mean is absent, stated as a test."""
    delays = [100] * 99 + [1_000_000]
    rep = delay_report(feed(*delays))
    mean = sum(delays) / len(delays)
    assert rep.median_ns == 100
    assert mean > 10_000          # the mean is ~100x the typical case
    assert not hasattr(rep, "mean_ns")


def test_tail_ratio_says_the_typical_and_bad_cases_are_different_problems():
    rep = delay_report(feed(*([100] * 90 + [5_000] * 10)))
    assert rep.median_ns == 100
    assert rep.p95_ns == 5_000
    assert rep.tail_ratio == 50


def test_exactly_five_percent_slow_does_not_move_the_p95():
    """Nearest rank is a rank, not a threshold, and this is the boundary.

    With 95 of 100 observations fast, the 95th smallest is still a fast one,
    so the p95 reports the fast case even though a twentieth of the feed is
    fifty times slower. That is the definition behaving correctly rather than
    a figure to be nudged: the number to look at here is ``max_ns``.
    """
    rep = delay_report(feed(*([100] * 95 + [5_000] * 5)))
    assert rep.p95_ns == 100
    assert rep.max_ns == 5_000


def test_a_flat_link_has_a_tail_ratio_of_one():
    rep = delay_report(feed(*([250] * 40)))
    assert rep.tail_ratio == 1


def test_negative_delays_are_counted_and_never_clamped():
    xs = [Arrival(2_000, 1_900, D(1)), Arrival(3_000, 3_100, D(2)),
          Arrival(4_000, 4_100, D(3)), Arrival(5_000, 5_100, D(4))]
    rep = delay_report(xs)
    assert rep.negative == 1
    assert rep.min_ns == -100          # not zero
    assert rep.clock_skew_share == D("0.25")


def test_out_of_order_arrivals_are_counted_not_sorted_away():
    xs = [Arrival(1_000, 1_100, D(1)),
          Arrival(2_000, 5_000, D(2)),   # this one is slow
          Arrival(3_000, 3_100, D(3)),   # and is overtaken here
          Arrival(4_000, 4_100, D(4))]
    rep = delay_report(xs)
    assert rep.out_of_order == 1
    assert rep.observations == 4


def test_out_of_order_is_independent_of_input_order():
    xs = [Arrival(1_000, 1_100, D(1)), Arrival(2_000, 5_000, D(2)),
          Arrival(3_000, 3_100, D(3))]
    assert delay_report(xs).out_of_order == delay_report(xs[::-1]).out_of_order


def test_an_assumed_delay_is_labelled_as_assumed():
    rep = delay_report(feed(100, 200, 300), assume_delay_ns=500)
    assert rep.assumed is True
    assert rep.median_ns == rep.p95_ns == rep.min_ns == rep.max_ns == 500


def test_a_measured_report_is_not_labelled_assumed():
    assert delay_report(feed(100, 200)).assumed is False


def test_an_assumed_delay_may_not_be_negative():
    with pytest.raises(ValueError, match="non-negative"):
        delay_report([], assume_delay_ns=-1)


def test_there_is_no_default_delay():
    """Nothing in the signature supplies one; the report is empty instead."""
    assert delay_report([]).min_ns is None


# -- the two series --------------------------------------------------------

def test_as_received_is_the_series_you_could_have_acted_on():
    xs = feed(250 * MS, 250 * MS)
    series = as_received(xs)
    # At the venue stamp, the value has not arrived.
    assert series.at(1 * S)[0] is None
    assert series.at(1 * S + 250 * MS)[0] == D(100)


def test_as_stamped_is_the_optimistic_view_and_is_kept_on_purpose():
    xs = feed(250 * MS, 250 * MS)
    assert as_stamped(xs).at(1 * S)[0] == D(100)


def test_the_two_series_differ_by_exactly_the_delay():
    xs = feed(250 * MS, 300 * MS, 100 * MS)
    stamped, received = as_stamped(xs), as_received(xs)
    for a in xs:
        assert stamped.at(a.venue_ns)[0] == a.value
        assert received.at(a.received_ns)[0] == a.value
        if a.delay_ns > 0:
            assert received.at(a.received_ns - 1)[0] != a.value


def test_timestamp_only_arrivals_have_no_series():
    xs = [Arrival(1_000, 1_100), Arrival(2_000, 2_100)]
    assert delay_report(xs).observations == 2      # this still works
    with pytest.raises(ValueError, match="no values"):
        as_received(xs)


def test_rows_without_a_value_are_dropped_from_the_series_only():
    xs = [Arrival(1_000, 1_100, D(1)), Arrival(2_000, 2_100),
          Arrival(3_000, 3_100, D(3))]
    assert len(as_received(xs)) == 2
    assert delay_report(xs).observations == 3


# -- view_gap --------------------------------------------------------------

def test_a_zero_delay_feed_has_no_gap_at_all():
    xs = feed(0, 0, 0, 0)
    gap = view_gap(xs, list(range(0, 6 * S, S)))
    assert gap.differ == 0
    assert gap.share == 0
    assert gap.earliest_gain_ns is None and gap.largest_gain_ns is None


def test_the_gap_appears_exactly_where_a_value_is_in_flight():
    xs = feed(400 * MS, 400 * MS)
    grid = [1 * S, 1 * S + 500 * MS, 2 * S, 2 * S + 500 * MS]
    gap = view_gap(xs, grid)
    # At each venue stamp the optimistic view is early; 500ms later it is not.
    assert gap.differ == 2
    assert gap.earliest_gain_ns == 1 * S


def test_largest_foresight_never_exceeds_the_largest_delay():
    xs = feed(120 * MS, 800 * MS, 50 * MS, 300 * MS)
    gap = view_gap(xs, list(range(0, 6 * S, 100 * MS)))
    assert gap.largest_gain_ns is not None
    assert gap.largest_gain_ns <= max(a.delay_ns for a in xs)


def test_an_empty_grid_reports_no_share_rather_than_zero():
    gap = view_gap(feed(100, 200), [])
    assert gap.grid_points == 0
    assert gap.share is None


def test_the_same_value_on_both_sides_is_not_a_disagreement():
    """The question is what the view showed, not which row produced it."""
    xs = [Arrival(1 * S, 1 * S + 500 * MS, D(100)),
          Arrival(2 * S, 2 * S + 500 * MS, D(100))]
    gap = view_gap(xs, [2 * S, 2 * S + 100 * MS])
    assert gap.differ == 0


def test_share_is_exact_and_matches_the_counts():
    xs = feed(300 * MS, 300 * MS, 300 * MS)
    grid = list(range(0, 5 * S, 250 * MS))
    gap = view_gap(xs, grid)
    assert gap.share == Decimal(gap.differ) / gap.grid_points
    assert Fraction(gap.differ, gap.grid_points) == Fraction(
        gap.share.as_integer_ratio()[0], gap.share.as_integer_ratio()[1])


def test_the_gap_grows_with_the_delay():
    grid = list(range(0, 6 * S, 100 * MS))
    small = view_gap(feed(50 * MS, 50 * MS, 50 * MS), grid)
    large = view_gap(feed(900 * MS, 900 * MS, 900 * MS), grid)
    assert large.differ > small.differ


# -- CSV -------------------------------------------------------------------

def test_read_arrivals_csv(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("venue_ns,received_ns,value\n"
                 "1000,1250,100.5\n2000,2100,100.6\n", encoding="utf-8")
    xs = read_arrivals_csv(str(p))
    assert [a.delay_ns for a in xs] == [250, 100]
    assert xs[0].value == D("100.5")


def test_a_row_with_only_one_clock_is_an_error_not_a_skip(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("venue_ns,received_ns\n1000,1250\n2000,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_arrivals_csv(str(p))


def test_a_missing_receipt_column_is_named_in_the_error(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("venue_ns,value\n1000,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="received_ns"):
        read_arrivals_csv(str(p))


def test_a_timestamps_only_file_reads_with_value_column_none(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("venue_ns,received_ns\n1000,1250\n2000,2100\n",
                 encoding="utf-8")
    xs = read_arrivals_csv(str(p), value_column=None)
    assert all(a.value is None for a in xs)
    assert delay_report(xs).observations == 2


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("venue_ns,received_ns\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no arrivals"):
        read_arrivals_csv(str(p))


def test_custom_column_names(tmp_path):
    p = tmp_path / "feed.csv"
    p.write_text("exch,rx,px\n1000,1250,9\n", encoding="utf-8")
    xs = read_arrivals_csv(str(p), venue_column="exch",
                           received_column="rx", value_column="px")
    assert xs[0].delay_ns == 250 and xs[0].value == D(9)


# -- the reason the module exists ------------------------------------------

def test_delayed_by_the_measured_median_reproduces_a_uniform_feed():
    """Measure the delay, hand it to ``delayed``, and the two views agree."""
    xs = feed(*([250 * MS] * 20))
    rep = delay_report(xs)
    assert rep.median_ns == 250 * MS
    shifted = as_stamped(xs).delayed(rep.median_ns)
    received = as_received(xs)
    for t in range(0, 25 * S, 100 * MS):
        assert shifted.at(t) == received.at(t)


def test_frozen_dataclasses():
    for obj in (Arrival(1, 2), DelayReport(0, 0, 0, None, None, None, None),
                ViewGap(0, 0, None, None)):
        with pytest.raises(Exception):
            obj.observations = 1  # type: ignore[misc]
