"""Independence tests: how many observations a set of labels is really worth."""
import random
from decimal import Decimal

import pytest

from mdnorm.independence import (
    EffectiveSample,
    Span,
    autocorrelation,
    concurrency,
    deflate_t_stat,
    effective_sample_size,
    effective_sample_size_series,
    label_spans,
    read_spans_csv,
    uniqueness,
)

D = Decimal


def ar1(phi, n=4000, seed=11):
    rng = random.Random(seed)
    out, x = [], 0.0
    for _ in range(n):
        x = phi * x + rng.gauss(0, 1)
        out.append(D(str(round(x, 8))))
    return out


# -- spans -----------------------------------------------------------------

def test_a_span_must_cover_a_positive_range():
    assert Span(0, 1).length == 1
    with pytest.raises(ValueError, match="positive range"):
        Span(5, 5)
    with pytest.raises(ValueError, match="positive range"):
        Span(5, 4)


def test_label_spans_matches_the_shape_forward_returns_produces():
    spans = label_spans(3, horizon=5)
    assert spans == [Span(0, 5), Span(1, 6), Span(2, 7)]


def test_label_spans_honours_step_and_start():
    assert label_spans(3, horizon=2, step=4, start=100) == [
        Span(100, 102), Span(104, 106), Span(108, 110)]


def test_label_spans_validates_its_arguments():
    with pytest.raises(ValueError, match="non-negative"):
        label_spans(-1, horizon=1)
    with pytest.raises(ValueError, match="horizon must be at least 1"):
        label_spans(1, horizon=0)
    with pytest.raises(ValueError, match="step must be at least 1"):
        label_spans(1, horizon=1, step=0)


def test_no_labels_means_no_spans():
    assert label_spans(0, horizon=5) == []


# -- concurrency -----------------------------------------------------------

def test_concurrency_records_the_points_where_the_count_changes():
    assert concurrency([Span(0, 3), Span(1, 4)]) == {0: 1, 1: 2, 3: 1, 4: 0}


def test_concurrency_of_nothing_is_empty():
    assert concurrency([]) == {}


def test_identical_spans_stack():
    assert concurrency([Span(0, 2)] * 3) == {0: 3, 2: 0}


def test_a_gap_between_spans_drops_back_to_zero():
    assert concurrency([Span(0, 1), Span(5, 6)]) == {0: 1, 1: 0, 5: 1, 6: 0}


# -- uniqueness ------------------------------------------------------------

def test_a_label_that_overlaps_nothing_is_worth_one_observation():
    assert uniqueness(label_spans(4, horizon=3, step=3)) == [D(1)] * 4


def test_two_labels_sharing_their_whole_window_are_worth_a_half_each():
    assert uniqueness([Span(0, 4), Span(0, 4)]) == [D("0.5"), D("0.5")]


def test_the_labels_at_the_ends_overlap_less_and_are_worth_more():
    u = uniqueness(label_spans(20, horizon=5))
    assert u[0] > u[10]
    assert u[-1] > u[10]


def test_uniqueness_of_nothing_is_nothing():
    assert uniqueness([]) == []


# -- effective sample size, exact case -------------------------------------

def test_non_overlapping_labels_lose_nothing():
    s = effective_sample_size(label_spans(100, horizon=5, step=5))
    assert s.nominal == 100
    assert s.effective == D(100)
    assert s.ratio == D(1)
    assert s.inflation == D(1)
    assert s.estimated is False


def test_a_horizon_of_one_sampled_every_step_is_already_independent():
    assert effective_sample_size(label_spans(100, horizon=1)).effective == D(100)


def test_daily_sampling_of_a_five_day_label_costs_about_four_fifths():
    """The headline: a thousand rows carrying two hundred observations."""
    s = effective_sample_size(label_spans(1_000, horizon=5))
    assert D(198) < s.effective < D(203)
    assert D("2.2") < s.inflation < D("2.3")       # near the square root of 5


def test_the_effective_count_tracks_the_horizon():
    for horizon in (2, 4, 10, 20):
        s = effective_sample_size(label_spans(1_000, horizon=horizon))
        assert abs(s.effective - D(1_000) / horizon) < D(horizon)


def test_the_effective_count_never_exceeds_the_nominal_one():
    for horizon in (1, 3, 7, 50):
        s = effective_sample_size(label_spans(200, horizon=horizon))
        assert s.effective <= D(200)


def test_an_empty_label_set_reports_no_ratio_rather_than_one():
    s = effective_sample_size([])
    assert s.nominal == 0
    assert s.effective == D(0)
    assert s.ratio is None
    assert s.inflation is None


def test_a_negative_effective_count_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        EffectiveSample(nominal=10, effective=D(-1))


# -- the t-statistic -------------------------------------------------------

def test_a_publishable_t_becomes_an_unpublishable_one():
    s = effective_sample_size(label_spans(1_000, horizon=5))
    adjusted = deflate_t_stat(D("2.0"), s)
    assert adjusted is not None
    assert D("0.85") < adjusted < D("0.95")


def test_deflating_against_independent_labels_changes_nothing():
    s = effective_sample_size(label_spans(100, horizon=5, step=5))
    assert deflate_t_stat(D("2.0"), s) == D("2.0")


def test_deflation_is_the_same_as_recomputing_on_the_effective_count():
    s = effective_sample_size(label_spans(500, horizon=4))
    t = D("3.0")
    direct = t / (D(s.nominal) / s.effective).sqrt()
    assert deflate_t_stat(t, s) == direct


def test_no_effective_sample_means_no_adjusted_statistic():
    assert deflate_t_stat(D("2.0"), EffectiveSample(0, D(0))) is None


# -- autocorrelation -------------------------------------------------------

def test_an_alternating_series_has_a_lag_one_autocorrelation_near_minus_one():
    values = [D(1) if i % 2 == 0 else D(-1) for i in range(100)]
    rho = autocorrelation(values, max_lag=2)
    assert rho[0] < D("-0.95")
    assert rho[1] > D("0.95")


def test_a_trending_series_is_strongly_positively_autocorrelated():
    rho = autocorrelation([D(i) for i in range(200)], max_lag=1)
    assert rho[0] > D("0.9")


def test_autocorrelation_validates_its_inputs():
    values = [D(1), D(2), D(3)]
    with pytest.raises(ValueError, match="at least 1"):
        autocorrelation(values, max_lag=0)
    with pytest.raises(ValueError, match="below the sample size"):
        autocorrelation(values, max_lag=3)
    with pytest.raises(ValueError, match="at least two"):
        autocorrelation([D(1)], max_lag=1)
    with pytest.raises(ValueError, match="constant series"):
        autocorrelation([D(1)] * 10, max_lag=2)


# -- effective sample size, estimated case ---------------------------------

def test_an_independent_series_keeps_almost_all_of_its_sample():
    rng = random.Random(7)
    values = [D(str(round(rng.gauss(0, 1), 8))) for _ in range(4000)]
    s = effective_sample_size_series(values, max_lag=50)
    assert s.ratio > D("0.9")
    assert s.estimated is True


@pytest.mark.parametrize("phi,expected", [(D("0.5"), D("0.333")),
                                          (D("0.3"), D("0.538"))])
def test_an_ar1_series_lands_near_its_closed_form(phi, expected):
    """For AR(1) the ratio is (1 - phi) / (1 + phi), which is checkable."""
    s = effective_sample_size_series(ar1(float(phi)), max_lag=100)
    assert abs(s.ratio - expected) < D("0.08")


def test_more_persistence_leaves_less_sample():
    low = effective_sample_size_series(ar1(0.2), max_lag=100)
    high = effective_sample_size_series(ar1(0.8), max_lag=100)
    assert high.ratio < low.ratio


def test_the_sum_stops_at_the_first_non_positive_autocorrelation():
    """Past that point the estimates are noise whose signs cancel arbitrarily.

    An alternating series has a negative lag-one autocorrelation, so the sum
    is empty and the estimate falls back to the nominal count rather than
    reporting an effective sample larger than the sample.
    """
    values = [D(1) if i % 2 == 0 else D(-1) for i in range(200)]
    s = effective_sample_size_series(values, max_lag=20)
    assert s.effective == D(200)
    assert s.ratio == D(1)


def test_an_estimated_count_is_marked_as_estimated():
    exact = effective_sample_size(label_spans(100, horizon=2))
    estimated = effective_sample_size_series(ar1(0.5, n=500), max_lag=20)
    assert exact.estimated is False
    assert estimated.estimated is True


# -- CSV -------------------------------------------------------------------

def test_read_spans_csv(tmp_path):
    p = tmp_path / "spans.csv"
    p.write_text("start,end\n0,5\n3,9\n", encoding="utf-8")
    assert read_spans_csv(str(p)) == [Span(0, 5), Span(3, 9)]


def test_a_reversed_span_in_a_file_names_its_line(tmp_path):
    p = tmp_path / "spans.csv"
    p.write_text("start,end\n0,5\n9,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_spans_csv(str(p))


def test_a_missing_column_is_an_error(tmp_path):
    p = tmp_path / "spans.csv"
    p.write_text("from,to\n0,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_spans_csv(str(p))


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "spans.csv"
    p.write_text("start,end\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no spans"):
        read_spans_csv(str(p))


def test_irregular_spans_from_a_file_are_measured_like_any_other(tmp_path):
    p = tmp_path / "spans.csv"
    p.write_text("start,end\n0,10\n0,10\n50,60\n", encoding="utf-8")
    s = effective_sample_size(read_spans_csv(str(p)))
    assert s.effective == D(2)          # a shared pair plus a lone label


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Span(0, 1), EffectiveSample(1, D(1))):
        with pytest.raises(Exception):
            obj.start = 5  # type: ignore[misc]
