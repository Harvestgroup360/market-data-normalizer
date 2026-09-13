"""Breadth tests: how many bets a cross-section really contains."""
import random
from decimal import Decimal

import pytest

from mdnorm.breadth import (
    BreadthReport,
    CorrelationMatrix,
    average_correlation,
    breadth_report,
    correlation_matrix,
    effective_bets,
    effective_observations,
    eigenvalues,
)

D = Decimal


def mat(rows, names=None):
    names = names or tuple(f"s{i}" for i in range(len(rows)))
    return CorrelationMatrix(
        names=names,
        rows=tuple(tuple(D(str(x)) for x in r) for r in rows))


def equicorrelated(n, rho):
    return mat([[1 if i == j else rho for j in range(n)] for i in range(n)])


def about(value, target, tol="0.0001"):
    return abs(D(value) - D(str(target))) < D(tol)


# -- the matrix type -------------------------------------------------------

def test_a_matrix_keeps_its_names_and_size():
    m = mat([[1, "0.3"], ["0.3", 1]], names=("AAA", "BBB"))
    assert m.names == ("AAA", "BBB")
    assert m.size == 2
    assert m[0, 1] == D("0.3")


def test_one_name_is_not_a_cross_section():
    with pytest.raises(ValueError, match="at least two names"):
        mat([[1]])


def test_a_name_cannot_repeat():
    with pytest.raises(ValueError, match="listed twice"):
        mat([[1, 0], [0, 1]], names=("A", "A"))


def test_the_diagonal_must_be_one():
    """A covariance matrix passed here would change every number silently."""
    with pytest.raises(ValueError, match="diagonal must be exactly one"):
        mat([[4, "0.3"], ["0.3", 1]])


def test_an_asymmetric_matrix_is_refused():
    with pytest.raises(ValueError, match="not symmetric"):
        mat([[1, "0.3"], ["0.4", 1]])


def test_a_correlation_outside_the_range_is_refused():
    with pytest.raises(ValueError, match="between -1 and 1"):
        mat([[1, "1.4"], ["1.4", 1]])


def test_a_ragged_matrix_is_refused():
    with pytest.raises(ValueError, match="not 2 by 2"):
        CorrelationMatrix(names=("a", "b"), rows=((D(1), D(0)), (D(0),)))


# -- building one from returns ---------------------------------------------

def test_a_matrix_is_built_from_aligned_series():
    m = correlation_matrix({
        "a": [D(1), D(2), D(3), D(4)],
        "b": [D(2), D(4), D(6), D(8)],
    })
    assert m.size == 2
    assert m.observations == 4
    assert about(m[0, 1], 1)            # perfectly proportional


def test_an_inverted_series_correlates_minus_one():
    m = correlation_matrix({
        "a": [D(1), D(2), D(3)],
        "b": [D(-1), D(-2), D(-3)],
    })
    assert about(m[0, 1], -1)


def test_the_result_is_symmetric_and_unit_diagonal():
    rng = random.Random(3)
    cols = {f"s{i}": [D(str(round(rng.gauss(0, 1), 8))) for _ in range(60)]
            for i in range(5)}
    m = correlation_matrix(cols)
    for i in range(5):
        assert m[i, i] == 1
        for j in range(5):
            assert m[i, j] == m[j, i]


def test_ragged_series_raise_rather_than_being_truncated():
    """Truncating would change which period the answer describes."""
    with pytest.raises(ValueError, match="different lengths"):
        correlation_matrix({"a": [D(1), D(2), D(3)], "b": [D(1), D(2)]})


def test_a_constant_series_raises_rather_than_reporting_zero():
    with pytest.raises(ValueError, match="does not vary"):
        correlation_matrix({"a": [D(1), D(1), D(1)], "b": [D(1), D(2), D(3)]})


def test_one_series_is_not_a_cross_section():
    with pytest.raises(ValueError, match="at least two series"):
        correlation_matrix({"a": [D(1), D(2)]})


def test_two_observations_are_the_minimum():
    with pytest.raises(ValueError, match="at least two observations"):
        correlation_matrix({"a": [D(1)], "b": [D(2)]})


# -- the average -----------------------------------------------------------

def test_the_average_excludes_the_diagonal():
    """Including it would pull every average toward one by exactly the amount
    that hides the problem."""
    m = mat([[1, "0.2", "0.4"], ["0.2", 1, "0.6"], ["0.4", "0.6", 1]])
    assert average_correlation(m) == D("0.4")


def test_the_average_of_an_uncorrelated_matrix_is_zero():
    assert average_correlation(equicorrelated(4, 0)) == 0


# -- eigenvalues -----------------------------------------------------------

def test_the_identity_has_unit_eigenvalues():
    assert eigenvalues(equicorrelated(4, 0)) == [1, 1, 1, 1]


def test_an_equicorrelated_matrix_has_the_known_spectrum():
    """One eigenvalue of 1+(n-1)rho, the rest 1-rho."""
    lam = eigenvalues(equicorrelated(5, "0.5"))
    assert about(lam[0], 3)
    assert all(about(v, "0.5") for v in lam[1:])


def test_the_eigenvalues_sum_to_the_trace():
    rng = random.Random(11)
    cols = {f"s{i}": [D(str(round(rng.gauss(0, 1), 8))) for _ in range(200)]
            for i in range(8)}
    lam = eigenvalues(correlation_matrix(cols))
    assert about(sum(lam), 8, "0.000001")


def test_the_eigenvalues_come_back_descending():
    lam = eigenvalues(equicorrelated(6, "0.3"))
    assert lam == sorted(lam, reverse=True)


def test_a_singular_matrix_still_decomposes():
    """Two identical names: one direction carries everything."""
    lam = eigenvalues(mat([[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    assert about(lam[0], 2) and about(lam[1], 1) and about(lam[2], 0)


# -- effective bets --------------------------------------------------------

def test_uncorrelated_names_are_all_bets():
    assert about(effective_bets(equicorrelated(5, 0)), 5)


def test_identical_names_are_one_bet():
    assert about(effective_bets(mat([[1, 1], [1, 1]])), 1)


def test_correlation_costs_bets():
    got = [effective_bets(equicorrelated(10, r))
           for r in ("0", "0.2", "0.5", "0.9")]
    assert got == sorted(got, reverse=True)
    assert about(got[0], 10)


def test_the_two_measures_are_not_the_same_quantity():
    """Three names at half correlation: two bets, one and a half observations."""
    m = equicorrelated(3, "0.5")
    assert about(effective_bets(m), 2)
    assert about(effective_observations(m), "1.5")


def test_they_coincide_only_at_the_extremes():
    for m in (equicorrelated(6, 0), mat([[1, 1], [1, 1]])):
        assert about(effective_bets(m), effective_observations(m))


# -- effective observations ------------------------------------------------

def test_the_closed_form_matches_its_definition():
    # n / (1 + (n-1) rho) = 10 / (1 + 9*0.4)
    assert about(effective_observations(equicorrelated(10, "0.4")),
                 "2.173913")


def test_an_average_that_breaks_the_closed_form_raises():
    """A sufficiently negative average makes the denominator non-positive."""
    m = mat([[1, "-0.9", "-0.9"], ["-0.9", 1, "-0.9"], ["-0.9", "-0.9", 1]])
    with pytest.raises(ArithmeticError, match="non-positive"):
        effective_observations(m)


# -- the report ------------------------------------------------------------

def test_the_report_carries_both_numbers_and_the_sample_size():
    rng = random.Random(5)
    cols = {f"s{i}": [D(str(round(rng.gauss(0, 1), 8))) for _ in range(120)]
            for i in range(6)}
    rep = breadth_report(correlation_matrix(cols))
    assert rep.names == 6
    assert rep.observations == 120
    assert rep.effective_bets > 0 and rep.effective_observations > 0


def test_the_overstatement_is_the_position_count_over_the_bets():
    rep = BreadthReport(names=500, observations=1000,
                        average_correlation=D("0.4"),
                        effective_bets=D("12.5"),
                        effective_observations=D("2.5"))
    assert rep.overstatement == 40
    assert about(rep.ratio_overstatement, "6.324555", "0.00001")


def test_a_thin_sample_is_flagged_not_corrected():
    rep = BreadthReport(names=500, observations=250,
                        average_correlation=D("0.1"),
                        effective_bets=D("100"),
                        effective_observations=D("9.8"))
    assert rep.thin_sample
    assert not BreadthReport(500, 1000, D("0.1"), D(100),
                             D("9.8")).thin_sample


def test_an_unknown_sample_size_is_not_a_thin_one():
    rep = BreadthReport(names=4, observations=None,
                        average_correlation=D(0), effective_bets=D(4),
                        effective_observations=D(4))
    assert not rep.thin_sample


def test_one_dominant_factor_is_reported():
    rep = breadth_report(equicorrelated(50, "0.8"))
    assert rep.dominated_by_one_factor
    assert not breadth_report(equicorrelated(50, "0.05")
                              ).dominated_by_one_factor


# -- the behaviour that motivated the module -------------------------------

def test_a_one_factor_cross_section_is_a_handful_of_bets():
    """Forty names driven by one market: the count is not the breadth."""
    rng = random.Random(7)
    market = [rng.gauss(0, 1) for _ in range(500)]
    cols = {}
    for i in range(40):
        beta = 0.6 + 0.4 * rng.random()
        cols[f"S{i}"] = [D(str(round(beta * market[t] + rng.gauss(0, 0.8), 8)))
                         for t in range(500)]
    rep = breadth_report(correlation_matrix(cols))
    assert rep.names == 40
    assert D("0.45") < rep.average_correlation < D("0.55")
    assert rep.effective_bets < 5
    assert rep.overstatement > 8
    assert rep.ratio_overstatement > 2


def test_the_count_feeds_a_deflated_t_statistic():
    """The connection the module exists to make."""
    from mdnorm.independence import EffectiveSample, deflate_t_stat

    rep = breadth_report(equicorrelated(100, "0.4"))
    sample = rep.as_sample()
    assert sample.nominal == 100
    assert sample.estimated
    assert sample.effective < 4          # 100 / (1 + 99*0.4)

    honest = deflate_t_stat(D("4.0"), sample)
    naive = deflate_t_stat(D("4.0"), EffectiveSample(nominal=100,
                                                     effective=D(100)))
    assert honest is not None and naive is not None
    assert honest < naive


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (equicorrelated(2, 0),
                BreadthReport(2, None, D(0), D(2), D(2))):
        with pytest.raises(Exception):
            obj.names = 9  # type: ignore[misc]
