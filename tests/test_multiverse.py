"""Multiverse tests: the grid of defensible pipelines, and what moves it."""
from decimal import Decimal

import pytest

from mdnorm.multiverse import (
    Choice,
    ChoiceEffect,
    SpecCurve,
    SpecResult,
    Specification,
    choice_effect,
    dominant_choice,
    explore,
    specifications,
)

D = Decimal


def spec(**labels):
    """A Specification whose values equal its labels — enough for summaries."""
    return Specification(
        selections=tuple(labels.items()),
        assignments=tuple(labels.items()),
    )


def curve(*pairs):
    return SpecCurve(results=tuple(
        SpecResult(specification=s, value=v) for s, v in pairs))


# -- the choice ------------------------------------------------------------

def test_a_choice_keeps_its_options_in_order():
    c = Choice("clip", [("none", None), ("5 sigma", D(5))])
    assert c.labels == ("none", "5 sigma")
    assert c.options[1][1] == D(5)


def test_a_choice_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Choice("", [("a", 1)])


def test_a_choice_with_no_options_is_refused():
    with pytest.raises(ValueError, match="was not a decision"):
        Choice("clip", [])


def test_a_choice_cannot_list_a_label_twice():
    with pytest.raises(ValueError, match="option label twice"):
        Choice("clip", [("none", None), ("none", 0)])


def test_a_choice_does_not_inspect_its_values():
    """A value can be anything the caller's pipeline understands."""
    f = lambda x: x
    c = Choice("how", [("identity", f), ("nothing", None), ("flag", True)])
    assert c.options[0][1] is f


# -- the grid --------------------------------------------------------------

def test_the_grid_is_every_combination():
    got = specifications([
        Choice("a", [("a1", 1), ("a2", 2)]),
        Choice("b", [("b1", 10), ("b2", 20), ("b3", 30)]),
    ])
    assert len(got) == 6
    assert {str(s) for s in got} == {
        "a=a1, b=b1", "a=a1, b=b2", "a=a1, b=b3",
        "a=a2, b=b1", "a=a2, b=b2", "a=a2, b=b3"}


def test_the_grid_multiplies():
    """Six binary decisions are sixty-four pipelines, which is the point."""
    got = specifications([Choice(f"c{i}", [("off", 0), ("on", 1)])
                          for i in range(6)])
    assert len(got) == 64


def test_a_specification_carries_both_labels_and_values():
    [s] = specifications([Choice("clip", [("5 sigma", D(5))])])
    assert s.labels == {"clip": "5 sigma"}
    assert s.values == {"clip": D(5)}
    assert s["clip"] == D(5)


def test_the_order_is_stable():
    choices = [Choice("a", [("a1", 1), ("a2", 2)]),
               Choice("b", [("b1", 1), ("b2", 2)])]
    assert [str(s) for s in specifications(choices)] == \
           [str(s) for s in specifications(choices)]


def test_an_empty_grid_is_refused():
    with pytest.raises(ValueError, match="at least one choice"):
        specifications([])


def test_a_choice_name_cannot_repeat():
    with pytest.raises(ValueError, match="listed twice"):
        specifications([Choice("clip", [("a", 1)]),
                        Choice("clip", [("b", 2)])])


# -- running it ------------------------------------------------------------

def test_every_specification_is_evaluated():
    specs = specifications([Choice("k", [("one", 1), ("two", 2),
                                         ("three", 3)])])
    got = explore(specs, lambda s: D(s["k"]) * 10)
    assert [r.value for r in got.results] == [D(10), D(20), D(30)]


def test_a_pipeline_that_raises_records_none_and_the_sweep_continues():
    """Losing sixty-three results to one bad cell would be a poor trade."""
    def run(s):
        if s["k"] == 2:
            raise ValueError("cannot clean it that way")
        return D(s["k"])

    got = explore(specifications([Choice("k", [("a", 1), ("b", 2),
                                               ("c", 3)])]), run)
    assert [r.value for r in got.results] == [D(1), None, D(3)]
    assert got.answered == 2 and got.unanswered == 1


def test_a_lookup_error_in_the_pipeline_is_caught_too():
    got = explore(specifications([Choice("k", [("a", 1)])]),
                  lambda s: s["missing"])
    assert got.results[0].value is None


def test_an_unexpected_exception_is_not_swallowed():
    """A bug in the caller's code should surface, not become a None."""
    def run(s):
        raise RuntimeError("this is a bug, not a bad cell")

    with pytest.raises(RuntimeError):
        explore(specifications([Choice("k", [("a", 1)])]), run)


def test_running_no_specifications_is_refused():
    with pytest.raises(ValueError, match="at least one specification"):
        explore([], lambda s: D(1))


# -- the summaries ---------------------------------------------------------

def test_the_spread_is_the_range_across_pipelines():
    c = curve((spec(a="1"), D("0.44")), (spec(a="2"), D("1.21")),
              (spec(a="3"), D("0.80")))
    assert c.lowest == D("0.44")
    assert c.highest == D("1.21")
    assert c.spread == D("0.77")
    assert c.median == D("0.80")


def test_the_median_of_an_even_count_is_the_midpoint():
    c = curve((spec(a="1"), D(1)), (spec(a="2"), D(2)),
              (spec(a="3"), D(3)), (spec(a="4"), D(4)))
    assert c.median == D("2.5")


def test_none_values_are_left_out_rather_than_counted_as_zero():
    c = curve((spec(a="1"), None), (spec(a="2"), D(4)), (spec(a="3"), D(6)))
    assert c.values == (D(4), D(6))
    assert c.median == D(5)
    assert c.answered == 2 and c.unanswered == 1


def test_a_sign_change_is_reported():
    c = curve((spec(a="1"), D("-0.2")), (spec(a="2"), D("0.9")))
    assert c.changes_sign
    assert c.positive == 1
    assert c.share_positive == D("0.5")


def test_one_sided_results_do_not_change_sign():
    c = curve((spec(a="1"), D(1)), (spec(a="2"), D(2)))
    assert not c.changes_sign
    assert c.share_positive == 1


def test_everything_is_none_when_nothing_was_answered():
    c = curve((spec(a="1"), None), (spec(a="2"), None))
    assert c.values == ()
    assert c.lowest is None and c.highest is None
    assert c.median is None and c.spread is None
    assert c.share_positive is None
    assert not c.changes_sign
    assert c.best is None and c.worst is None
    assert c.trials == 0


def test_the_best_and_worst_name_their_specifications():
    c = curve((spec(clip="none"), D("0.44")),
              (spec(clip="5 sigma"), D("1.21")))
    assert c.best.specification.labels["clip"] == "5 sigma"
    assert c.worst.specification.labels["clip"] == "none"


def test_the_names_come_from_the_grid_in_order():
    got = explore(specifications([Choice("clip", [("a", 1)]),
                                  Choice("stale", [("b", 2)])]),
                  lambda s: D(1))
    assert got.names == ("clip", "stale")


def test_an_empty_curve_has_no_names():
    assert SpecCurve(results=()).names == ()


# -- the trial count, and its caveat ---------------------------------------

def test_the_trial_count_is_the_number_that_answered():
    c = curve((spec(a="1"), D(1)), (spec(a="2"), None), (spec(a="3"), D(3)))
    assert c.trials == 2


def test_the_trial_count_feeds_a_deflated_sharpe():
    """The connection the module exists to make."""
    from mdnorm.metrics import deflated_sharpe_ratio, trial_variance

    c = curve(*[(spec(a=str(i)), D(f"0.0{i + 1}")) for i in range(16)])
    var = trial_variance(list(c.values))
    best = c.highest
    naive = deflated_sharpe_ratio(best, observations=1000, trials=1,
                                  variance=var)
    honest = deflated_sharpe_ratio(best, observations=1000, trials=c.trials,
                                   variance=var)
    assert honest <= naive


# -- attribution -----------------------------------------------------------

def test_a_choice_effect_is_the_median_under_each_option():
    c = curve((spec(clip="none", stale="keep"), D(10)),
              (spec(clip="none", stale="drop"), D(20)),
              (spec(clip="5 sigma", stale="keep"), D(2)),
              (spec(clip="5 sigma", stale="drop"), D(4)))
    e = choice_effect(c, "clip")
    assert e.median_map == {"none": D(15), "5 sigma": D(3)}
    assert e.spread == D(12)


def test_an_option_nothing_answered_for_carries_none_not_zero():
    c = curve((spec(clip="none"), D(10)), (spec(clip="5 sigma"), None))
    e = choice_effect(c, "clip")
    assert e.median_map == {"none": D(10), "5 sigma": None}
    assert e.spread is None          # one option left, nothing to compare


def test_asking_about_a_choice_that_is_not_there_raises():
    c = curve((spec(clip="none"), D(1)))
    with pytest.raises(ValueError, match="no choice named"):
        choice_effect(c, "stale")


def test_the_dominant_choice_is_the_one_that_moves_the_median_most():
    c = curve((spec(clip="none", stale="keep"), D(10)),
              (spec(clip="none", stale="drop"), D(11)),
              (spec(clip="5 sigma", stale="keep"), D(1)),
              (spec(clip="5 sigma", stale="drop"), D(2)))
    d = dominant_choice(c)
    assert d.name == "clip"
    assert d.spread == D(9)


def test_no_dominant_choice_when_every_option_lands_on_the_same_median():
    """A real outcome: the decisions interact, none of them owns the spread."""
    c = curve((spec(a="x", b="p"), D(1)), (spec(a="x", b="q"), D(3)),
              (spec(a="y", b="p"), D(3)), (spec(a="y", b="q"), D(1)))
    assert dominant_choice(c) is None


def test_no_dominant_choice_when_nothing_was_answered():
    assert dominant_choice(curve((spec(a="x"), None))) is None


# -- the behaviour that motivated the module -------------------------------

def test_defensible_pipelines_disagree_about_the_headline():
    """Three binary cleaning decisions, eight pipelines, one published number."""
    values = {("none", "keep", "ordinary"): D("1.21"),
              ("none", "keep", "robust"): D("1.02"),
              ("none", "drop", "ordinary"): D("0.98"),
              ("none", "drop", "robust"): D("0.81"),
              ("5 sigma", "keep", "ordinary"): D("0.74"),
              ("5 sigma", "keep", "robust"): D("0.66"),
              ("5 sigma", "drop", "ordinary"): D("0.55"),
              ("5 sigma", "drop", "robust"): D("0.44")}
    specs = specifications([
        Choice("clip", [("none", 0), ("5 sigma", 5)]),
        Choice("stale", [("keep", False), ("drop", True)]),
        Choice("scale", [("ordinary", False), ("robust", True)]),
    ])
    c = explore(specs, lambda s: values[tuple(s.labels[n] for n in
                                              ("clip", "stale", "scale"))])
    assert c.trials == 8
    assert c.highest == D("1.21") and c.lowest == D("0.44")
    assert c.spread == D("0.77")
    assert not c.changes_sign
    assert dominant_choice(c).name == "clip"


def test_the_flattering_cell_is_identifiable():
    """Which combination produced the number somebody wanted to publish."""
    values = {("none", "keep"): D("1.21"), ("none", "drop"): D("0.90"),
              ("5 sigma", "keep"): D("0.70"), ("5 sigma", "drop"): D("0.44")}
    specs = specifications([Choice("clip", [("none", 0), ("5 sigma", 5)]),
                            Choice("stale", [("keep", 0), ("drop", 1)])])
    c = explore(specs, lambda s: values[(s.labels["clip"],
                                         s.labels["stale"])])
    assert str(c.best.specification) == "clip=none, stale=keep"


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Choice("a", [("x", 1)]), spec(a="x"),
                SpecResult(spec(a="x"), None), SpecCurve(()),
                ChoiceEffect("a", ())):
        with pytest.raises(Exception):
            obj.name = "z"  # type: ignore[misc]
