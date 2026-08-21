"""Revision tests: what was known when, and the two questions about it."""
from decimal import Decimal

import pytest

from mdnorm import (
    Revision,
    RevisionSeries,
    align,
    read_revisions_csv,
)

D = Decimal
DAY = 86_400 * 1_000_000_000

Q1 = 0
APRIL = 90 * DAY
MAY = 120 * DAY
JUNE = 150 * DAY


def quarterly():
    """One figure for Q1, published in April and revised down in May."""
    return RevisionSeries([
        Revision(event_ts_ns=Q1, known_ts_ns=APRIL, value=D("2.1")),
        Revision(event_ts_ns=Q1, known_ts_ns=MAY, value=D("1.6")),
    ])


# -- one revision ------------------------------------------------------------

def test_a_value_cannot_be_known_before_the_period_it_describes():
    """Something available in advance is a forecast, not a revision."""
    with pytest.raises(ValueError, match="forecast"):
        Revision(event_ts_ns=100, known_ts_ns=99, value=D(1))


def test_known_at_the_same_instant_is_allowed():
    assert Revision(event_ts_ns=100, known_ts_ns=100, value=D(1)).value == D(1)


@pytest.mark.parametrize("kwargs", [
    {"event_ts_ns": -1}, {"known_ts_ns": -1, "event_ts_ns": -5},
])
def test_negative_timestamps_are_rejected(kwargs):
    args = {"event_ts_ns": 1, "known_ts_ns": 1, "value": D(1)}
    args.update(kwargs)
    with pytest.raises(ValueError):
        Revision(**args)


# -- what was known when -----------------------------------------------------

def test_nothing_is_known_before_the_first_release():
    """Not the first release brought forward — nothing."""
    assert quarterly().as_of(event_ts_ns=Q1, known_ts_ns=10 * DAY) is None


def test_the_value_at_release_is_the_first_release():
    assert quarterly().as_of(event_ts_ns=Q1, known_ts_ns=APRIL) == D("2.1")


def test_between_releases_the_older_version_stands():
    assert quarterly().as_of(event_ts_ns=Q1, known_ts_ns=MAY - 1) == D("2.1")


def test_after_the_revision_the_new_value_stands():
    s = quarterly()
    assert s.as_of(event_ts_ns=Q1, known_ts_ns=MAY) == D("1.6")
    assert s.as_of(event_ts_ns=Q1, known_ts_ns=JUNE) == D("1.6")


def test_an_unknown_event_has_no_value():
    s = quarterly()
    assert s.as_of(event_ts_ns=999 * DAY, known_ts_ns=JUNE) is None
    assert s.first_release(999 * DAY) is None and s.final(999 * DAY) is None
    assert s.revision_count(999 * DAY) == 0 and s.published_at(999 * DAY) == ()


# -- first, final, and the difference ----------------------------------------

def test_first_release_and_final_are_different_numbers():
    s = quarterly()
    assert s.first_release(Q1) == D("2.1")
    assert s.final(Q1) == D("1.6")


def test_using_the_final_value_is_look_ahead():
    """The row is dated correctly and the value is still from the future.

    A decision taken on 1 May could only see 2.1. Reading 1.6 into that row
    is look-ahead that no timestamp check will catch, because the timestamp
    is right — it is the version that is wrong.
    """
    s = quarterly()
    decision = MAY - DAY
    assert s.as_of(event_ts_ns=Q1, known_ts_ns=decision) == D("2.1")
    assert s.final(Q1) != s.as_of(event_ts_ns=Q1, known_ts_ns=decision)


def test_republishing_the_same_number_is_not_a_revision():
    s = RevisionSeries([
        Revision(Q1, APRIL, D("2.1")),
        Revision(Q1, MAY, D("2.1")),
        Revision(Q1, JUNE, D("1.9")),
    ])
    assert s.revision_count(Q1) == 1
    assert s.published_at(Q1) == (APRIL, MAY, JUNE)


def test_a_series_that_was_never_revised():
    s = RevisionSeries([Revision(Q1, APRIL, D("2.1"))])
    assert s.revision_count(Q1) == 0
    assert s.first_release(Q1) == s.final(Q1)


def test_two_versions_at_the_same_instant_keep_the_later_input():
    s = RevisionSeries([
        Revision(Q1, APRIL, D("2.1")),
        Revision(Q1, APRIL, D("2.2")),
    ])
    assert s.as_of(event_ts_ns=Q1, known_ts_ns=APRIL) == D("2.2")
    assert s.published_at(Q1) == (APRIL,)


def test_out_of_order_revisions_are_sorted():
    s = RevisionSeries([Revision(Q1, MAY, D("1.6")), Revision(Q1, APRIL, D("2.1"))])
    assert s.first_release(Q1) == D("2.1") and s.final(Q1) == D("1.6")


# -- the two questions -------------------------------------------------------

def test_a_vintage_is_the_table_as_it_looked_that_day():
    s = RevisionSeries([
        Revision(Q1, APRIL, D("2.1")),
        Revision(Q1, MAY, D("1.6")),
        Revision(100 * DAY, JUNE, D("3.0")),
    ])
    april_view = s.vintage_at(APRIL + DAY)
    assert len(april_view) == 1                       # Q2 not out yet
    assert april_view.at(Q1)[0] == D("2.1")

    june_view = s.vintage_at(JUNE + DAY)
    assert len(june_view) == 2
    assert june_view.at(Q1)[0] == D("1.6")            # now the revised figure


def test_the_publication_stream_is_keyed_by_when_you_could_read_it():
    s = quarterly()
    ks = s.known_series()
    assert ks.at(APRIL - 1) == (None, None)
    assert ks.at(APRIL)[0] == D("2.1")
    assert ks.at(MAY)[0] == D("1.6")


def test_a_vintage_read_at_the_wrong_time_reports_a_value_nobody_had():
    """Why the two objects are separate rather than one.

    A vintage is keyed by event time. Ask it for "the latest event at 1 March"
    and it answers with the Q1 figure — which in March had not been published.
    The publication stream gives the honest nothing.
    """
    s = quarterly()
    march = 60 * DAY
    assert s.vintage_at(JUNE).at(march)[0] == D("1.6")     # looks fine, is not
    assert s.known_series().at(march) == (None, None)


def test_a_vintage_only_contains_what_had_been_released():
    s = quarterly()
    assert len(s.vintage_at(APRIL - 1)) == 0


def test_the_publication_stream_carries_every_version():
    s = quarterly()
    assert len(s.known_series()) == 2


# -- the diagnostic ----------------------------------------------------------

def test_the_summary_measures_how_far_corrections_move_things():
    s = RevisionSeries([
        Revision(Q1, APRIL, D("2.1")), Revision(Q1, MAY, D("1.6")),
        Revision(100 * DAY, JUNE, D("3.0")),
    ])
    summary = s.revision_summary()
    assert summary.events == 2
    assert summary.revised_events == 1
    assert summary.max_absolute_change == D("0.5")
    assert summary.mean_absolute_change == D("0.25")
    assert summary.revised_fraction == D("0.5")


def test_a_series_with_no_corrections_reports_zero():
    s = RevisionSeries([Revision(Q1, APRIL, D("2.1"))])
    summary = s.revision_summary()
    assert summary.revised_events == 0
    assert summary.max_absolute_change == 0


def test_an_empty_series_has_nothing_to_summarise():
    summary = RevisionSeries([]).revision_summary()
    assert summary.events == 0
    assert summary.max_absolute_change is None
    assert summary.revised_fraction is None


def test_a_revision_upwards_counts_the_same_as_one_downwards():
    up = RevisionSeries([Revision(Q1, APRIL, D(1)), Revision(Q1, MAY, D(3))])
    down = RevisionSeries([Revision(Q1, APRIL, D(3)), Revision(Q1, MAY, D(1))])
    assert (up.revision_summary().max_absolute_change
            == down.revision_summary().max_absolute_change == D(2))


def test_events_come_back_in_time_order():
    s = RevisionSeries([Revision(100 * DAY, JUNE, D(1)), Revision(Q1, APRIL, D(2))])
    assert s.events == (Q1, 100 * DAY)
    assert len(s) == 2


# -- reading from a file -----------------------------------------------------

def test_revisions_are_read_from_csv(tmp_path):
    p = tmp_path / "rev.csv"
    p.write_text("event,known,value\n"
                 "2026-01-01T00:00:00Z,2026-04-01T00:00:00Z,2.1\n"
                 "2026-01-01T00:00:00Z,2026-05-01T00:00:00Z,1.6\n")
    s = RevisionSeries(read_revisions_csv(str(p)))
    assert s.revision_count(s.events[0]) == 1
    assert s.final(s.events[0]) == D("1.6")


def test_the_event_ts_spelling_is_accepted_too(tmp_path):
    p = tmp_path / "rev.csv"
    p.write_text("event_ts,known_ts,value\n1000,2000,5\n")
    r = read_revisions_csv(str(p), ts_unit="s")[0]
    assert (r.event_ts_ns, r.known_ts_ns, r.value) == (10**12, 2 * 10**12, D(5))


def test_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "rev.csv"
    p.write_text("event,known,value\n1000,2000,5\n,,\n")
    assert len(read_revisions_csv(str(p), ts_unit="s")) == 1


def test_a_bad_revision_row_names_its_line(tmp_path):
    p = tmp_path / "rev.csv"
    p.write_text("event,known,value\n2000,1000,5\n")
    with pytest.raises(ValueError, match=":2:"):
        read_revisions_csv(str(p), ts_unit="s")


def test_an_unparseable_value_names_its_line(tmp_path):
    p = tmp_path / "rev.csv"
    p.write_text("event,known,value\n1000,2000,not-a-number\n")
    with pytest.raises(ValueError, match=":2:"):
        read_revisions_csv(str(p), ts_unit="s")


# -- integration -------------------------------------------------------------

def test_the_publication_stream_joins_like_any_other_series():
    """The point of returning an AsOfSeries: it goes straight into align."""
    s = quarterly()
    rows = align({"gdp": s.known_series()},
                 interval_ns=30 * DAY, start_ns=0, end_ns=181 * DAY)
    values = {r.ts_ns // DAY: r.values["gdp"] for r in rows}
    assert values[60] is None            # March: nothing published yet
    assert values[90] == D("2.1")        # April: first release
    assert values[120] == D("1.6")       # May: revised


def test_a_delayed_feed_and_a_revised_one_are_different_problems():
    """A delay shifts one value; a revision replaces it.

    mdnorm.align.AsOfSeries.delayed models arrival lag on a value that never
    changes. This module handles the case where the value itself is corrected,
    which no amount of shifting can express.
    """
    s = quarterly()
    stream = s.known_series()
    assert stream.at(APRIL)[0] == D("2.1")
    assert stream.at(MAY)[0] == D("1.6")
    delayed = stream.delayed(DAY)
    assert delayed.at(APRIL) == (None, None)
    assert delayed.at(APRIL + DAY)[0] == D("2.1")
