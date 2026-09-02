"""Seasonality tests: the shape of a day, and not reading the rest of the year."""
import datetime as dt
from datetime import date, time
from decimal import Decimal

import pytest

from mdnorm import Session, session_bounds
from mdnorm.calendars import EarlyClose, TradingCalendar
from mdnorm.seasonality import (
    ProfileBucket,
    SessionProfile,
    ProfileLeak,
    Sample,
    bucket_index,
    deseasonalise,
    expanding_profiles,
    full_sample_deseasonalise,
    profile_leak,
    read_samples_csv,
    session_profile,
)

D = Decimal
S = 1_000_000_000
MIN = 60 * S
HOUR = 3600 * S

RTH = Session(start=time(9, 30), end=time(16, 0), tz="America/New_York")
FLAT = Session(start=time(0, 0), end=time(4, 0), tz="UTC")   # 4 hours, easy maths


def weekdays(start, n):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def build(days, per_bucket, session=FLAT, bucket_ns=HOUR, offset_ns=60 * S):
    """One sample per bucket per day; ``per_bucket(day_index, bucket)`` gives it."""
    out = []
    for k, day in enumerate(days):
        open_ns, close_ns = session_bounds(day, session)
        n = (close_ns - open_ns) // bucket_ns
        for i in range(n):
            v = per_bucket(k, i)
            if v is None:
                continue
            out.append(Sample(open_ns + i * bucket_ns + offset_ns, D(str(v))))
    return out


DAYS = weekdays(date(2026, 1, 5), 20)
SHAPE = {0: 40, 1: 10, 2: 10, 3: 20}       # a four-hour day with a heavy open


# -- bucket_index ----------------------------------------------------------

def test_bucket_index_counts_from_the_open_not_from_midnight():
    open_ns, _ = session_bounds(date(2026, 1, 5), FLAT)
    assert bucket_index(open_ns, FLAT, HOUR) == 0
    assert bucket_index(open_ns + HOUR, FLAT, HOUR) == 1
    assert bucket_index(open_ns + 3 * HOUR + 59 * MIN, FLAT, HOUR) == 3


def test_bucket_index_is_none_outside_the_session():
    open_ns, close_ns = session_bounds(date(2026, 1, 5), FLAT)
    assert bucket_index(open_ns - 1, FLAT, HOUR) is None
    assert bucket_index(close_ns, FLAT, HOUR) is None


def test_bucket_index_refuses_a_non_positive_bucket():
    with pytest.raises(ValueError, match="positive"):
        bucket_index(0, FLAT, 0)


def test_buckets_stay_aligned_to_the_open_across_daylight_saving():
    """A US session opens at 09:30 local on both sides of the DST change."""
    before, after = date(2026, 3, 6), date(2026, 3, 12)
    for day in (before, after):
        open_ns, _ = session_bounds(day, RTH)
        assert bucket_index(open_ns + 5 * MIN, RTH, 30 * MIN) == 0
        assert bucket_index(open_ns + 35 * MIN, RTH, 30 * MIN) == 1


# -- profile ---------------------------------------------------------------

def test_profile_has_one_bucket_per_slot_of_the_session():
    p = session_profile(build(DAYS, lambda k, i: SHAPE[i]), FLAT, bucket_ns=HOUR)
    assert len(p) == 4
    assert [b.index for b in p.buckets] == [0, 1, 2, 3]
    assert p.sessions == 20


def test_a_session_that_does_not_divide_evenly_gets_a_final_short_bucket():
    p = session_profile(build(DAYS, lambda k, i: 1), FLAT, bucket_ns=HOUR)
    assert len(p) == 4
    p90 = session_profile(build(DAYS, lambda k, i: 1), FLAT, bucket_ns=90 * MIN)
    assert len(p90) == 3          # 4 hours over 90 minutes rounds up


def test_profile_values_are_the_mean_of_each_bucket():
    p = session_profile(build(DAYS, lambda k, i: SHAPE[i]), FLAT, bucket_ns=HOUR)
    assert [b.value for b in p.buckets] == [D(40), D(10), D(10), D(20)]
    assert p.level == D(20)


def test_factor_is_relative_to_the_session_average():
    p = session_profile(build(DAYS, lambda k, i: SHAPE[i]), FLAT, bucket_ns=HOUR)
    assert p.factor_at(0) == D(2)                # the open is twice the day
    assert p.factor_at(HOUR) == D("0.5")
    assert p.factor_at(3 * HOUR) == D(1)


def test_factor_ignores_a_level_shift_because_it_is_a_ratio():
    plain = session_profile(build(DAYS, lambda k, i: SHAPE[i]), FLAT, bucket_ns=HOUR)
    scaled = session_profile(build(DAYS, lambda k, i: SHAPE[i] * 1000), FLAT,
                     bucket_ns=HOUR)
    assert plain.factor_at(0) == scaled.factor_at(0)


def test_a_thin_bucket_reports_nothing_rather_than_the_average():
    """The flattering fill would make the adjusted series look smooth here."""
    samples = build(DAYS, lambda k, i: SHAPE[i] if i != 2 or k < 3 else None)
    p = session_profile(samples, FLAT, bucket_ns=HOUR, min_observations=5)
    assert p.buckets[2].observations == 3
    assert p.buckets[2].value is None
    assert p.factor_at(2 * HOUR) is None
    assert p.empty_buckets == 1
    assert not p.complete


def test_an_empty_bucket_is_left_out_of_the_level_rather_than_counted_as_zero():
    samples = build(DAYS, lambda k, i: SHAPE[i] if i != 2 else None)
    p = session_profile(samples, FLAT, bucket_ns=HOUR)
    assert p.level == (D(40) + D(10) + D(20)) / 3     # not divided by four


def test_a_bucket_of_zeroes_is_not_an_empty_bucket():
    samples = build(DAYS, lambda k, i: 0 if i == 2 else SHAPE[i])
    p = session_profile(samples, FLAT, bucket_ns=HOUR)
    assert p.buckets[2].value == D(0)
    assert p.buckets[2].observations == 20
    assert p.complete


def test_profile_refuses_a_non_positive_bucket_or_threshold():
    with pytest.raises(ValueError, match="positive"):
        session_profile([], FLAT, bucket_ns=0)
    with pytest.raises(ValueError, match="at least 1"):
        session_profile([], FLAT, bucket_ns=HOUR, min_observations=0)


def test_samples_outside_the_session_do_not_reach_the_profile():
    open_ns, close_ns = session_bounds(DAYS[0], FLAT)
    samples = build(DAYS, lambda k, i: SHAPE[i]) + [
        Sample(open_ns - HOUR, D(999)), Sample(close_ns + HOUR, D(999))]
    p = session_profile(samples, FLAT, bucket_ns=HOUR)
    assert [b.value for b in p.buckets] == [D(40), D(10), D(10), D(20)]


def test_at_and_factor_at_are_none_past_the_end_of_the_day():
    p = session_profile(build(DAYS, lambda k, i: SHAPE[i]), FLAT, bucket_ns=HOUR)
    assert p.at(4 * HOUR) is None
    assert p.factor_at(4 * HOUR) is None
    assert p.at(-1) is None


# -- expanding_profiles ----------------------------------------------------

def test_nothing_is_emitted_until_there_is_enough_history():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    got = list(expanding_profiles(samples, FLAT, bucket_ns=HOUR,
                                  min_sessions=5))
    assert [d for d, _ in got] == DAYS[5:]
    assert len(got) == 15


def test_min_sessions_of_one_still_skips_the_very_first_day():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    got = list(expanding_profiles(samples, FLAT, bucket_ns=HOUR,
                                  min_sessions=1))
    assert [d for d, _ in got] == DAYS[1:]


def test_a_day_never_appears_in_its_own_profile():
    """The whole point of the module, as a test.

    One day is given a wildly different shape. The profile handed to that day
    must be identical to the profile of the days before it, and must not move
    at all because of the day it is about to be applied to.
    """
    def shape(k, i):
        return 10_000 if (k == 12 and i == 1) else SHAPE[i]

    samples = build(DAYS, shape)
    got = dict(expanding_profiles(samples, FLAT, bucket_ns=HOUR,
                                  min_sessions=5))
    before = session_profile(build(DAYS[:12], lambda k, i: SHAPE[i]), FLAT,
                     bucket_ns=HOUR)
    assert got[DAYS[12]].buckets == before.buckets
    assert got[DAYS[12]].factor_at(HOUR) == D("0.5")
    # The day after does see it.
    assert got[DAYS[13]].factor_at(HOUR) > D(1)


def test_each_expanding_profile_equals_a_full_profile_of_the_earlier_days():
    import random
    random.seed(31337)
    values = {(k, i): random.randint(1, 500) for k in range(20) for i in range(4)}
    samples = build(DAYS, lambda k, i: values[(k, i)])
    for day, shape in expanding_profiles(samples, FLAT, bucket_ns=HOUR,
                                         min_sessions=4):
        cut = DAYS.index(day)
        expected = session_profile(build(DAYS[:cut], lambda k, i: values[(k, i)]),
                           FLAT, bucket_ns=HOUR)
        assert shape.buckets == expected.buckets
        assert shape.sessions == cut


def test_expanding_profiles_refuses_a_non_positive_min_sessions():
    with pytest.raises(ValueError, match="at least 1"):
        list(expanding_profiles([], FLAT, bucket_ns=HOUR, min_sessions=0))


# -- deseasonalise ---------------------------------------------------------

def test_a_perfectly_regular_day_flattens_to_its_own_level():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    out = deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    assert {s.value for s in out} == {D(20)}        # every point is the level


def test_the_first_sessions_are_dropped_rather_than_adjusted_by_noise():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    out = deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    assert len(out) == 15 * 4
    open_ns, _ = session_bounds(DAYS[4], FLAT)
    assert all(s.ts_ns >= open_ns for s in out)


def test_a_sample_with_no_factor_is_dropped_not_passed_through_unchanged():
    """A point silently divided by one is a point claiming to be adjusted.

    The third bucket only exists on the last two days, so for most of the
    sample its expanding profile cannot clear ``min_observations`` and has no
    factor to give. Those points leave the output rather than passing through
    at their raw value dressed as an adjusted one.
    """
    samples = build(DAYS, lambda k, i: SHAPE[i] if i != 2 or k >= 18 else None)
    out = deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5,
                        min_observations=3)
    assert len(out) == 15 * 3          # the third bucket never survives
    assert all(s.value == D(70) / 3 for s in out)


def test_the_full_sample_version_adjusts_every_day_including_the_first():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    out = full_sample_deseasonalise(samples, FLAT, bucket_ns=HOUR)
    assert len(out) == 20 * 4


def test_the_two_versions_agree_when_the_shape_never_changes():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    pit = deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    full = {s.ts_ns: s.value for s in
            full_sample_deseasonalise(samples, FLAT, bucket_ns=HOUR)}
    assert all(full[s.ts_ns] == s.value for s in pit)


def test_the_two_versions_disagree_when_the_shape_moves():
    """A closing auction that grows through the sample, which really happens."""
    def shape(k, i):
        return SHAPE[i] + (60 if i == 3 and k >= 10 else 0)

    samples = build(DAYS, shape)
    pit = {s.ts_ns: s.value for s in
           deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5)}
    full = {s.ts_ns: s.value for s in
            full_sample_deseasonalise(samples, FLAT, bucket_ns=HOUR)}
    assert any(full[ts] != v for ts, v in pit.items())


# -- profile_leak -----------------------------------------------------------

def test_no_leak_when_every_day_has_the_same_shape():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    r = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    assert r.differ == 0
    assert r.max_gap == 0
    assert r.differing_fraction == 0


def test_a_shape_that_moves_shows_up_as_a_leak():
    def shape(k, i):
        return SHAPE[i] + (60 if i == 3 and k >= 10 else 0)

    samples = build(DAYS, shape)
    r = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    assert r.differ > 0
    assert r.max_gap is not None and r.max_gap > D("0.1")
    assert r.worst_ts_ns is not None


def test_the_worst_leak_is_early_where_the_most_future_is_borrowed():
    def shape(k, i):
        return SHAPE[i] + (60 if i == 3 and k >= 10 else 0)

    samples = build(DAYS, shape)
    r = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    open_ns, _ = session_bounds(DAYS[10], FLAT)
    assert r.worst_ts_ns < open_ns


def test_leak_report_counts_only_what_both_profiles_could_adjust():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    r = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5)
    assert r.samples == 20 * 4
    assert r.knowable == 15 * 4


def test_tolerance_moves_the_count_but_never_the_max():
    def shape(k, i):
        return SHAPE[i] + (60 if i == 3 and k >= 10 else 0)

    samples = build(DAYS, shape)
    tight = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5,
                        tolerance=D("0.001"))
    loose = profile_leak(samples, FLAT, bucket_ns=HOUR, min_sessions=5,
                        tolerance=D("0.5"))
    assert tight.differ > loose.differ
    assert tight.max_gap == loose.max_gap


def test_an_empty_input_reports_nothing_rather_than_zero_leak():
    r = profile_leak([], FLAT, bucket_ns=HOUR, min_sessions=5)
    assert r.knowable == 0
    assert r.differing_fraction is None
    assert r.max_gap is None


# -- calendars -------------------------------------------------------------

def _calendar(days, early=None):
    return TradingCalendar(
        FLAT,
        first_day=days[0],
        last_day=days[-1],
        early_closes=tuple(EarlyClose(day, time(2, 0)) for day in (early or ())),
    )


def test_an_early_close_day_is_left_out_and_counted():
    short = DAYS[7]
    samples = build(DAYS, lambda k, i: SHAPE[i])
    cal = _calendar(DAYS, early=[short])
    p = session_profile(samples, FLAT, bucket_ns=HOUR, calendar=cal)
    assert p.sessions == 19
    assert p.excluded == 1


def test_include_short_keeps_it_when_the_caller_insists():
    short = DAYS[7]
    samples = build(DAYS, lambda k, i: SHAPE[i])
    cal = _calendar(DAYS, early=[short])
    p = session_profile(samples, FLAT, bucket_ns=HOUR, calendar=cal,
                include_short=True)
    assert p.sessions == 20
    assert p.excluded == 0


def test_a_half_days_close_does_not_contaminate_the_middle_of_the_day():
    """The reason short sessions are excluded, stated as a test.

    On the short day the closing surge lands in bucket 1, which on every
    other day is a quiet part of the morning.
    """
    short = DAYS[7]

    def shape(k, i):
        if DAYS[k] == short:
            return 400 if i == 1 else (SHAPE[i] if i < 2 else None)
        return SHAPE[i]

    samples = build(DAYS, shape)
    cal = _calendar(DAYS, early=[short])
    clean = session_profile(samples, FLAT, bucket_ns=HOUR, calendar=cal)
    mixed = session_profile(samples, FLAT, bucket_ns=HOUR)
    assert clean.buckets[1].value == D(10)
    assert mixed.buckets[1].value > D(10)


def test_no_calendar_means_every_day_is_treated_as_full_length():
    samples = build(DAYS, lambda k, i: SHAPE[i])
    p = session_profile(samples, FLAT, bucket_ns=HOUR)
    assert p.sessions == 20
    assert p.excluded == 0


# -- CSV -------------------------------------------------------------------

def test_read_samples_csv(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("ts_ns,value\n1000,12.5\n2000,13\n", encoding="utf-8")
    rows = read_samples_csv(str(p))
    assert [r.ts_ns for r in rows] == [1000, 2000]
    assert rows[0].value == D("12.5")


def test_a_row_missing_a_value_is_an_error_not_a_skip(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("ts_ns,value\n1000,12.5\n2000,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_samples_csv(str(p))


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("ts_ns,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        read_samples_csv(str(p))


def test_custom_columns(tmp_path):
    p = tmp_path / "v.csv"
    p.write_text("t,vol\n5,7\n", encoding="utf-8")
    rows = read_samples_csv(str(p), ts_column="t", value_column="vol")
    assert rows[0] == Sample(5, D(7))


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Sample(1, D(1)), ProfileBucket(0, 0, 1, 1, D(1)),
                SessionProfile(1, (), 0), ProfileLeak(0, 0, 0, None, None, None)):
        with pytest.raises(Exception):
            obj.index = 5  # type: ignore[misc]


def test_the_adjustment_rounds_once_rather_than_twice():
    """Dividing by a factor that is itself a quotient rounds twice.

    Adjusting a value that already equals its bucket's mean should give back
    the profile's level exactly. It does under one division and misses by a
    last digit under two, which is the reason the implementation rearranges
    the arithmetic rather than reusing ``factor_at``.
    """
    samples = build(DAYS, lambda k, i: SHAPE[i] if i != 2 or k >= 18 else None)
    out = deseasonalise(samples, FLAT, bucket_ns=HOUR, min_sessions=5,
                        min_observations=3)
    assert len(out) == 15 * 3
    assert {s.value for s in out} == {D(70) / 3}
