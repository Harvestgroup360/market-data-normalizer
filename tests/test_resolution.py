"""Resolution tests: what a feed's timestamps can and cannot distinguish."""
from decimal import Decimal

import pytest

from mdnorm import EventType, MarketEvent
from mdnorm.resolution import (
    ClassificationRisk,
    LADDER,
    Resolution,
    classification_risk,
    detect_resolution,
    order_is_determined,
    read_timestamps_csv,
    tie_groups,
)

D = Decimal
US = 1_000
MS = 1_000_000
S = 1_000_000_000
BASE = 1_700_000_000 * S


def stamps(unit, n=200, start=BASE):
    return [start + i * unit for i in range(n)]


def quote(ts, bid="100.00", ask="100.02"):
    return MarketEvent(symbol="X", venue="v", ts_ns=ts,
                       event_type=EventType.QUOTE,
                       bid_price=D(bid), ask_price=D(ask))


def trade(ts, price):
    return MarketEvent(symbol="X", venue="v", ts_ns=ts,
                       event_type=EventType.TRADE,
                       price=D(price), size=D(1))


# -- detection -------------------------------------------------------------

@pytest.mark.parametrize("unit", LADDER)
def test_every_rung_of_the_ladder_is_detected(unit):
    assert detect_resolution(stamps(unit)).granularity_ns == unit


def test_the_coarsest_divisor_wins_not_the_first():
    """Millisecond stamps divide by a microsecond too; the answer is the ms."""
    r = detect_resolution(stamps(MS))
    assert r.granularity_ns == MS


def test_one_odd_timestamp_drops_the_answer_to_the_nanosecond():
    ts = stamps(MS) + [BASE + 7]
    assert detect_resolution(ts).granularity_ns == 1


def test_a_nanosecond_feed_reads_as_a_nanosecond_feed():
    ts = [BASE + i * 7 + (i % 3) for i in range(200)]
    assert detect_resolution(ts).granularity_ns == 1


def test_overstated_digits_counts_the_padding():
    assert detect_resolution(stamps(MS)).overstated_digits == 6
    assert detect_resolution(stamps(US)).overstated_digits == 3
    assert detect_resolution(stamps(1)).overstated_digits == 0


def test_too_few_observations_is_undetermined_not_one_nanosecond():
    r = detect_resolution(stamps(MS, n=5))
    assert r.undetermined
    assert r.granularity_ns is None
    assert r.overstated_digits is None


def test_the_threshold_counts_distinct_timestamps_not_rows():
    """A thousand copies of one round number is one reading of the clock."""
    r = detect_resolution([BASE] * 1000)
    assert r.observations == 1000
    assert r.distinct == 1
    assert r.undetermined


def test_min_observations_is_adjustable_and_validated():
    assert detect_resolution(stamps(MS, n=5), min_observations=5
                             ).granularity_ns == MS
    with pytest.raises(ValueError, match="at least 1"):
        detect_resolution([], min_observations=0)


def test_zero_divides_by_everything_and_is_therefore_no_evidence():
    r = detect_resolution([0] * 50)
    assert r.observations == 50
    assert r.undetermined


def test_an_empty_input_is_undetermined():
    r = detect_resolution([])
    assert r.observations == 0 and r.distinct == 0
    assert r.undetermined
    assert r.tied_share is None


def test_negative_timestamps_are_refused():
    with pytest.raises(ValueError, match="non-negative"):
        detect_resolution([1, -1])


# -- ties ------------------------------------------------------------------

def test_ties_are_counted_by_group_and_by_observation():
    ts = stamps(MS, n=100) + stamps(MS, n=30)
    r = detect_resolution(ts)
    assert r.ties == 30                 # thirty values appear twice
    assert r.tied_observations == 60    # sixty rows are involved
    assert r.largest_tie == 2
    assert r.distinct == 100


def test_tied_share_is_of_observations_not_of_distinct_values():
    ts = stamps(MS, n=100) + stamps(MS, n=30)
    assert detect_resolution(ts).tied_share == D(60) / 130


def test_a_feed_with_no_repeats_has_no_ties():
    r = detect_resolution(stamps(MS))
    assert r.ties == 0 and r.tied_observations == 0
    assert r.tied_share == 0
    assert r.largest_tie == 1


def test_tie_groups_lists_only_repeats_in_timestamp_order():
    assert list(tie_groups([9, 5, 5, 7, 9, 9])) == [(5, 2), (9, 3)]


def test_tie_groups_is_empty_when_everything_is_distinct():
    assert list(tie_groups([1, 2, 3])) == []


# -- ordering --------------------------------------------------------------

def test_two_events_in_one_tick_have_no_determined_order():
    assert not order_is_determined(BASE, BASE + 999_999, MS)
    assert order_is_determined(BASE, BASE + MS, MS)


def test_differing_timestamps_can_still_be_undetermined():
    """The whole point: the difference is finer than the clock that made it."""
    assert BASE + 400_000 != BASE + 900_000
    assert not order_is_determined(BASE + 400_000, BASE + 900_000, MS)


def test_order_is_determined_refuses_a_non_positive_granularity():
    with pytest.raises(ValueError, match="positive"):
        order_is_determined(1, 2, 0)


# -- classification risk ---------------------------------------------------

def test_quotes_in_earlier_ticks_carry_no_risk():
    events = []
    for i in range(60):
        events.append(quote(BASE + i * MS))
        events.append(trade(BASE + i * MS + 500_000, "100.02"))
    # Stamped at the microsecond, so the trade and its quote are separable.
    r = classification_risk(events, granularity_ns=US)
    assert r.trades == 60
    assert r.same_tick == 0
    assert r.changed == 0


def test_a_quote_in_the_trades_own_tick_is_counted_as_exposure():
    events = []
    for i in range(60):
        events.append(quote(BASE + i * MS))
        events.append(trade(BASE + i * MS, "100.02"))
    r = classification_risk(events, granularity_ns=MS)
    assert r.trades == 60
    assert r.same_tick == 60


def test_changed_counts_only_the_trades_whose_side_actually_moves():
    """A trade above one quote's mid can be below the previous quote's mid.

    The quote alternates between two levels each millisecond and the trade
    prints between the two mids, so every trade is classified one way against
    its own tick's quote and the other way against the last quote that is
    provably earlier.
    """
    low, high = ("100.00", "100.02"), ("100.04", "100.06")   # mids 100.01, 100.05
    events = [quote(BASE, *high)]
    for i in range(1, 41):
        ts = BASE + i * MS
        events.append(quote(ts, *(high if i % 2 == 0 else low)))
        events.append(trade(ts, "100.03"))
    r = classification_risk(events, granularity_ns=MS)
    assert r.same_tick == 40
    assert r.changed == 40


def test_a_quote_that_does_not_move_changes_no_classification():
    """Exposure without consequence, which is the common and boring case."""
    events = [quote(BASE)]
    for i in range(1, 41):
        ts = BASE + i * MS
        events.append(quote(ts))
        events.append(trade(ts, "100.02"))
    r = classification_risk(events, granularity_ns=MS)
    assert r.same_tick == 40
    assert r.changed == 0


def test_an_undetermined_resolution_reports_nothing_rather_than_reassurance():
    events = [quote(BASE), trade(BASE, "100.02")]
    r = classification_risk(events)
    assert r.granularity_ns is None
    assert r.trades == 1
    assert r.same_tick == 0 and r.changed == 0


def test_the_granularity_is_detected_when_not_supplied():
    events = []
    for i in range(60):
        events.append(quote(BASE + i * MS))
        events.append(trade(BASE + i * MS, "100.02"))
    r = classification_risk(events)
    assert r.granularity_ns == MS
    assert r.same_tick == 60


def test_classified_counts_what_the_rule_could_answer_at_all():
    events = [trade(BASE + i * MS, "100.02") for i in range(30)]
    r = classification_risk(events, granularity_ns=MS)
    assert r.trades == 30
    assert r.classified == 0           # no quotes, nothing to classify against
    assert r.same_tick == 0            # and therefore no tie to be exposed to


def test_a_trade_at_the_mid_is_not_classified_either_way():
    events = [quote(BASE), trade(BASE + MS, "100.01")]
    r = classification_risk(events, granularity_ns=MS)
    assert r.classified == 0


def test_shares_are_exact_fractions_of_the_trade_count():
    events = []
    for i in range(50):
        events.append(quote(BASE + i * MS))
        events.append(trade(BASE + i * MS, "100.02"))
    r = classification_risk(events, granularity_ns=MS)
    assert r.exposed_share == D(r.same_tick) / 50
    assert r.changed_share == D(r.changed) / 50


def test_no_trades_reports_no_share_rather_than_zero():
    r = classification_risk([quote(BASE)], granularity_ns=MS)
    assert r.trades == 0
    assert r.exposed_share is None and r.changed_share is None


def test_classification_risk_refuses_a_non_positive_granularity():
    with pytest.raises(ValueError, match="positive"):
        classification_risk([], granularity_ns=0)


def test_a_coarser_stated_granularity_finds_more_exposure():
    """Stating the resolution you actually have changes the answer."""
    events = []
    for i in range(60):
        events.append(quote(BASE + i * MS))
        events.append(trade(BASE + i * MS + 400_000, "100.02"))
    fine = classification_risk(events, granularity_ns=US)
    coarse = classification_risk(events, granularity_ns=MS)
    assert fine.same_tick == 0
    assert coarse.same_tick == 60


# -- CSV -------------------------------------------------------------------

def test_read_timestamps_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("ts_ns,value\n1000,x\n2000,y\n", encoding="utf-8")
    assert read_timestamps_csv(str(p)) == [1000, 2000]


def test_a_row_that_does_not_parse_is_an_error_not_a_skip(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("ts_ns\n1000\nnope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        read_timestamps_csv(str(p))


def test_a_missing_column_names_itself(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("time\n1000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ts_ns"):
        read_timestamps_csv(str(p))


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("ts_ns\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no timestamps"):
        read_timestamps_csv(str(p))


def test_custom_column(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("t\n5\n", encoding="utf-8")
    assert read_timestamps_csv(str(p), ts_column="t") == [5]


# -- types -----------------------------------------------------------------

def test_frozen_dataclasses():
    for obj in (Resolution(0, 0, None, 0, 0, 0),
                ClassificationRisk(0, 0, 0, 0, None)):
        with pytest.raises(Exception):
            obj.trades = 1  # type: ignore[misc]
