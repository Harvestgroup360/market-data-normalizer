"""Point-in-time membership and cross-sectional tests."""
from decimal import Decimal

import pytest

from mdnorm import (
    AlignedRow,
    EventType,
    Listing,
    MarketEvent,
    Universe,
    align,
    cross_section,
    cross_sectional_rank,
    cross_sectional_zscore,
    mask_to_universe,
    read_listings_csv,
)

D = Decimal
DAY = 86_400 * 1_000_000_000


def row(**kwargs):
    values = {k: (None if v is None else D(str(v))) for k, v in kwargs.items()}
    return values


def trade(ts, price, symbol="X"):
    return MarketEvent(symbol=symbol, venue="v", event_type=EventType.TRADE,
                       ts_ns=ts, price=D(str(price)), size=D("1"))


# -- one listing -------------------------------------------------------------

def test_the_listing_interval_is_half_open():
    """Listed on the first day, not a member on the day it stops."""
    listing = Listing("AAA", listed_ns=10, delisted_ns=20)
    assert listing.active_at(9) is False
    assert listing.active_at(10) is True
    assert listing.active_at(19) is True
    assert listing.active_at(20) is False


def test_a_listing_with_no_delisting_runs_on():
    listing = Listing("AAA", listed_ns=10)
    assert listing.active_at(10**18) is True


@pytest.mark.parametrize("kwargs", [
    {"symbol": ""}, {"listed_ns": -1},
    {"listed_ns": 20, "delisted_ns": 20}, {"listed_ns": 20, "delisted_ns": 10},
])
def test_listings_are_validated(kwargs):
    args = {"symbol": "AAA", "listed_ns": 0, "delisted_ns": None}
    args.update(kwargs)
    with pytest.raises(ValueError):
        Listing(**args)


# -- the universe ------------------------------------------------------------

def test_membership_changes_over_time():
    u = Universe([Listing("AAA", 0),
                  Listing("BBB", 0, delisted_ns=10 * DAY),
                  Listing("CCC", 5 * DAY)])
    assert u.members_at(0) == ("AAA", "BBB")
    assert u.members_at(5 * DAY) == ("AAA", "BBB", "CCC")
    assert u.members_at(10 * DAY) == ("AAA", "CCC")


def test_members_come_back_in_a_stable_order():
    a = Universe([Listing("zeta", 0), Listing("alpha", 0)])
    b = Universe([Listing("alpha", 0), Listing("zeta", 0)])
    assert a.members_at(1) == b.members_at(1) == ("alpha", "zeta")


def test_a_symbol_can_list_delist_and_list_again():
    u = Universe([Listing("AAA", 0, delisted_ns=10),
                  Listing("AAA", 50)])
    assert u.contains("AAA", 5) and not u.contains("AAA", 20)
    assert u.contains("AAA", 60)
    assert u.symbols == ("AAA",) and len(u) == 1


def test_an_unknown_symbol_is_never_a_member():
    u = Universe([Listing("AAA", 0)])
    assert u.contains("ZZZ", 1) is False


def test_the_size_of_the_cross_section_is_queryable():
    u = Universe([Listing("AAA", 0), Listing("BBB", 10)])
    assert (u.size_at(0), u.size_at(10)) == (1, 2)


def test_an_empty_universe():
    u = Universe([])
    assert u.members_at(1) == () and len(u) == 0


# -- masking -----------------------------------------------------------------

def test_masking_blanks_names_that_were_not_trading_and_counts_them():
    rows = [AlignedRow(0, row(AAA=1, BBB=2), {"AAA": 0, "BBB": 0}),
            AlignedRow(50, row(AAA=1, BBB=2), {"AAA": 0, "BBB": 0})]
    u = Universe([Listing("AAA", 0), Listing("BBB", 0, delisted_ns=10)])
    masked, removed = mask_to_universe(rows, u)
    assert masked[0].values == {"AAA": D(1), "BBB": D(2)}
    assert masked[1].values == {"AAA": D(1), "BBB": None}
    assert removed == 1


def test_masking_does_not_count_cells_that_were_already_empty():
    rows = [AlignedRow(50, row(AAA=1, BBB=None), {"AAA": 0, "BBB": None})]
    u = Universe([Listing("AAA", 0), Listing("BBB", 0, delisted_ns=10)])
    _, removed = mask_to_universe(rows, u)
    assert removed == 0


def test_masking_leaves_the_original_rows_alone():
    rows = [AlignedRow(50, row(AAA=1), {"AAA": 0})]
    mask_to_universe(rows, Universe([]))
    assert rows[0].values == {"AAA": D(1)}


def test_a_masked_cell_loses_its_age_too():
    rows = [AlignedRow(50, row(AAA=1), {"AAA": 999})]
    masked, _ = mask_to_universe(rows, Universe([]))
    assert masked[0].ages_ns["AAA"] is None


# -- the bias this module exists for -----------------------------------------

def test_a_delisted_name_keeps_being_ranked_without_a_universe():
    """The survivorship demonstration, end to end.

    A forward-filled price does not stop when an instrument does, so a name
    that delisted on day 3 still carries a value on day 5 and still takes a
    place in the ranking. Point-in-time membership is what removes it.
    """
    rows = [AlignedRow(5 * DAY, row(ALIVE=10, OTHER=20, DEAD=30),
                       {"ALIVE": 0, "OTHER": 0, "DEAD": 2 * DAY})]
    naive = cross_section(rows, cross_sectional_rank)[0].values
    assert naive == {"ALIVE": D(1), "OTHER": D(2), "DEAD": D(3)}

    pit = Universe([Listing("ALIVE", 0), Listing("OTHER", 0),
                    Listing("DEAD", 0, delisted_ns=3 * DAY)])
    honest = cross_section(rows, cross_sectional_rank, universe=pit)[0].values
    assert honest == {"ALIVE": D(1), "OTHER": D(2), "DEAD": None}


def test_the_percentile_denominator_follows_the_universe():
    """Two instruments in the cross-section is a different scale from three."""
    rows = [AlignedRow(0, row(A=1, B=2, C=3), {"A": 0, "B": 0, "C": 0}),
            AlignedRow(50, row(A=1, B=2, C=3), {"A": 0, "B": 0, "C": 0})]
    u = Universe([Listing("A", 0), Listing("B", 0),
                  Listing("C", 0, delisted_ns=10)])
    out = cross_section(rows, lambda v: cross_sectional_rank(v, pct=True),
                        universe=u)
    assert out[0].values["B"] == D("0.5")      # middle of three
    assert out[1].values["B"] == D(1)          # top of two


def test_the_removed_count_is_the_signal_that_listings_are_real():
    """Zero removals across a long window means present-day membership."""
    rows = [AlignedRow(t * DAY, row(A=1, B=2), {"A": 0, "B": 0})
            for t in range(20)]
    survivors_only = Universe([Listing("A", 0), Listing("B", 0)])
    _, removed = mask_to_universe(rows, survivors_only)
    assert removed == 0

    real = Universe([Listing("A", 0), Listing("B", 0, delisted_ns=10 * DAY)])
    _, removed_real = mask_to_universe(rows, real)
    assert removed_real == 10


# -- ranking -----------------------------------------------------------------

def test_ranking_is_ascending_by_default():
    assert cross_sectional_rank(row(a=10, b=30, c=20)) == {
        "a": D(1), "b": D(3), "c": D(2)}


def test_ranking_can_be_reversed():
    assert cross_sectional_rank(row(a=10, b=30), ascending=False) == {
        "a": D(2), "b": D(1)}


def test_ties_share_the_average_rank():
    assert cross_sectional_rank(row(a=1, b=1, c=3)) == {
        "a": D("1.5"), "b": D("1.5"), "c": D(3)}


def test_a_three_way_tie():
    out = cross_sectional_rank(row(a=1, b=1, c=1))
    assert out == {"a": D(2), "b": D(2), "c": D(2)}


def test_missing_values_are_not_ranked_at_either_end():
    """Not last, not middle — a name that was not trading has no place."""
    out = cross_sectional_rank(row(a=10, b=None, c=20))
    assert out == {"a": D(1), "b": None, "c": D(2)}


def test_percentile_ranks_span_zero_to_one():
    out = cross_sectional_rank(row(a=1, b=2, c=3, d=4), pct=True)
    assert out["a"] == 0 and out["d"] == 1
    assert all(D(0) <= v <= D(1) for v in out.values())


def test_a_single_value_has_no_percentile():
    assert cross_sectional_rank(row(a=1), pct=True) == {"a": None}
    assert cross_sectional_rank(row(a=1)) == {"a": D(1)}


def test_ranking_an_empty_or_all_missing_row():
    assert cross_sectional_rank({}) == {}
    assert cross_sectional_rank(row(a=None, b=None)) == {"a": None, "b": None}


def test_every_input_key_comes_back():
    out = cross_sectional_rank(row(a=1, b=None, c=3))
    assert set(out) == {"a", "b", "c"}


# -- cross-sectional standardisation -----------------------------------------

def test_the_cross_section_is_standardised_against_itself():
    out = cross_sectional_zscore(row(a=10, b=20, c=30))
    assert out["b"] == 0
    assert out["a"] == -out["c"]


def test_a_flat_cross_section_has_no_zscore():
    assert cross_sectional_zscore(row(a=5, b=5, c=5)) == {
        "a": None, "b": None, "c": None}


def test_too_few_names_to_standardise():
    assert cross_sectional_zscore(row(a=1, b=2)) == {"a": None, "b": None}
    assert cross_sectional_zscore(row(a=1, b=2), ddof=0)["a"] is not None


def test_missing_names_stay_missing_when_standardising():
    out = cross_sectional_zscore(row(a=10, b=20, c=30, d=None))
    assert out["d"] is None


def test_a_negative_ddof_is_rejected():
    with pytest.raises(ValueError):
        cross_sectional_zscore(row(a=1, b=2, c=3), ddof=-1)


def test_standardising_uses_only_this_row():
    """Nothing from another timestamp may enter a cross-sectional statistic."""
    early = cross_sectional_zscore(row(a=1, b=2, c=3))
    same_row_later = cross_sectional_zscore(row(a=1, b=2, c=3))
    assert early == same_row_later


# -- applying it across a matrix ---------------------------------------------

def test_cross_section_maps_every_row():
    rows = [AlignedRow(0, row(a=1, b=2), {"a": 0, "b": 0}),
            AlignedRow(1, row(a=9, b=2), {"a": 0, "b": 0})]
    out = cross_section(rows, cross_sectional_rank)
    assert [r.ts_ns for r in out] == [0, 1]
    assert out[0].values["a"] == D(1) and out[1].values["a"] == D(2)


def test_cross_section_carries_the_ages_through():
    """A rank is exactly as old as the price it was computed from."""
    rows = [AlignedRow(0, row(a=1, b=2), {"a": 42, "b": 7})]
    out = cross_section(rows, cross_sectional_rank)
    assert out[0].ages_ns == {"a": 42, "b": 7}


def test_cross_section_of_no_rows():
    assert cross_section([], cross_sectional_rank) == []


# -- reading listings from a file --------------------------------------------

def test_listings_are_read_from_csv(tmp_path):
    p = tmp_path / "listings.csv"
    p.write_text("symbol,listed,delisted\n"
                 "AAA,2020-01-01T00:00:00Z,\n"
                 "BBB,2020-01-01T00:00:00Z,2021-06-01T00:00:00Z\n")
    listings = read_listings_csv(str(p))
    assert [x.symbol for x in listings] == ["AAA", "BBB"]
    assert listings[0].delisted_ns is None
    assert listings[1].delisted_ns > listings[1].listed_ns


def test_epoch_timestamps_in_listings(tmp_path):
    p = tmp_path / "listings.csv"
    p.write_text("symbol,listed,delisted\nAAA,1000,2000\n")
    listing = read_listings_csv(str(p), ts_unit="s")[0]
    assert listing.listed_ns == 1000 * 10**9


def test_blank_lines_in_listings_are_skipped(tmp_path):
    p = tmp_path / "listings.csv"
    p.write_text("symbol,listed,delisted\nAAA,2020-01-01T00:00:00Z,\n,,\n")
    assert len(read_listings_csv(str(p))) == 1


def test_a_bad_listing_row_names_its_line(tmp_path):
    p = tmp_path / "listings.csv"
    p.write_text("symbol,listed,delisted\nAAA,not-a-date,\n")
    with pytest.raises(ValueError, match=":2:"):
        read_listings_csv(str(p))


# -- integration -------------------------------------------------------------

def test_the_whole_path_from_ticks_to_a_point_in_time_cross_section():
    SEC = 1_000_000_000
    events_a = [trade(i * 60 * SEC, 100 + i, "A") for i in range(10)]
    events_b = [trade(i * 60 * SEC, 50 + i, "B") for i in range(10)]
    events_c = [trade(i * 60 * SEC, 10 + i, "C") for i in range(4)]   # stops
    rows = align({"A": events_a, "B": events_b, "C": events_c},
                 interval_ns=60 * SEC)

    # Without a universe, C's last price is carried forward and keeps ranking.
    naive = cross_section(rows, cross_sectional_rank)
    assert naive[-1].values["C"] is not None

    pit = Universe([Listing("A", 0), Listing("B", 0),
                    Listing("C", 0, delisted_ns=4 * 60 * SEC)])
    honest = cross_section(rows, cross_sectional_rank, universe=pit)
    assert honest[-1].values["C"] is None
    assert honest[0].values["C"] is not None
