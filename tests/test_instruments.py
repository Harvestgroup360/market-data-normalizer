"""Tests for mdnorm.instruments."""
import pytest

from mdnorm.instruments import (
    Segment,
    SymbolAssignment,
    SymbolMap,
    key_by_instrument,
    read_symbol_map_csv,
    series_segments,
)

DAY = 86_400_000_000_000
T0 = 1_700_000_000_000_000_000


def d(n):
    """A timestamp n days after T0."""
    return T0 + n * DAY


# A ticker used by one instrument, delisted, then reassigned to another.
REUSE = [
    SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(10)),
    SymbolAssignment("ABC", "INST-2", start_ns=d(20)),
]

# One instrument that changed its ticker.
RENAME = [
    SymbolAssignment("OLD", "INST-9", start_ns=d(0), end_ns=d(5)),
    SymbolAssignment("NEW", "INST-9", start_ns=d(5)),
]


# -- SymbolAssignment --------------------------------------------------------


def test_assignment_covers_its_own_interval():
    a = SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(10))
    assert a.covers(d(0)) is True
    assert a.covers(d(9)) is True


def test_the_end_is_exclusive():
    a = SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(10))
    assert a.covers(d(10)) is False


def test_before_the_start_is_not_covered():
    a = SymbolAssignment("ABC", "INST-1", start_ns=d(5))
    assert a.covers(d(4)) is False
    assert a.covers(d(5)) is True


def test_an_open_binding_covers_everything_after_it():
    a = SymbolAssignment("ABC", "INST-1", start_ns=d(0))
    assert a.open_ended is True
    assert a.covers(d(10_000)) is True


def test_assignment_rejects_empty_identifiers():
    with pytest.raises(ValueError):
        SymbolAssignment("", "INST-1", start_ns=0)
    with pytest.raises(ValueError):
        SymbolAssignment("ABC", "", start_ns=0)


def test_assignment_rejects_a_backwards_interval():
    with pytest.raises(ValueError):
        SymbolAssignment("ABC", "INST-1", start_ns=d(10), end_ns=d(5))
    with pytest.raises(ValueError):
        SymbolAssignment("ABC", "INST-1", start_ns=d(5), end_ns=d(5))


def test_assignment_is_comparable_by_value():
    a = SymbolAssignment("ABC", "INST-1", start_ns=0, end_ns=1)
    b = SymbolAssignment("ABC", "INST-1", start_ns=0, end_ns=1)
    assert a == b


# -- construction ------------------------------------------------------------


def test_map_size_and_contents():
    m = SymbolMap(REUSE)
    assert len(m) == 2
    assert m.symbols == ["ABC"]
    assert m.instruments == ["INST-1", "INST-2"]


def test_an_empty_map_resolves_nothing():
    m = SymbolMap([])
    assert len(m) == 0
    assert m.instrument_at("ABC", d(1)) is None
    assert m.reused_symbols() == []


def test_overlapping_assignments_are_rejected():
    with pytest.raises(ValueError) as exc:
        SymbolMap([
            SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(10)),
            SymbolAssignment("ABC", "INST-2", start_ns=d(5)),
        ])
    assert "two instruments at the same time" in str(exc.value)


def test_an_open_binding_followed_by_another_is_rejected():
    with pytest.raises(ValueError):
        SymbolMap([
            SymbolAssignment("ABC", "INST-1", start_ns=d(0)),
            SymbolAssignment("ABC", "INST-2", start_ns=d(20)),
        ])


def test_touching_intervals_are_fine():
    m = SymbolMap([
        SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(10)),
        SymbolAssignment("ABC", "INST-2", start_ns=d(10)),
    ])
    assert m.instrument_at("ABC", d(9)) == "INST-1"
    assert m.instrument_at("ABC", d(10)) == "INST-2"


def test_the_same_symbol_on_two_instruments_is_fine_when_disjoint():
    m = SymbolMap(REUSE)
    assert m.instrument_at("ABC", d(5)) == "INST-1"
    assert m.instrument_at("ABC", d(25)) == "INST-2"


def test_input_order_does_not_matter():
    a = SymbolMap(REUSE)
    b = SymbolMap(list(reversed(REUSE)))
    for t in (d(5), d(15), d(25)):
        assert a.instrument_at("ABC", t) == b.instrument_at("ABC", t)


# -- resolution --------------------------------------------------------------


def test_a_gap_between_bindings_resolves_to_nothing():
    """The ticker named nothing between the delisting and the reassignment."""
    m = SymbolMap(REUSE)
    assert m.instrument_at("ABC", d(15)) is None


def test_before_the_first_binding_resolves_to_nothing():
    m = SymbolMap(REUSE)
    assert m.instrument_at("ABC", d(-1)) is None


def test_an_unknown_symbol_resolves_to_nothing():
    assert SymbolMap(REUSE).instrument_at("ZZZ", d(1)) is None


def test_the_gap_is_not_filled_with_the_next_owner():
    """A value from the gap must not be attached to the instrument that
    happened to take the ticker afterwards. That is the splice."""
    m = SymbolMap(REUSE)
    assert m.instrument_at("ABC", d(19)) is None
    assert m.instrument_at("ABC", d(20)) == "INST-2"


def test_assignment_at_returns_the_binding_itself():
    a = SymbolMap(REUSE).assignment_at("ABC", d(5))
    assert a.instrument_id == "INST-1"
    assert a.end_ns == d(10)


def test_reverse_lookup_gives_the_ticker_of_the_day():
    m = SymbolMap(RENAME)
    assert m.symbol_at("INST-9", d(1)) == "OLD"
    assert m.symbol_at("INST-9", d(6)) == "NEW"


def test_reverse_lookup_outside_any_binding():
    assert SymbolMap(RENAME).symbol_at("INST-9", d(-1)) is None
    assert SymbolMap(RENAME).symbol_at("NOPE", d(1)) is None


def test_history_and_assignments_are_ordered():
    m = SymbolMap(RENAME)
    assert [a.symbol for a in m.history("INST-9")] == ["OLD", "NEW"]
    m2 = SymbolMap(REUSE)
    assert [a.instrument_id for a in m2.assignments_of("ABC")] == ["INST-1", "INST-2"]


def test_history_of_an_unknown_instrument_is_empty():
    assert SymbolMap(REUSE).history("NOPE") == []
    assert SymbolMap(REUSE).assignments_of("NOPE") == []


# -- diagnostics -------------------------------------------------------------


def test_reuse_is_reported():
    assert SymbolMap(REUSE).reused_symbols() == [("ABC", 2)]


def test_a_renamed_instrument_is_not_a_reused_symbol():
    m = SymbolMap(RENAME)
    assert m.reused_symbols() == []
    assert m.renamed_instruments() == [("INST-9", 2)]


def test_two_bindings_of_one_symbol_to_one_instrument_are_not_reuse():
    """A ticker that lapsed and came back on the same instrument is one
    instrument, not two."""
    m = SymbolMap([
        SymbolAssignment("ABC", "INST-1", start_ns=d(0), end_ns=d(5)),
        SymbolAssignment("ABC", "INST-1", start_ns=d(10)),
    ])
    assert m.reused_symbols() == []


def test_the_report_counts_everything():
    m = SymbolMap(REUSE + RENAME)
    r = m.report()
    assert r.assignments == 4
    assert r.symbols == 3            # ABC, OLD, NEW
    assert r.instruments == 3        # INST-1, INST-2, INST-9
    assert r.reused_symbols == 1
    assert r.renamed_instruments == 1
    assert r.open_ended == 2         # ABC -> INST-2, NEW -> INST-9


def test_a_present_day_only_file_reports_no_reuse():
    """The finding this module exists for: a file with one open binding per
    ticker cannot express reuse, so a zero here means the file, not the market."""
    m = SymbolMap([
        SymbolAssignment("AAA", "INST-1", start_ns=d(0)),
        SymbolAssignment("BBB", "INST-2", start_ns=d(0)),
    ])
    r = m.report()
    assert r.reused_symbols == 0
    assert r.open_ended == r.assignments


# -- key_by_instrument -------------------------------------------------------


ROWS = [
    {"ts_ns": d(1), "symbol": "ABC", "px": "10"},
    {"ts_ns": d(5), "symbol": "ABC", "px": "11"},
    {"ts_ns": d(15), "symbol": "ABC", "px": "12"},   # in the gap
    {"ts_ns": d(25), "symbol": "ABC", "px": "80"},   # the other instrument
]


def test_rows_are_keyed_by_the_instrument_of_their_own_timestamp():
    out, counts = key_by_instrument(ROWS, SymbolMap(REUSE))
    assert [r["instrument_id"] for r in out] == ["INST-1", "INST-1", "INST-2"]
    assert counts == {"mapped": 3, "unmapped": 1, "reassigned": 2}


def test_the_reassigned_count_is_the_point():
    """Rows whose ticker pointed somewhere else than it points today."""
    _, counts = key_by_instrument(ROWS, SymbolMap(REUSE))
    assert counts["reassigned"] == 2


def test_nothing_is_reassigned_when_a_ticker_never_moved():
    rows = [{"ts_ns": d(1), "symbol": "OLD"}, {"ts_ns": d(6), "symbol": "NEW"}]
    _, counts = key_by_instrument(rows, SymbolMap(RENAME))
    assert counts["reassigned"] == 0
    assert counts["mapped"] == 2


def test_unmapped_rows_can_be_kept():
    out, counts = key_by_instrument(ROWS, SymbolMap(REUSE), drop_unmapped=False)
    assert len(out) == 4
    assert "instrument_id" not in out[2]
    assert counts["unmapped"] == 1


def test_the_original_rows_are_not_modified():
    before = [dict(r) for r in ROWS]
    key_by_instrument(ROWS, SymbolMap(REUSE))
    assert ROWS == before


def test_other_fields_survive():
    out, _ = key_by_instrument(ROWS, SymbolMap(REUSE))
    assert out[0]["px"] == "10"
    assert out[0]["symbol"] == "ABC"


def test_custom_field_names():
    rows = [{"t": d(1), "ticker": "ABC"}]
    out, counts = key_by_instrument(rows, SymbolMap(REUSE), symbol_field="ticker",
                                    ts_field="t", target_field="iid")
    assert out[0]["iid"] == "INST-1"
    assert counts["mapped"] == 1


def test_a_malformed_row_counts_as_unmapped():
    rows = [{"ts_ns": "not a number", "symbol": "ABC"}, {"symbol": "ABC"},
            {"ts_ns": d(1)}]
    out, counts = key_by_instrument(rows, SymbolMap(REUSE))
    assert out == []
    assert counts["unmapped"] == 3


def test_no_rows_at_all():
    out, counts = key_by_instrument([], SymbolMap(REUSE))
    assert out == []
    assert counts == {"mapped": 0, "unmapped": 0, "reassigned": 0}


# -- series_segments ---------------------------------------------------------


def test_a_clean_series_is_one_segment():
    ts = [d(1), d(2), d(3)]
    segs, unresolved = series_segments("ABC", ts, SymbolMap(REUSE))
    assert len(segs) == 1
    assert unresolved == 0
    assert segs[0].instrument_id == "INST-1"
    assert (segs[0].start_index, segs[0].stop_index) == (0, 3)


def test_reuse_splits_the_series():
    ts = [d(1), d(5), d(25), d(26)]
    segs, unresolved = series_segments("ABC", ts, SymbolMap(REUSE))
    assert [s.instrument_id for s in segs] == ["INST-1", "INST-2"]
    assert [(s.start_index, s.stop_index) for s in segs] == [(0, 2), (2, 4)]
    assert unresolved == 0


def test_a_gap_closes_the_segment_and_is_counted():
    ts = [d(1), d(15), d(25)]
    segs, unresolved = series_segments("ABC", ts, SymbolMap(REUSE))
    assert [s.instrument_id for s in segs] == ["INST-1", "INST-2"]
    assert unresolved == 1
    assert segs[0].stop_index == 1


def test_segment_timestamps_bound_the_stretch():
    ts = [d(1), d(5), d(25)]
    segs, _ = series_segments("ABC", ts, SymbolMap(REUSE))
    assert segs[0].start_ns == d(1) and segs[0].end_ns == d(5)
    assert segs[1].start_ns == d(25) and segs[1].end_ns == d(25)


def test_segment_length_is_its_row_count():
    ts = [d(1), d(2), d(3), d(25)]
    segs, _ = series_segments("ABC", ts, SymbolMap(REUSE))
    assert len(segs[0]) == 3
    assert len(segs[1]) == 1


def test_a_rename_does_not_split_anything():
    """The instrument is the same throughout; only its label changed."""
    ts = [d(1), d(6)]
    a, _ = series_segments("OLD", ts, SymbolMap(RENAME))
    assert [s.instrument_id for s in a] == ["INST-9"]
    assert a[0].stop_index == 1      # OLD only names it for the first row


def test_a_wholly_unresolved_series_yields_no_segments():
    segs, unresolved = series_segments("ZZZ", [d(1), d(2)], SymbolMap(REUSE))
    assert segs == []
    assert unresolved == 2


def test_an_empty_series():
    segs, unresolved = series_segments("ABC", [], SymbolMap(REUSE))
    assert segs == [] and unresolved == 0


def test_out_of_order_timestamps_are_rejected():
    with pytest.raises(ValueError):
        series_segments("ABC", [d(5), d(1)], SymbolMap(REUSE))


def test_repeated_timestamps_are_allowed():
    segs, _ = series_segments("ABC", [d(1), d(1), d(2)], SymbolMap(REUSE))
    assert len(segs) == 1 and len(segs[0]) == 3


def test_segment_is_comparable_by_value():
    a = Segment("ABC", "INST-1", 0, 2, d(0), d(1))
    b = Segment("ABC", "INST-1", 0, 2, d(0), d(1))
    assert a == b


# -- causality ---------------------------------------------------------------


def test_segments_before_a_change_are_unaffected_by_later_timestamps():
    """The standing property: nothing before index i may depend on anything
    after it."""
    base = [d(1), d(2), d(3), d(4)]
    tampered = base[:2] + [d(25), d(26)]
    a, _ = series_segments("ABC", base, SymbolMap(REUSE))
    b, _ = series_segments("ABC", tampered, SymbolMap(REUSE))
    assert a[0].start_index == b[0].start_index
    assert b[0].stop_index == 2
    assert b[0].instrument_id == a[0].instrument_id


def test_resolution_of_one_row_does_not_depend_on_the_others():
    m = SymbolMap(REUSE)
    assert m.instrument_at("ABC", d(5)) == "INST-1"
    _ = key_by_instrument(ROWS, m)
    assert m.instrument_at("ABC", d(5)) == "INST-1"


# -- reading a file ----------------------------------------------------------


def test_read_a_symbol_map(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(
        "symbol,instrument_id,start_ns,end_ns,venue\n"
        f"ABC,INST-1,{d(0)},{d(10)},xnys\n"
        f"ABC,INST-2,{d(20)},,xnys\n"
    )
    rows = read_symbol_map_csv(str(p))
    assert len(rows) == 2
    assert rows[0].venue == "xnys"
    assert rows[1].end_ns is None
    m = SymbolMap(rows)
    assert m.reused_symbols() == [("ABC", 2)]


def test_reading_tolerates_a_missing_venue_column(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(f"symbol,instrument_id,start_ns\nABC,INST-1,{d(0)}\n")
    rows = read_symbol_map_csv(str(p))
    assert rows[0].venue is None
    assert rows[0].end_ns is None


def test_a_row_missing_an_identifier_is_an_error(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(f"symbol,instrument_id,start_ns\nABC,,{d(0)}\n")
    with pytest.raises(ValueError) as exc:
        read_symbol_map_csv(str(p))
    assert "instrument_id" in str(exc.value)


def test_a_row_missing_a_start_is_an_error(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text("symbol,instrument_id,start_ns\nABC,INST-1,\n")
    with pytest.raises(ValueError):
        read_symbol_map_csv(str(p))


def test_the_error_names_the_line(tmp_path):
    p = tmp_path / "map.csv"
    p.write_text(f"symbol,instrument_id,start_ns\nABC,INST-1,{d(0)}\nBBB,,{d(0)}\n")
    with pytest.raises(ValueError) as exc:
        read_symbol_map_csv(str(p))
    assert ":3:" in str(exc.value)
