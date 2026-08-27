"""Index membership tests: two dates, bounded guesses, and survivorship."""
import pytest

from mdnorm import (
    Basis,
    ChangeKind,
    IndexChange,
    IndexSnapshot,
    MembershipHistory,
    read_index_changes_csv,
    survivorship_gap,
)

DAY = 86_400 * 1_000_000_000


def d(n):
    return n * DAY


# -- IndexChange ------------------------------------------------------------


def test_announcement_may_not_follow_the_effective_date():
    with pytest.raises(ValueError):
        IndexChange("I1", ChangeKind.ADD, effective_ns=d(10), announced_ns=d(11))


def test_announcement_equal_to_effective_is_allowed():
    c = IndexChange("I1", ChangeKind.ADD, effective_ns=d(10), announced_ns=d(10))
    assert c.ts_on(Basis.ANNOUNCED) == d(10)


def test_empty_identifier_is_rejected():
    with pytest.raises(ValueError):
        IndexChange("", ChangeKind.ADD, effective_ns=d(1))


def test_missing_announcement_falls_back_to_effective():
    """A change with no announcement answers the announced question at its
    effective date rather than pretending to an earlier one."""
    c = IndexChange("I1", ChangeKind.ADD, effective_ns=d(10))
    assert c.ts_on(Basis.ANNOUNCED) == d(10)
    assert c.ts_on(Basis.EFFECTIVE) == d(10)


# -- the two bases ----------------------------------------------------------


def announced_history():
    """One name announced on day 5, joining on day 10."""
    return MembershipHistory.from_changes([
        IndexChange("OLD", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("NEW", ChangeKind.ADD, effective_ns=d(10), announced_ns=d(5)),
        IndexChange("OLD", ChangeKind.DELETE, effective_ns=d(10),
                    announced_ns=d(5)),
    ])


def test_effective_and_announced_disagree_between_the_dates():
    h = announced_history()
    assert h.members_at(d(7), basis=Basis.EFFECTIVE) == ("OLD",)
    assert h.members_at(d(7), basis=Basis.ANNOUNCED) == ("NEW",)


def test_the_two_bases_agree_once_the_change_has_taken_effect():
    h = announced_history()
    assert h.members_at(d(10), basis=Basis.EFFECTIVE) == ("NEW",)
    assert h.members_at(d(10), basis=Basis.ANNOUNCED) == ("NEW",)


def test_membership_starts_on_the_effective_day_itself():
    h = announced_history()
    assert h.members_at(d(9), basis=Basis.EFFECTIVE) == ("OLD",)
    assert "NEW" in h.members_at(d(10), basis=Basis.EFFECTIVE)


def test_is_member_matches_members_at():
    h = announced_history()
    for t in (d(0), d(7), d(10), d(20)):
        for iid in ("OLD", "NEW"):
            assert (h.is_member(iid, t, basis=Basis.EFFECTIVE)
                    == (iid in h.members_at(t, basis=Basis.EFFECTIVE)))


def test_nothing_is_a_member_before_the_record_begins():
    h = announced_history()
    assert h.members_at(0 - 1, basis=Basis.EFFECTIVE) == ()


# -- spells and re-entry ----------------------------------------------------


def test_re_entry_produces_two_separate_spells():
    h = MembershipHistory.from_changes([
        IndexChange("A", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("A", ChangeKind.DELETE, effective_ns=d(5)),
        IndexChange("A", ChangeKind.ADD, effective_ns=d(9)),
    ])
    assert h.intervals_of("A", basis=Basis.EFFECTIVE) == ((d(0), d(5)), (d(9), None))
    assert h.members_at(d(6), basis=Basis.EFFECTIVE) == ()
    assert h.members_at(d(9), basis=Basis.EFFECTIVE) == ("A",)


def test_universe_round_trip_reproduces_membership():
    h = MembershipHistory.from_changes([
        IndexChange("A", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("A", ChangeKind.DELETE, effective_ns=d(5)),
        IndexChange("B", ChangeKind.ADD, effective_ns=d(3)),
    ])
    u = h.to_universe(basis=Basis.EFFECTIVE)
    for t in (d(0), d(4), d(5), d(8)):
        assert tuple(sorted(u.members_at(t))) == h.members_at(t, basis=Basis.EFFECTIVE)


# -- snapshots --------------------------------------------------------------


def snapshot_history():
    return MembershipHistory.from_snapshots([
        IndexSnapshot(d(0), frozenset({"A", "B"})),
        IndexSnapshot(d(30), frozenset({"A", "C"})),
    ])


def test_snapshots_infer_both_directions():
    h = snapshot_history()
    kinds = {(i.instrument_id, i.kind) for i in h.inferred}
    assert kinds == {("B", ChangeKind.DELETE), ("C", ChangeKind.ADD)}


def test_snapshot_changes_are_dated_at_the_later_snapshot():
    """Conservative direction: never claim membership earlier than the file
    can support."""
    h = snapshot_history()
    assert h.members_at(d(29), basis=Basis.EFFECTIVE) == ("A", "B")
    assert h.members_at(d(30), basis=Basis.EFFECTIVE) == ("A", "C")


def test_snapshot_uncertainty_is_the_gap_between_files():
    h = snapshot_history()
    assert all(i.uncertainty_ns == d(30) for i in h.inferred)
    assert h.report().max_uncertainty_ns == d(30)


def test_first_snapshot_is_not_extended_backwards():
    h = snapshot_history()
    assert h.members_at(d(0) - 1, basis=Basis.EFFECTIVE) == ()
    # and the initial members are not counted as inferred, because nothing
    # was deduced about them
    assert all(i.instrument_id in {"B", "C"} for i in h.inferred)


def test_a_single_snapshot_yields_no_inference_at_all():
    h = MembershipHistory.from_snapshots([IndexSnapshot(d(5), frozenset({"A"}))])
    assert h.inferred == ()
    assert h.report().max_uncertainty_ns is None
    assert h.report().deletions == 0


# -- the report -------------------------------------------------------------


def test_report_counts_what_the_file_supports():
    h = announced_history()
    r = h.report()
    assert (r.instruments, r.changes, r.additions, r.deletions) == (2, 3, 2, 1)
    assert r.without_announcement == 1          # the day-zero OLD add
    assert r.never_removed == 1                 # NEW has no departure
    assert r.first_ns == d(0) and r.last_ns == d(10)


def test_announcement_coverage_is_a_fraction_or_none():
    assert announced_history().report().announcement_coverage == pytest.approx(2 / 3)
    empty = MembershipHistory.from_changes([])
    assert empty.report().announcement_coverage is None


def test_a_today_list_shows_up_as_every_name_never_removed():
    """The signature of a snapshot pretending to be a history."""
    h = MembershipHistory.from_changes([
        IndexChange(f"I{i}", ChangeKind.ADD, effective_ns=d(0)) for i in range(5)
    ])
    r = h.report()
    assert r.never_removed == r.instruments == 5
    assert r.deletions == 0


# -- survivorship -----------------------------------------------------------


def test_survivorship_gap_reports_both_directions():
    h = MembershipHistory.from_changes([
        IndexChange("STAYER", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("LEAVER", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("LEAVER", ChangeKind.DELETE, effective_ns=d(20)),
        IndexChange("JOINER", ChangeKind.ADD, effective_ns=d(20)),
    ])
    missed, phantom = survivorship_gap(h, d(5))
    assert missed == ("LEAVER",)     # dropped by a today-list study
    assert phantom == ("JOINER",)    # held before it joined


def test_survivorship_gap_is_empty_when_nothing_changed():
    h = MembershipHistory.from_changes([
        IndexChange("A", ChangeKind.ADD, effective_ns=d(0)),
    ])
    assert survivorship_gap(h, d(5)) == ((), ())


def test_survivorship_gap_accepts_an_explicit_today():
    h = MembershipHistory.from_changes([
        IndexChange("A", ChangeKind.ADD, effective_ns=d(0)),
        IndexChange("A", ChangeKind.DELETE, effective_ns=d(50)),
    ])
    assert survivorship_gap(h, d(5), today_ns=d(10)) == ((), ())
    assert survivorship_gap(h, d(5), today_ns=d(60)) == (("A",), ())


# -- CSV --------------------------------------------------------------------


def test_read_index_changes_csv(tmp_path):
    p = tmp_path / "changes.csv"
    p.write_text(
        "instrument_id,action,effective,announced\n"
        "A,add,2026-01-10T00:00:00Z,2026-01-05T00:00:00Z\n"
        f"B,delete,{d(400)},\n"
    )
    h = read_index_changes_csv(str(p), name="idx")
    assert h.name == "idx"
    assert len(h) == 2
    r = h.report()
    assert (r.additions, r.deletions, r.without_announcement) == (1, 1, 1)


def test_read_index_changes_csv_rejects_an_unknown_action(tmp_path):
    p = tmp_path / "changes.csv"
    p.write_text("instrument_id,action,effective\nA,included,0\n")
    with pytest.raises(ValueError) as exc:
        read_index_changes_csv(str(p))
    assert "unknown action" in str(exc.value)


def test_read_index_changes_csv_rejects_a_swapped_pair(tmp_path):
    p = tmp_path / "changes.csv"
    p.write_text(
        f"instrument_id,action,effective,announced\nA,add,{d(1)},{d(9)}\n")
    with pytest.raises(ValueError) as exc:
        read_index_changes_csv(str(p))
    assert "announced_ns follows effective_ns" in str(exc.value)


def test_repr_is_informative():
    assert "MembershipHistory" in repr(announced_history())
