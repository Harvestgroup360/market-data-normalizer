"""Provenance tests: what a run read, what it was told, and whether it holds."""
import json
from decimal import Decimal

import pytest

from mdnorm.provenance import (
    Drift,
    DriftKind,
    Digest,
    InputRef,
    Manifest,
    digest_bytes,
    digest_file,
    manifest,
    read_manifest,
    verify,
    write_manifest,
)

D = Decimal


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# -- digests ---------------------------------------------------------------

def test_the_digest_is_of_the_bytes():
    got = digest_bytes(b"a,b\n1,2\n")
    assert got.algorithm == "sha256"
    assert got.bytes == 8
    assert len(got.hexdigest) == 64


def test_the_same_bytes_hash_the_same_way():
    assert digest_bytes(b"x").hexdigest == digest_bytes(b"x").hexdigest


def test_line_endings_are_a_difference_because_they_parse_differently():
    assert digest_bytes(b"a\n").hexdigest != digest_bytes(b"a\r\n").hexdigest


def test_a_file_digest_matches_its_contents(tmp_path):
    p = write(tmp_path, "in.csv", "a,b\n1,2\n")
    assert digest_file(p).hexdigest == digest_bytes(b"a,b\n1,2\n").hexdigest


def test_an_empty_file_still_has_a_digest(tmp_path):
    p = write(tmp_path, "empty.csv", "")
    d = digest_file(p)
    assert d.bytes == 0 and len(d.hexdigest) == 64


def test_the_short_form_is_for_printing():
    assert len(digest_bytes(b"x").short) == 12


def test_a_digest_covers_a_non_negative_number_of_bytes():
    with pytest.raises(ValueError, match="non-negative"):
        Digest("sha256", "ab", -1)


# -- parameters ------------------------------------------------------------

def test_a_float_parameter_is_refused(tmp_path):
    with pytest.raises(TypeError, match="rendering"):
        manifest(command="x", parameters={"threshold": 0.1})


def test_a_decimal_parameter_is_kept_exactly(tmp_path):
    m = manifest(command="x", parameters={"threshold": D("0.10")})
    assert m.parameter_map["threshold"] == "0.10"


def test_ints_bools_and_strings_are_accepted():
    m = manifest(command="x", parameters={"trials": 500, "purged": True,
                                          "tz": "UTC", "lag": None})
    assert m.parameter_map == {"trials": "500", "purged": "true",
                               "tz": "UTC", "lag": ""}


def test_an_unsupported_parameter_type_is_refused():
    with pytest.raises(TypeError, match="must be str, int, bool"):
        manifest(command="x", parameters={"window": [1, 2, 3]})


def test_a_parameter_cannot_be_listed_twice():
    with pytest.raises(ValueError, match="listed twice"):
        Manifest("x", "lib", "1", "3.12", 0,
                 parameters=(("k", "1"), ("k", "2")))


# -- the fingerprint -------------------------------------------------------

def test_two_runs_of_the_same_thing_fingerprint_identically(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    a = manifest(command="sharpe", inputs=[p], parameters={"trials": 500})
    b = manifest(command="sharpe", inputs=[p], parameters={"trials": 500})
    assert a.created_ns != b.created_ns or True      # clock may not have moved
    assert a.fingerprint == b.fingerprint


def test_the_clock_is_not_part_of_the_fingerprint(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    a = manifest(command="x", inputs=[p])
    b = Manifest(a.command, a.library, a.version, a.python,
                 created_ns=a.created_ns + 10 ** 12, inputs=a.inputs,
                 outputs=a.outputs, parameters=a.parameters)
    assert a.fingerprint == b.fingerprint


def test_the_note_is_not_part_of_the_fingerprint(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    a = manifest(command="x", inputs=[p], note="first attempt")
    b = manifest(command="x", inputs=[p], note="after the meeting")
    assert a.fingerprint == b.fingerprint


def test_a_changed_parameter_changes_the_fingerprint(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    a = manifest(command="x", inputs=[p], parameters={"trials": 500})
    b = manifest(command="x", inputs=[p], parameters={"trials": 50})
    assert a.fingerprint != b.fingerprint


def test_a_changed_input_changes_the_fingerprint(tmp_path):
    a = manifest(command="x", inputs=[write(tmp_path, "one.csv", "a\n1\n")])
    b = manifest(command="x", inputs=[write(tmp_path, "two.csv", "a\n2\n")])
    assert a.fingerprint != b.fingerprint


def test_parameter_order_does_not_change_the_fingerprint(tmp_path):
    a = manifest(command="x", parameters={"a": 1, "b": 2})
    b = manifest(command="x", parameters={"b": 2, "a": 1})
    assert a.fingerprint == b.fingerprint


def test_outputs_are_not_part_of_the_fingerprint(tmp_path):
    """Same inputs, different results is the finding, not a contradiction."""
    p = write(tmp_path, "in.csv", "a\n1\n")
    o1 = write(tmp_path, "out1.csv", "r\n1\n")
    o2 = write(tmp_path, "out2.csv", "r\n999\n")
    a = manifest(command="x", inputs=[p], outputs=[o1])
    b = manifest(command="x", inputs=[p], outputs=[o2])
    assert a.fingerprint == b.fingerprint


def test_the_canonical_form_is_stable_text(tmp_path):
    m = manifest(command="x", parameters={"b": 2, "a": 1})
    body = json.loads(m.canonical())
    assert body["parameters"] == [["a", "1"], ["b", "2"]]
    assert "created_ns" not in body and "note" not in body


# -- verification ----------------------------------------------------------

def test_an_unchanged_run_is_reproducible(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    m = manifest(command="x", inputs=[p], parameters={"trials": 500})
    v = verify(m, parameters={"trials": 500})
    assert v.reproducible and v.drifts == ()


def test_a_changed_input_is_reported(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    m = manifest(command="x", inputs=[p])
    (tmp_path / "in.csv").write_text("a\n2\n")
    v = verify(m)
    assert not v.reproducible
    [c] = v.of_kind(DriftKind.DIGEST)
    assert c.name == p and c.was != c.now


def test_a_missing_input_is_reported_separately(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    m = manifest(command="x", inputs=[p])
    (tmp_path / "in.csv").unlink()
    [c] = verify(m).of_kind(DriftKind.MISSING)
    assert c.name == p and c.now is None


def test_a_changed_parameter_is_reported(tmp_path):
    """The case nobody notices: no file was edited."""
    p = write(tmp_path, "in.csv", "a\n1\n")
    m = manifest(command="x", inputs=[p], parameters={"trials": 500})
    v = verify(m, parameters={"trials": 50})
    assert not v.reproducible
    [c] = v.of_kind(DriftKind.PARAMETER)
    assert (c.name, c.was, c.now) == ("trials", "500", "50")


def test_a_parameter_that_was_not_recorded_is_flagged_as_added(tmp_path):
    m = manifest(command="x", parameters={"trials": 500})
    [c] = verify(m, parameters={"trials": 500, "embargo": 10}).of_kind(
        DriftKind.ADDED)
    assert c.name == "embargo" and c.was is None


def test_a_dropped_parameter_is_reported_too():
    m = manifest(command="x", parameters={"trials": 500})
    [c] = verify(m, parameters={}).of_kind(DriftKind.PARAMETER)
    assert (c.name, c.was, c.now) == ("trials", "500", None)


def test_parameters_are_only_checked_when_given(tmp_path):
    """Leaving them out asks the weaker question, and asks it honestly."""
    p = write(tmp_path, "in.csv", "a\n1\n")
    m = manifest(command="x", inputs=[p], parameters={"trials": 500})
    assert verify(m).reproducible


def test_a_different_library_version_is_reported(tmp_path):
    m = manifest(command="x")
    stale = Manifest(m.command, m.library, "0.0.1", m.python, m.created_ns)
    [c] = verify(stale).of_kind(DriftKind.VERSION)
    assert c.was == "0.0.1"
    assert verify(stale, check_version=False).reproducible


def test_every_difference_is_reported_not_just_the_first(tmp_path):
    a = write(tmp_path, "a.csv", "1\n")
    b = write(tmp_path, "b.csv", "2\n")
    m = manifest(command="x", inputs=[a, b], parameters={"trials": 500})
    (tmp_path / "a.csv").write_text("9\n")
    (tmp_path / "b.csv").unlink()
    v = verify(m, parameters={"trials": 1})
    assert len(v.drifts) == 3
    assert {c.kind for c in v.drifts} == {
        DriftKind.DIGEST, DriftKind.MISSING, DriftKind.PARAMETER}


def test_a_change_prints_what_moved():
    c = Drift(DriftKind.PARAMETER, "trials", "500", "50")
    assert "trials" in str(c) and "500" in str(c) and "50" in str(c)


# -- round trip ------------------------------------------------------------

def test_a_manifest_survives_a_round_trip(tmp_path):
    p = write(tmp_path, "in.csv", "a\n1\n")
    out = str(tmp_path / "run.json")
    m = manifest(command="sharpe", inputs=[p], parameters={"trials": 500},
                 rows={p: 2}, note="quarterly review")
    write_manifest(m, out)
    back = read_manifest(out)
    assert back.fingerprint == m.fingerprint
    assert back.command == "sharpe"
    assert back.parameter_map == {"trials": "500"}
    assert back.inputs[0].rows == 2
    assert back.note == "quarterly review"


def test_the_written_file_is_json_a_human_can_read(tmp_path):
    out = str(tmp_path / "run.json")
    write_manifest(manifest(command="x"), out)
    text = (tmp_path / "run.json").read_text()
    assert text.endswith("\n")
    assert json.loads(text)["command"] == "x"


def test_an_edited_manifest_is_refused(tmp_path):
    out = str(tmp_path / "run.json")
    write_manifest(manifest(command="x", parameters={"trials": 500}), out)
    raw = json.loads((tmp_path / "run.json").read_text())
    raw["parameters"]["trials"] = "50"
    (tmp_path / "run.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="edited since it was written"):
        read_manifest(out)


def test_a_manifest_from_a_different_schema_is_refused(tmp_path):
    out = tmp_path / "run.json"
    out.write_text(json.dumps({"schema": 99, "command": "x", "library": "l",
                               "version": "1"}))
    with pytest.raises(ValueError, match="schema"):
        read_manifest(str(out))


def test_a_manifest_with_no_fingerprint_field_still_reads(tmp_path):
    """An older writer that recorded no fingerprint is not an edited file."""
    out = tmp_path / "run.json"
    body = manifest(command="x").to_dict()
    body.pop("fingerprint")
    out.write_text(json.dumps(body))
    assert read_manifest(str(out)).command == "x"


# -- types -----------------------------------------------------------------

def test_an_input_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        InputRef("", digest_bytes(b""))


def test_rows_cannot_be_negative():
    with pytest.raises(ValueError, match="non-negative"):
        InputRef("a", digest_bytes(b""), rows=-1)


def test_an_input_cannot_be_listed_twice():
    ref = InputRef("a.csv", digest_bytes(b""))
    with pytest.raises(ValueError, match="listed twice"):
        Manifest("x", "lib", "1", "3.12", 0, inputs=(ref, ref))


def test_frozen_dataclasses():
    for obj in (digest_bytes(b""), InputRef("a", digest_bytes(b"")),
                Drift(DriftKind.DIGEST, "a", None, None),
                Manifest("x", "l", "1", "3.12", 0)):
        with pytest.raises(Exception):
            obj.name = "z"  # type: ignore[misc]
