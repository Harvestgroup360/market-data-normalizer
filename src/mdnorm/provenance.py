"""A result you cannot reproduce is not a result, it is an anecdote.

Every other module in this library refuses to guess a constant. ``metrics``
will not invent the number of trials a search ran; ``coverage`` will not pick
a gap threshold; ``halts`` will not infer a pause; ``independence`` will not
choose a truncation lag. Each of those refusals hands the caller a decision —
and nothing, anywhere, wrote down what they decided::

    from mdnorm import manifest, verify, write_manifest

    m = manifest(command="sharpe", inputs=["pnl.csv"],
                 parameters={"trials": 500, "risk_free": "0.04"})
    write_manifest(m, "run.json")

    verify(read_manifest("run.json")).reproducible   # False
    verify(read_manifest("run.json")).drifts          # pnl.csv: digest

**The parameters are the part that goes missing.** An input file that changes
is usually noticed, because somebody had to change it. A trial count that was
500 in the run and 50 in the write-up is noticed by nobody, and it is the
difference between a deflated Sharpe that survives and one that does not. A
manifest records the arguments beside the data, because they are the same
kind of fact.

**A fingerprint that includes the clock is useless.** Two runs of the same
pipeline over the same inputs with the same arguments must produce the same
fingerprint, or the fingerprint answers no question worth asking.
:attr:`Manifest.fingerprint` therefore excludes ``created_ns`` and the
free-text note, and covers exactly what would change the numbers.

**Floats are refused.** A parameter of ``0.1`` is not a value, it is a
rendering of one, and the rendering differs by platform and by Python
version. Pass a string or a :class:`~decimal.Decimal` and the manifest
records what you meant; pass a float and it raises rather than fingerprinting
something it cannot promise to reproduce.

**The digest is of the bytes, not of the meaning.** A CSV re-exported with
different line endings is a different file, and this says so. That is
deliberate: the parse is what changed, and quietly forgiving it is how a
pipeline comes to depend on a detail nobody chose.

**Nothing here judges.** :func:`verify` reports what moved and stops. Whether
a changed input is a correction or a corruption is not a question a library
can answer, and a tool that decided would be trusted for a judgment it is not
entitled to make.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Digest",
    "InputRef",
    "Manifest",
    "DriftKind",
    "Drift",
    "Verification",
    "digest_bytes",
    "digest_file",
    "manifest",
    "verify",
    "read_manifest",
    "write_manifest",
]

_ALGORITHM = "sha256"
_CHUNK = 1 << 20
_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class Digest:
    """A content hash of one file, with the size that produced it."""

    algorithm: str
    hexdigest: str
    bytes: int

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("a digest covers a non-negative number of bytes")

    @property
    def short(self) -> str:
        """The first twelve characters, for printing beside a filename."""
        return self.hexdigest[:12]


@dataclass(frozen=True, slots=True)
class InputRef:
    """One file a run read or wrote, as it was at the time.

    ``rows`` is optional and never computed here: counting rows means parsing,
    parsing means a schema, and a manifest that needed to understand its
    inputs could not record an input it did not understand.
    """

    name: str
    digest: Digest
    rows: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an input needs a name")
        if self.rows is not None and self.rows < 0:
            raise ValueError("rows must be non-negative")


class DriftKind(str, Enum):
    """What moved between the manifest and the world.

    Named ``Drift`` rather than ``Change`` because :mod:`mdnorm.membership`
    already exports a ``ChangeKind`` for index additions and deletions, and
    two things called the same name in one namespace is a bug waiting for a
    reader in a hurry.
    """

    MISSING = "missing"
    DIGEST = "digest"
    ROWS = "rows"
    ADDED = "added"
    PARAMETER = "parameter"
    VERSION = "version"


@dataclass(frozen=True, slots=True)
class Drift:
    """One difference, stated as what it was and what it is now."""

    kind: DriftKind
    name: str
    was: Optional[str]
    now: Optional[str]

    def __str__(self) -> str:
        return f"{self.name}: {self.kind.value} {self.was!r} -> {self.now!r}"


def _canonical(value: Any) -> str:
    """One unambiguous string for a parameter value, or a refusal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        raise TypeError(
            f"a float parameter cannot be recorded reproducibly: {value!r} is "
            "a rendering of a value rather than the value, and the rendering "
            "differs between platforms. Pass a str or a Decimal.")
    if isinstance(value, (int, str, Decimal)):
        return str(value)
    if value is None:
        return ""
    raise TypeError(
        f"parameter values must be str, int, bool, Decimal or None, not "
        f"{type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Manifest:
    """What a run read, what it was told, and what produced it."""

    command: str
    library: str
    version: str
    python: str
    created_ns: int
    inputs: Tuple[InputRef, ...] = ()
    outputs: Tuple[InputRef, ...] = ()
    parameters: Tuple[Tuple[str, str], ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        names = [i.name for i in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("an input is listed twice")
        keys = [k for k, _ in self.parameters]
        if len(set(keys)) != len(keys):
            raise ValueError("a parameter is listed twice")

    @property
    def parameter_map(self) -> Dict[str, str]:
        return dict(self.parameters)

    def canonical(self) -> str:
        """The exact text the fingerprint is taken over.

        Sorted, minimally separated JSON with no timestamp and no note, so
        that two runs which would produce the same numbers produce the same
        bytes here.
        """
        body = {
            "schema": _SCHEMA,
            "command": self.command,
            "library": self.library,
            "version": self.version,
            "inputs": [[i.name, i.digest.algorithm, i.digest.hexdigest,
                        i.digest.bytes, i.rows] for i in
                       sorted(self.inputs, key=lambda x: x.name)],
            "parameters": [[k, v] for k, v in sorted(self.parameters)],
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    @property
    def fingerprint(self) -> str:
        """A hash of :meth:`canonical` — the run's identity.

        Outputs are deliberately not part of it. The question a fingerprint
        answers is "was this the same run", and a run is defined by what went
        in and what it was told, not by what came out. Two fingerprints that
        match and outputs that differ is the finding, not a contradiction.
        """
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        def ref(i: InputRef) -> Dict[str, Any]:
            return {"name": i.name, "algorithm": i.digest.algorithm,
                    "digest": i.digest.hexdigest, "bytes": i.digest.bytes,
                    "rows": i.rows}

        return {
            "schema": _SCHEMA,
            "command": self.command,
            "library": self.library,
            "version": self.version,
            "python": self.python,
            "created_ns": self.created_ns,
            "fingerprint": self.fingerprint,
            "inputs": [ref(i) for i in self.inputs],
            "outputs": [ref(o) for o in self.outputs],
            "parameters": dict(self.parameters),
            "note": self.note,
        }


def digest_bytes(data: bytes) -> Digest:
    """Hash a block of bytes already in memory."""
    return Digest(_ALGORITHM, hashlib.sha256(data).hexdigest(), len(data))


def digest_file(path: str) -> Digest:
    """Hash a file, streaming, so a multi-gigabyte input costs no memory.

    Compressed inputs are hashed as they sit on disk. Two files whose
    contents decompress to the same bytes have different digests, which is
    correct: the run read one of them.
    """
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
            total += len(block)
    return Digest(_ALGORITHM, h.hexdigest(), total)


def _refs(paths: Iterable[str], rows: Optional[Mapping[str, int]]) -> Tuple[InputRef, ...]:
    counts = dict(rows or {})
    return tuple(InputRef(name=p, digest=digest_file(p), rows=counts.get(p))
                 for p in paths)


def manifest(
    *,
    command: str,
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    parameters: Optional[Mapping[str, Any]] = None,
    rows: Optional[Mapping[str, int]] = None,
    note: str = "",
) -> Manifest:
    """Record a run: what it read, what it wrote, what it was told.

    Parameter values must be ``str``, ``int``, ``bool``, ``Decimal`` or
    ``None``. A float raises, because the manifest cannot promise to
    reproduce a number it can only render.
    """
    from . import __version__

    items = tuple(sorted((k, _canonical(v))
                         for k, v in dict(parameters or {}).items()))
    return Manifest(
        command=command,
        library="market-data-normalizer",
        version=__version__,
        python=f"{sys.version_info.major}.{sys.version_info.minor}."
               f"{sys.version_info.micro}",
        created_ns=time.time_ns(),
        inputs=_refs(inputs, rows),
        outputs=_refs(outputs, rows),
        parameters=items,
        note=note,
    )


@dataclass(frozen=True, slots=True)
class Verification:
    """What a manifest expected, against what is there now."""

    manifest: Manifest
    drifts: Tuple[Drift, ...]

    @property
    def reproducible(self) -> bool:
        """Whether nothing that would change the numbers has moved."""
        return not self.drifts

    def of_kind(self, kind: DriftKind) -> Tuple[Drift, ...]:
        return tuple(c for c in self.drifts if c.kind is kind)


def verify(
    recorded: Manifest,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    check_version: bool = True,
) -> Verification:
    """Re-read the inputs and report every difference, in order.

    ``parameters`` is what the run is being repeated with. Leave it out and
    only the files are checked, which is the weaker question: the arguments
    are where reproducibility usually fails, because nobody had to edit a
    file to change them.

    Outputs are not checked. A manifest exists so that a *different* output
    from the same inputs is visible as a finding, and verifying the outputs
    against themselves would hide exactly that.
    """
    from . import __version__

    drifts: List[Drift] = []

    if check_version and recorded.version != __version__:
        drifts.append(Drift(DriftKind.VERSION, "library",
                              recorded.version, __version__))

    for ref in sorted(recorded.inputs, key=lambda x: x.name):
        try:
            now = digest_file(ref.name)
        except OSError:
            drifts.append(Drift(DriftKind.MISSING, ref.name,
                                  ref.digest.short, None))
            continue
        if now.hexdigest != ref.digest.hexdigest:
            drifts.append(Drift(DriftKind.DIGEST, ref.name,
                                  ref.digest.short, now.short))

    if parameters is not None:
        given = {k: _canonical(v) for k, v in dict(parameters).items()}
        recorded_params = recorded.parameter_map
        for key in sorted(set(given) | set(recorded_params)):
            was = recorded_params.get(key)
            now_value = given.get(key)
            if was == now_value:
                continue
            kind = (DriftKind.ADDED if was is None else DriftKind.PARAMETER)
            drifts.append(Drift(kind, key, was, now_value))

    return Verification(manifest=recorded, drifts=tuple(drifts))


def write_manifest(m: Manifest, path: str) -> None:
    """Write a manifest as indented JSON, sorted, with a trailing newline."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(m.to_dict(), fh, indent=2, sort_keys=True,
                  ensure_ascii=False)
        fh.write("\n")


def read_manifest(path: str) -> Manifest:
    """Read a manifest and check its fingerprint still matches its contents.

    A manifest edited by hand is worse than no manifest, because it carries
    the authority of a record while stating something that never happened.
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    schema = raw.get("schema")
    if schema != _SCHEMA:
        raise ValueError(
            f"manifest schema {schema!r} is not {_SCHEMA}; this file was "
            "written by a different version of the library")

    def ref(d: Mapping[str, Any]) -> InputRef:
        return InputRef(
            name=d["name"],
            digest=Digest(d.get("algorithm", _ALGORITHM), d["digest"],
                          int(d["bytes"])),
            rows=None if d.get("rows") is None else int(d["rows"]))

    m = Manifest(
        command=raw["command"],
        library=raw["library"],
        version=raw["version"],
        python=raw.get("python", ""),
        created_ns=int(raw.get("created_ns", 0)),
        inputs=tuple(ref(d) for d in raw.get("inputs", ())),
        outputs=tuple(ref(d) for d in raw.get("outputs", ())),
        parameters=tuple(sorted((k, str(v)) for k, v
                                in dict(raw.get("parameters", {})).items())),
        note=raw.get("note", ""),
    )
    stated = raw.get("fingerprint")
    if stated is not None and stated != m.fingerprint:
        raise ValueError(
            f"manifest fingerprint does not match its contents: the file "
            f"says {stated[:12]} and the contents hash to "
            f"{m.fingerprint[:12]}. It has been edited since it was written.")
    return m
