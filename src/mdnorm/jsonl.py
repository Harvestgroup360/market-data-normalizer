"""NDJSON (JSON Lines) read/write helpers.

CSV is fine for tabular hand-offs, but modern data stacks — log shippers,
object stores, streaming loaders — speak newline-delimited JSON. These
helpers mirror :mod:`mdnorm.csvio`: events and bars go out as one JSON
object per line, and event files come back as normalized
:class:`MarketEvent` objects. Standard library only.
"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Iterable, Iterator, List, Optional, Union

from .bars import Bar
from .fileio import open_text
from .records import to_records
from .schema import EventType, MarketEvent, Side


def write_jsonl(
    items: Iterable[Union[MarketEvent, Bar]],
    path: str,
    *,
    as_float: bool = False,
) -> int:
    """Write events and/or bars to an NDJSON file. Returns the row count.

    Each record is one compact JSON object per line, produced by
    :func:`mdnorm.records.to_records` (Decimals as strings by default,
    ``as_float=True`` for numeric output).
    """
    n = 0
    with open_text(path, "w") as f:
        for r in to_records(items, as_float=as_float):
            f.write(json.dumps(r, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def _dec(value: object) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid decimal value: {value!r}") from exc


def event_from_dict(d: dict) -> MarketEvent:
    """Rebuild a :class:`MarketEvent` from a flat dict.

    Inverse of :func:`mdnorm.records.event_to_dict`; accepts both string
    (lossless) and numeric price/size values.
    """
    try:
        return MarketEvent(
            symbol=d["symbol"],
            venue=d["venue"],
            event_type=EventType(d["event_type"]),
            ts_ns=int(d["ts_ns"]),
            price=_dec(d.get("price")),
            size=_dec(d.get("size")),
            side=Side(d["side"]) if d.get("side") else None,
            bid_price=_dec(d.get("bid_price")),
            bid_size=_dec(d.get("bid_size")),
            ask_price=_dec(d.get("ask_price")),
            ask_size=_dec(d.get("ask_size")),
        )
    except KeyError as exc:
        raise ValueError(f"missing required field: {exc.args[0]}") from exc


def read_jsonl_events(path: str) -> List[MarketEvent]:
    """Read an NDJSON file of events back into :class:`MarketEvent` objects.

    Blank lines are skipped. Malformed lines raise :class:`ValueError`
    with the offending line number.
    """
    return list(iter_jsonl_events(path))


def iter_jsonl_events(path: str) -> Iterator[MarketEvent]:
    """Stream an NDJSON file of events, one :class:`MarketEvent` at a time.

    Lazy counterpart of :func:`read_jsonl_events`; transparently reads
    ``.jsonl.gz`` / ``.ndjson.gz``.
    """
    with open_text(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON ({exc})") from exc
            if not isinstance(d, dict):
                raise ValueError(f"line {line_no}: expected a JSON object")
            yield event_from_dict(d)
