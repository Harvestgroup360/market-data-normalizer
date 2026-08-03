"""File-level CSV read/write helpers.

Row-level (`from_csv_row`) and dict flattening (`to_records`) are the building
blocks; these wrap them for whole files, so a CSV on disk becomes a list of
normalized events, and events or bars go back out to a tidy CSV. Standard
library only.
"""
from __future__ import annotations

import csv
from typing import Iterable, List, Mapping, Optional, Union

from .bars import Bar
from .normalizers import from_csv_row
from .records import to_records
from .schema import MarketEvent


def read_csv_trades(
    path: str,
    *,
    venue: str,
    mapping: Optional[Mapping[str, str]] = None,
    ts_unit: Optional[str] = None,
) -> List[MarketEvent]:
    """Read a CSV file of trades into normalized :class:`MarketEvent` objects.

    Columns are mapped and parsed by :func:`from_csv_row`; ``mapping`` renames
    columns and ``ts_unit`` selects epoch parsing (otherwise ISO-8601).
    """
    out: List[MarketEvent] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(
                from_csv_row(row, venue=venue, mapping=mapping, ts_unit=ts_unit)
            )
    return out


def write_records_csv(
    items: Iterable[Union[MarketEvent, Bar]],
    path: str,
    *,
    as_float: bool = False,
) -> int:
    """Write events and/or bars to a CSV file. Returns the number of rows.

    Fields are the union of keys across all records, so mixed events/bars still
    produce a well-formed file (missing cells are left blank).
    """
    records = to_records(items, as_float=as_float)
    if not records:
        open(path, "w", encoding="utf-8").close()
        return 0

    fieldnames: List[str] = []
    for r in records:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(records)
    return len(records)
