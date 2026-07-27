"""Timestamp parsing helpers.

Everything is normalized to integer nanoseconds since the Unix epoch (UTC).
"""
from __future__ import annotations

from datetime import datetime, timezone

_NS_PER_S = 1_000_000_000


def epoch_to_ns(value: float | int, unit: str = "s") -> int:
    """Convert an epoch timestamp expressed in ``unit`` to nanoseconds.

    ``unit`` is one of ``s``, ``ms``, ``us``, ``ns``.
    """
    factors = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
    if unit not in factors:
        raise ValueError(f"unknown unit: {unit!r}")
    return int(round(float(value) * factors[unit]))


def iso_to_ns(value: str) -> int:
    """Parse an ISO-8601 string to nanoseconds since epoch (UTC).

    A trailing ``Z`` is accepted. Naive datetimes are assumed to be UTC.
    """
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return _dt_to_ns(dt)


def fix_utc_to_ns(value: str) -> int:
    """Parse a FIX ``UTCTimestamp`` (tag 60) to nanoseconds.

    Formats: ``YYYYMMDD-HH:MM:SS`` or ``YYYYMMDD-HH:MM:SS.sss``.
    """
    fmt = "%Y%m%d-%H:%M:%S.%f" if "." in value else "%Y%m%d-%H:%M:%S"
    dt = datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
    return _dt_to_ns(dt)


def _dt_to_ns(dt: datetime) -> int:
    dt = dt.astimezone(timezone.utc)
    return int(dt.timestamp() * _NS_PER_S) if dt.microsecond == 0 else (
        int(round(dt.timestamp() * _NS_PER_S))
    )
