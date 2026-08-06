"""Shared file-opening helper with transparent gzip support.

Any path ending in ``.gz`` is opened through :mod:`gzip` in text mode, so
multi-gigabyte compressed dumps flow through the same readers and writers
as plain files. Standard library only.
"""
from __future__ import annotations

import gzip
from typing import IO


def open_text(path: str, mode: str = "r") -> IO[str]:
    """Open ``path`` in text mode, decompressing/compressing ``.gz`` files.

    ``mode`` is ``"r"`` or ``"w"``; newline handling matches what
    :mod:`csv` expects (``newline=""``).
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return open(path, mode, encoding="utf-8", newline="")
