

def test_the_typed_marker_ships_with_the_package():
    """The PEP 561 marker, without which the annotations are invisible.

    The classifier in `pyproject.toml` claims this package is typed. It said
    so once before while the marker was missing, which made the claim useless
    to every dependent; this test is what stops that recurring.
    """
    import mdnorm
    from pathlib import Path

    marker = Path(mdnorm.__file__).with_name("py.typed")
    assert marker.exists(), "py.typed is missing; the Typed classifier lies"


def test_filter_session_gives_back_what_it_was_given():
    """The element type survives the filter, at runtime as well as in types."""
    from decimal import Decimal

    from mdnorm import Bar, US_EQUITY_RTH, filter_session

    bars = [Bar(start_ns=0, interval_ns=60 * 10**9, open=Decimal("1"),
                high=Decimal("1"), low=Decimal("1"), close=Decimal("1"),
                volume=Decimal("1"), trades=1)]
    out = filter_session(bars, US_EQUITY_RTH)
    assert all(isinstance(b, Bar) for b in out)


def test_the_package_exports_no_duplicate_names():
    """One name in `__all__` twice means one of them is unreachable.

    `provenance` was written with a `ChangeKind` of its own, which silently
    shadowed the one `membership` has exported since 1.19.0 for index
    additions and deletions. Nothing failed — the older type simply stopped
    being reachable as `mdnorm.ChangeKind`. This is the check that would have
    caught it.
    """
    from collections import Counter

    import mdnorm

    repeated = [name for name, n in Counter(mdnorm.__all__).items() if n > 1]
    assert not repeated, f"exported twice: {repeated}"


def test_every_exported_name_actually_resolves():
    """An `__all__` entry with nothing behind it breaks `import *` silently."""
    import mdnorm

    missing = [n for n in mdnorm.__all__ if not hasattr(mdnorm, n)]
    assert not missing, f"named in __all__ but absent: {missing}"


def test_the_two_kinds_of_change_stay_distinct():
    """Index membership and run provenance are different questions."""
    import mdnorm
    from mdnorm.membership import ChangeKind
    from mdnorm.provenance import DriftKind

    assert mdnorm.ChangeKind is ChangeKind
    assert mdnorm.DriftKind is DriftKind
    assert ChangeKind is not DriftKind
