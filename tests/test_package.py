

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
