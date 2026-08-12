"""Back-adjustment tests: splits, dividends, contract rolls."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from mdnorm import (
    AdjustMethod,
    Bar,
    EventType,
    MarketEvent,
    Pipeline,
    Side,
    adjust_bars,
    adjust_events,
    adjustment_at,
    dividend,
    read_actions_csv,
    roll,
    split,
)

DAY_NS = 86_400_000_000_000


def ns(iso: str) -> int:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp()) * 1_000_000_000


def trade(ts_ns, price, size="10"):
    return MarketEvent(
        symbol="AAPL", venue="x", event_type=EventType.TRADE,
        ts_ns=ts_ns, price=Decimal(price), size=Decimal(size), side=Side.BUY,
    )


def bar(start_ns, o, h, low, c, vol="100"):
    return Bar(
        start_ns=start_ns, interval_ns=DAY_NS,
        open=Decimal(o), high=Decimal(h), low=Decimal(low), close=Decimal(c),
        volume=Decimal(vol), trades=5, vwap=Decimal(c),
    )


D1 = ns("2026-03-02T00:00:00Z")
D2 = ns("2026-03-03T00:00:00Z")
D3 = ns("2026-03-04T00:00:00Z")
D4 = ns("2026-03-05T00:00:00Z")


# -- splits ----------------------------------------------------------------

def test_split_divides_earlier_prices_and_multiplies_sizes():
    events = [trade(D1, "400", "10"), trade(D3, "101", "40")]
    out = adjust_events(events, [split(D2, Decimal("4"))])
    assert out[0].price == Decimal("100")   # 400 / 4
    assert out[0].size == Decimal("40")     # 10 * 4
    # The most recent segment is left exactly as it printed.
    assert out[1].price == Decimal("101")
    assert out[1].size == Decimal("40")


def test_split_makes_the_return_across_the_event_continuous():
    """A 4-for-1 on an unchanged stock must produce a zero return."""
    events = [trade(D1, "400"), trade(D3, "100")]
    a, b = adjust_events(events, [split(D2, Decimal("4"))])
    assert b.price == a.price


def test_reverse_split():
    events = [trade(D1, "2"), trade(D3, "20")]
    out = adjust_events(events, [split(D2, Decimal("0.1"))])
    assert out[0].price == Decimal("20")


def test_action_exactly_on_a_timestamp_leaves_that_print_alone():
    """Adjustment applies strictly before the action."""
    events = [trade(D2, "100")]
    out = adjust_events(events, [split(D2, Decimal("4"))])
    assert out[0].price == Decimal("100")


def test_splits_compound():
    events = [trade(D1, "600")]
    out = adjust_events(
        events, [split(D2, Decimal("2")), split(D3, Decimal("3"))]
    )
    assert out[0].price == Decimal("100")   # 600 / (2 * 3)


def test_split_adjusts_quote_sides():
    q = MarketEvent(
        symbol="AAPL", venue="x", event_type=EventType.QUOTE, ts_ns=D1,
        bid_price=Decimal("400"), bid_size=Decimal("5"),
        ask_price=Decimal("404"), ask_size=Decimal("5"),
    )
    out = adjust_events([q], [split(D2, Decimal("4"))])[0]
    assert out.bid_price == Decimal("100")
    assert out.ask_price == Decimal("101")
    assert out.bid_size == Decimal("20")


# -- dividends -------------------------------------------------------------

def test_dividend_ratio_uses_an_explicit_reference_price():
    events = [trade(D1, "100"), trade(D3, "99.50")]
    out = adjust_events(
        events, [dividend(D2, Decimal("0.50"), ref_price=Decimal("100"))]
    )
    assert out[0].price == Decimal("99.500")   # 100 * (1 - 0.5/100)


def test_dividend_reference_is_read_from_the_series_when_omitted():
    """The reference is the last raw print before the ex-date."""
    events = [trade(D1, "90"), trade(D2 - 1, "100"), trade(D3, "99.50")]
    out = adjust_events(events, [dividend(D2, Decimal("0.50"))])
    assert out[1].price == Decimal("99.500")   # ref = 100, the print before
    assert out[0].price == Decimal("89.550")   # 90 * 0.995, same factor


def test_dividend_difference_method_subtracts_the_cash():
    events = [trade(D1, "100")]
    out = adjust_events(
        events, [dividend(D2, Decimal("0.50"))],
        method=AdjustMethod.DIFFERENCE,
    )
    assert out[0].price == Decimal("99.50")


def test_dividend_leaves_size_alone():
    events = [trade(D1, "100", "7")]
    out = adjust_events(
        events, [dividend(D2, Decimal("1"), ref_price=Decimal("100"))]
    )
    assert out[0].size == Decimal("7")


def test_dividend_without_a_usable_reference_is_an_error():
    events = [trade(D3, "99")]           # nothing before the ex-date
    with pytest.raises(ValueError, match="reference price"):
        adjust_events(events, [dividend(D2, Decimal("0.50"))])


def test_dividend_larger_than_the_reference_is_an_error():
    events = [trade(D1, "0.40")]
    with pytest.raises(ValueError, match="not smaller than"):
        adjust_events(events, [dividend(D2, Decimal("0.50"))])


# -- futures rolls ---------------------------------------------------------

def test_roll_difference_removes_the_contract_spread():
    """Front contract at 5290.25 rolls to the next at 5312.50."""
    events = [trade(D1, "5280"), trade(D3, "5320")]
    out = adjust_events(
        events,
        [roll(D2, Decimal("5290.25"), Decimal("5312.50"))],
        method=AdjustMethod.DIFFERENCE,
    )
    assert out[0].price == Decimal("5302.25")   # 5280 + 22.25
    assert out[1].price == Decimal("5320")


def test_roll_ratio_scales_instead():
    events = [trade(D1, "100")]
    out = adjust_events(events, [roll(D2, Decimal("100"), Decimal("110"))])
    assert out[0].price == Decimal("110")


def test_roll_difference_can_drive_a_long_history_non_positive():
    """A documented artefact of difference adjustment, not a crash."""
    events = [trade(D1, "10")]
    out = adjust_events(
        events, [roll(D2, Decimal("100"), Decimal("80"))],
        method=AdjustMethod.DIFFERENCE,
    )
    assert out[0].price == Decimal("-10")


def test_roll_requires_positive_prices():
    with pytest.raises(ValueError):
        roll(D2, Decimal("0"), Decimal("100"))


# -- mixed sequences -------------------------------------------------------

def test_split_and_dividend_compose_in_the_right_order():
    events = [trade(D1, "400")]
    out = adjust_events(events, [
        dividend(D2, Decimal("1"), ref_price=Decimal("400")),
        split(D3, Decimal("4")),
    ])
    # dividend factor 399/400, then the split divides by 4
    assert out[0].price == Decimal("400") * Decimal("399") / Decimal("400") / 4


def test_actions_supplied_out_of_order_are_sorted():
    events = [trade(D1, "600")]
    unordered = [split(D3, Decimal("3")), split(D2, Decimal("2"))]
    assert adjust_events(events, unordered)[0].price == Decimal("100")


def test_no_actions_returns_the_input_unchanged():
    events = [trade(D1, "100")]
    assert adjust_events(events, []) == events


def test_adjustment_at_reports_the_factors_in_force():
    actions = [split(D2, Decimal("2")), split(D3, Decimal("5"))]
    factor, offset, size = adjustment_at(D1, actions)
    assert factor == Decimal(1) / Decimal(10)
    assert offset == 0 and size == Decimal(10)
    # after every action, nothing applies
    assert adjustment_at(D4, actions) == (Decimal(1), Decimal(0), Decimal(1))


# -- bars ------------------------------------------------------------------

def test_adjust_bars_restates_ohlc_and_volume():
    bars = [bar(D1, "400", "440", "396", "420", "100"), bar(D3, "105", "106", "104", "105", "400")]
    out = adjust_bars(bars, [split(D2, Decimal("4"))])
    assert (out[0].open, out[0].high, out[0].low, out[0].close) == (
        Decimal("100"), Decimal("110"), Decimal("99"), Decimal("105"),
    )
    assert out[0].volume == Decimal("400")
    assert out[0].vwap == Decimal("105")
    assert out[0].trades == 5 and out[0].start_ns == D1
    assert out[1].close == Decimal("105")   # untouched


def test_adjust_bars_dividend_reference_comes_from_the_prior_close():
    bars = [bar(D1, "100", "100", "100", "100"), bar(D3, "99", "99", "99", "99")]
    out = adjust_bars(bars, [dividend(D2, Decimal("1"))])
    assert out[0].close == Decimal("99")    # 100 * (1 - 1/100)


def test_pipeline_adjust_step_on_events_then_bars():
    events = [trade(D1, "400", "10"), trade(D3, "100", "10")]
    pipe = Pipeline().adjust([split(D2, Decimal("4"))]).time_bars(DAY_NS)
    bars = pipe.run(events)
    assert pipe.steps == ["adjust", "time_bars"]
    assert bars[0].close == Decimal("100") and bars[1].close == Decimal("100")


def test_pipeline_adjust_step_after_aggregation():
    events = [trade(D1, "400"), trade(D3, "100")]
    pipe = Pipeline().time_bars(DAY_NS).adjust([split(D2, Decimal("4"))])
    bars = pipe.run(events)
    assert bars[0].close == Decimal("100")


# -- CSV input -------------------------------------------------------------

def test_read_actions_csv(tmp_path):
    path = tmp_path / "actions.csv"
    path.write_text(
        "timestamp,kind,value,ref_price\n"
        "2026-03-03T00:00:00Z,split,4,\n"
        "2026-03-04T00:00:00Z,dividend,0.25,190.50\n"
        "\n"
        "2026-03-05T00:00:00Z,roll,5312.50,5290.25\n"
    )
    actions = read_actions_csv(str(path))
    assert len(actions) == 3
    assert actions[0].value == Decimal("4")
    assert actions[1].ref_price == Decimal("190.50")
    # for a roll, value is the new leg and ref_price the expiring one
    assert actions[2].value == Decimal("5312.50")
    assert actions[2].ref_price == Decimal("5290.25")


def test_read_actions_csv_reports_the_offending_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,kind,value\n"
        "2026-03-03T00:00:00Z,split,4\n"
        "2026-03-04T00:00:00Z,merger,1\n"
    )
    with pytest.raises(ValueError, match=r"bad\.csv:3:.*unknown kind"):
        read_actions_csv(str(path))


def test_read_actions_csv_rejects_a_roll_without_a_reference(tmp_path):
    path = tmp_path / "roll.csv"
    path.write_text("timestamp,kind,value\n2026-03-03T00:00:00Z,roll,100\n")
    with pytest.raises(ValueError, match="ref_price"):
        read_actions_csv(str(path))
