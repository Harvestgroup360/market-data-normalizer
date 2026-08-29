"""CLI tests (argument parsing and end-to-end file conversions)."""
import argparse
import csv
from decimal import Decimal

import pytest

from mdnorm import read_jsonl_events
from mdnorm.cli import main, parse_interval

CSV_HEADER = "symbol,ts,price,size,side\n"
CSV_ROWS = (
    "BTCUSD,2026-08-04T00:00:01Z,100.0,1,buy\n"
    "BTCUSD,2026-08-04T00:00:30Z,101.0,2,sell\n"
    "BTCUSD,2026-08-04T00:01:10Z,100.5,1,buy\n"
)


@pytest.fixture()
def trades_csv(tmp_path):
    path = tmp_path / "trades.csv"
    path.write_text(CSV_HEADER + CSV_ROWS)
    return path


def test_parse_interval():
    assert parse_interval("30s") == 30 * 10**9
    assert parse_interval("1m") == 60 * 10**9
    assert parse_interval("4h") == 4 * 3600 * 10**9
    assert parse_interval("1d") == 86400 * 10**9


@pytest.mark.parametrize("bad", ["", "m", "1x", "-5m", "1.5h", "10"])
def test_parse_interval_rejects_garbage(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_interval(bad)


def test_bars_csv_to_csv(tmp_path, trades_csv, capsys):
    out = tmp_path / "bars.csv"
    rc = main(["bars", str(trades_csv), "--venue", "binance",
               "--interval", "1m", "-o", str(out)])
    assert rc == 0
    assert "wrote 2 bar(s)" in capsys.readouterr().out

    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["open"] == "100.0" and rows[0]["close"] == "101.0"
    assert rows[0]["trades"] == "2"


def test_bars_with_fill_gaps_and_float(tmp_path, capsys):
    src = tmp_path / "gappy.csv"
    src.write_text(
        CSV_HEADER
        + "BTCUSD,2026-08-04T00:00:01Z,100.0,1,buy\n"
        + "BTCUSD,2026-08-04T00:02:01Z,102.0,1,buy\n"  # minute 1 is empty
    )
    out = tmp_path / "bars.csv"
    rc = main(["bars", str(src), "--interval", "1m", "--fill-gaps",
               "--as-float", "-o", str(out)])
    assert rc == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3
    assert float(rows[1]["volume"]) == 0.0  # flat filler bar


def test_convert_csv_to_jsonl_round_trip(tmp_path, trades_csv, capsys):
    out = tmp_path / "events.jsonl"
    rc = main(["convert", str(trades_csv), "--venue", "kraken",
               "-o", str(out)])
    assert rc == 0
    events = read_jsonl_events(str(out))
    assert len(events) == 3
    assert events[0].venue == "kraken"
    assert events[0].symbol == "BTC-USD"


def test_bars_accepts_jsonl_input(tmp_path, trades_csv):
    mid = tmp_path / "events.jsonl"
    assert main(["convert", str(trades_csv), "-o", str(mid)]) == 0
    out = tmp_path / "bars.jsonl"
    assert main(["bars", str(mid), "--interval", "1m", "-o", str(out)]) == 0
    lines = [l for l in open(out) if l.strip()]
    assert len(lines) == 2


def test_quality_reports_issues(tmp_path, capsys):
    src = tmp_path / "dirty.csv"
    src.write_text(
        CSV_HEADER
        + "BTCUSD,2026-08-04T00:00:01Z,100.0,1,buy\n"
        + "BTCUSD,2026-08-04T00:00:02Z,1000.0,1,buy\n"  # outlier
        + "BTCUSD,2026-08-04T00:00:01Z,100.0,1,buy\n"   # out of order
    )
    assert main(["quality", str(src)]) == 0
    out = capsys.readouterr().out
    assert "2 issue(s)" in out
    assert "outlier: 1" in out and "out_of_order: 1" in out


def test_quality_clean_file(trades_csv, capsys):
    assert main(["quality", str(trades_csv)]) == 0
    assert "no issues found" in capsys.readouterr().out


def test_missing_input_is_reported(tmp_path, capsys):
    rc = main(["quality", str(tmp_path / "nope.csv")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


# -- metrics -----------------------------------------------------------------


def _returns_csv(path, values):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "ret"])
        for i, v in enumerate(values):
            w.writerow([i, v])
    return str(path)


_PNL = ["0.004", "-0.002", "0.006", "0.001", "-0.005", "0.003", "0.002",
        "-0.001", "0.005", "0.000", "0.002", "-0.003", "0.004", "0.001"]


def test_metrics_reports_the_basics(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    assert main(["metrics", src, "--column", "ret"]) == 0
    err = capsys.readouterr().err
    assert "observations         14" in err
    assert "Sharpe (per period)" in err
    assert "max drawdown" in err


def test_metrics_is_per_period_without_a_calendar(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    assert main(["metrics", src, "--column", "ret"]) == 0
    err = capsys.readouterr().err
    assert "Sharpe (annualised)  n/a" in err
    assert "no safe default" in err or "per period" in err


def test_metrics_annualises_when_given_the_calendar(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    rc = main(["metrics", src, "--column", "ret", "--interval", "1d",
               "--sessions-per-year", "252", "--session-length", "6h"])
    assert rc == 0
    assert "Sharpe (annualised)  n/a" not in capsys.readouterr().err


def test_metrics_warns_when_no_trial_count_is_given(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    assert main(["metrics", src, "--column", "ret"]) == 0
    assert "configurations were tried" in capsys.readouterr().err


def test_metrics_deflates_a_search(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    rc = main(["metrics", src, "--column", "ret",
               "--trials", "5000", "--trial-variance", "0.05"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "deflated (5000 trials)" in err
    assert "not clearly better" in err


def test_metrics_rejects_half_a_search(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    rc = main(["metrics", src, "--column", "ret", "--trials", "10"])
    assert rc == 1
    assert "must be given together" in capsys.readouterr().err


def test_metrics_accepts_prices(tmp_path, capsys):
    src = _returns_csv(tmp_path / "px.csv", ["100", "101", "99", "103", "102"])
    assert main(["metrics", src, "--column", "ret", "--prices"]) == 0
    err = capsys.readouterr().err
    assert "observations         4" in err
    assert "missing              1" in err  # the first return has no predecessor


def test_metrics_writes_a_csv(tmp_path):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    out = tmp_path / "metrics.csv"
    assert main(["metrics", src, "--column", "ret", "-o", str(out)]) == 0
    rows = {r[0]: r[1] for r in csv.reader(open(out))}
    assert rows["observations"] == "14"
    assert "sharpe_per_period" in rows
    assert rows["trials"] == "n/a"


def test_metrics_rejects_an_unknown_column(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    assert main(["metrics", src, "--column", "nope"]) == 1
    assert "no such column" in capsys.readouterr().err


def test_metrics_rejects_an_empty_file(tmp_path, capsys):
    src = tmp_path / "empty.csv"
    src.write_text("ts_ns,ret\n")
    assert main(["metrics", str(src), "--column", "ret"]) == 1
    assert "empty input" in capsys.readouterr().err


# -- costs -------------------------------------------------------------------


def _costs_csv(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "ret", "turnover"])
        for i, (r, t) in enumerate(zip(_PNL, ["0.3"] * len(_PNL))):
            w.writerow([i, r, t])
    return str(path)


def test_costs_prices_one_trade(capsys):
    rc = main(["costs", "--notional", "500000", "--quantity", "20000",
               "--fee-bps", "1", "--spread-bps", "4",
               "--impact-coefficient", "0.5",
               "--adv", "1000000", "--volatility", "0.02"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "commission         1" in err
    assert "spread             2" in err
    assert "impact             14.1421" in err


def test_costs_warns_when_the_model_ignores_size(capsys):
    rc = main(["costs", "--notional", "500000", "--quantity", "20000",
               "--fee-bps", "1", "--spread-bps", "4",
               "--adv", "1000000", "--volatility", "0.02"])
    assert rc == 0
    assert "does not depend on trade size" in capsys.readouterr().err


def test_costs_refuses_an_impact_model_without_liquidity(capsys):
    rc = main(["costs", "--notional", "1", "--quantity", "1",
               "--impact-coefficient", "0.5"])
    assert rc == 1
    assert "needs --adv and --volatility" in capsys.readouterr().err


def test_costs_applies_a_flat_cost_to_a_series(tmp_path, capsys):
    src = _costs_csv(tmp_path / "pnl.csv")
    assert main(["costs", src, "--cost-bps", "5"]) == 0
    err = capsys.readouterr().err
    assert "gross return" in err and "net return" in err
    assert "share of gross" in err


def test_costs_flags_a_result_the_costs_eat(tmp_path, capsys):
    src = _costs_csv(tmp_path / "pnl.csv")
    assert main(["costs", src, "--cost-bps", "60"]) == 0
    err = capsys.readouterr().err
    assert "not after them" in err or "consume" in err


def test_costs_writes_a_net_column(tmp_path):
    src = _costs_csv(tmp_path / "pnl.csv")
    out = tmp_path / "net.csv"
    assert main(["costs", src, "--cost-bps", "5", "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert "net" in rows[0]
    assert Decimal(rows[0]["net"]) < Decimal(rows[0]["ret"])


def test_costs_needs_a_cost_for_a_series(tmp_path, capsys):
    src = _costs_csv(tmp_path / "pnl.csv")
    assert main(["costs", src]) == 1
    assert "--cost-bps" in capsys.readouterr().err


def test_costs_reports_breakeven_and_capacity(capsys):
    rc = main(["costs", "--fee-bps", "1", "--spread-bps", "4",
               "--impact-coefficient", "0.5",
               "--adv", "1000000", "--volatility", "0.02",
               "--edge-bps", "20"])
    assert rc == 0
    err = capsys.readouterr().err
    # remaining 17 bps over a scale of 100 -> 0.17 squared
    assert "0.0289" in err
    assert "28900" in err


def test_costs_says_when_no_size_works(capsys):
    rc = main(["costs", "--fee-bps", "1", "--spread-bps", "4",
               "--impact-coefficient", "0.5",
               "--adv", "1000000", "--volatility", "0.02",
               "--edge-bps", "2"])
    assert rc == 0
    assert "already exceed the edge" in capsys.readouterr().err


def test_costs_rejects_half_a_liquidity_description(capsys):
    rc = main(["costs", "--adv", "1000000", "--edge-bps", "10"])
    assert rc == 1
    assert "must be given together" in capsys.readouterr().err


def test_costs_rejects_an_unknown_column(tmp_path, capsys):
    src = _costs_csv(tmp_path / "pnl.csv")
    assert main(["costs", src, "--cost-bps", "5", "--column", "nope"]) == 1
    assert "no such column" in capsys.readouterr().err


# -- the annualisation trap --------------------------------------------------


def test_metrics_warns_when_the_interval_exceeds_the_session(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    rc = main(["metrics", src, "--column", "ret", "--interval", "1d",
               "--sessions-per-year", "252", "--session-length", "6h"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "fewer than one bar per session" in err
    assert "63 periods a year" in err


def test_metrics_is_quiet_when_the_calendar_is_consistent(tmp_path, capsys):
    src = _returns_csv(tmp_path / "pnl.csv", _PNL)
    rc = main(["metrics", src, "--column", "ret", "--interval", "1d",
               "--sessions-per-year", "252", "--session-length", "1d"])
    assert rc == 0
    assert "fewer than one bar per session" not in capsys.readouterr().err


def test_features_warns_about_the_same_trap(tmp_path, capsys):
    src = tmp_path / "matrix.csv"
    with open(src, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "BTC"])
        for i, v in enumerate([100, 101, 102, 103, 104, 105]):
            w.writerow([i, v])
    out = tmp_path / "feats.csv"
    rc = main(["features", str(src), "-o", str(out), "--vol", "3",
               "--interval", "1d", "--sessions-per-year", "252",
               "--session-length", "6h"])
    assert rc == 0
    assert "fewer than one bar per session" in capsys.readouterr().err


# -- instruments -------------------------------------------------------------


_DAY = 86_400_000_000_000
_T0 = 1_700_000_000_000_000_000


def _day(n):
    return _T0 + n * _DAY


def _map_csv(path, rows=None):
    rows = rows if rows is not None else [
        ("ABC", "US0000000001", _day(0), _day(10)),
        ("ABC", "US0000000002", _day(20), ""),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "instrument_id", "start_ns", "end_ns"])
        for r in rows:
            w.writerow(r)
    return str(path)


def _px_csv(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "symbol", "px"])
        for i in (1, 5, 15, 25, 26):
            w.writerow([_day(i), "ABC", 10 + i])
    return str(path)


def test_instruments_reports_reuse(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    assert main(["instruments", m]) == 0
    err = capsys.readouterr().err
    assert "reused symbols       1" in err
    assert "US0000000001 -> US0000000002" in err


def test_instruments_flags_a_snapshot_file(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv",
                 [("AAA", "I1", _day(0), ""), ("BBB", "I2", _day(0), "")])
    assert main(["instruments", m]) == 0
    err = capsys.readouterr().err
    assert "cannot express reuse" in err
    assert "every binding is open-ended" in err


def test_instruments_rekeys_and_counts_reassignments(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    p = _px_csv(tmp_path / "px.csv")
    assert main(["instruments", m, p]) == 0
    err = capsys.readouterr().err
    assert "rows mapped          4" in err
    assert "rows unmapped        1" in err
    assert "rows reassigned      2" in err
    assert "spliced onto the wrong history" in err


def test_instruments_writes_the_instrument_id(tmp_path):
    m = _map_csv(tmp_path / "map.csv")
    p = _px_csv(tmp_path / "px.csv")
    out = tmp_path / "keyed.csv"
    assert main(["instruments", m, p, "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 4
    assert {r["instrument_id"] for r in rows} == {"US0000000001", "US0000000002"}


def test_instruments_can_keep_unmapped_rows(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    p = _px_csv(tmp_path / "px.csv")
    out = tmp_path / "keyed.csv"
    assert main(["instruments", m, p, "--keep-unmapped", "-o", str(out)]) == 0
    assert "(kept)" in capsys.readouterr().err
    assert len(list(csv.DictReader(open(out)))) == 5


def test_instruments_segments_one_ticker(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    p = _px_csv(tmp_path / "px.csv")
    assert main(["instruments", m, p, "--segments", "ABC"]) == 0
    err = capsys.readouterr().err
    assert "ABC: 2 segment(s), 1 unresolved" in err
    assert "not one instrument" in err


def test_instruments_rejects_an_overlapping_map(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv",
                 [("ABC", "I1", _day(0), _day(10)), ("ABC", "I2", _day(5), "")])
    assert main(["instruments", m]) == 1
    assert "two instruments at the same time" in capsys.readouterr().err


def test_instruments_rejects_a_map_with_a_hole(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv", [("ABC", "", _day(0), "")])
    assert main(["instruments", m]) == 1
    assert "instrument_id" in capsys.readouterr().err


def test_instruments_rejects_an_unknown_column(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    p = _px_csv(tmp_path / "px.csv")
    assert main(["instruments", m, p, "--symbol-field", "nope"]) == 1
    assert "no such column" in capsys.readouterr().err


def test_instruments_reports_unparseable_timestamps(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv")
    p = tmp_path / "px.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "symbol"])
        w.writerow(["not-a-number", "ABC"])
        w.writerow([_day(1), "ABC"])
    assert main(["instruments", m, str(p)]) == 0
    err = capsys.readouterr().err
    assert "unparseable ts_ns" in err
    assert "rows mapped          1" in err


def test_instruments_says_when_nothing_was_reassigned(tmp_path, capsys):
    m = _map_csv(tmp_path / "map.csv", [("ABC", "I1", _day(0), "")])
    p = _px_csv(tmp_path / "px.csv")
    assert main(["instruments", m, p]) == 0
    err = capsys.readouterr().err
    assert "rows reassigned      0" in err
    assert "no row needed reassigning" in err


# -- mixfreq ----------------------------------------------------------------


def _periods_csv(path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "end", "value"])
        w.writerow([0, 86_400_000_000_000, "1.5"])
        w.writerow([86_400_000_000_000, 172_800_000_000_000, "2.5"])
    return str(path)


def test_mixfreq_reports_the_leak(tmp_path, capsys):
    p = _periods_csv(tmp_path / "periods.csv")
    assert main(["mixfreq", p, "--interval", str(21_600_000_000_000)]) == 0
    err = capsys.readouterr().err
    assert "periods              2" in err
    assert "of those, too early" in err
    assert "every grid point leaks" in err


def test_mixfreq_warns_that_zero_lag_is_a_claim(tmp_path, capsys):
    p = _periods_csv(tmp_path / "periods.csv")
    assert main(["mixfreq", p, "--interval", str(86_400_000_000_000)]) == 0
    assert "publication lag is zero" in capsys.readouterr().err


def test_mixfreq_lag_is_not_warned_about_when_stated(tmp_path, capsys):
    p = _periods_csv(tmp_path / "periods.csv")
    assert main(["mixfreq", p, "--interval", str(86_400_000_000_000),
                 "--lag", str(3_600_000_000_000)]) == 0
    assert "publication lag is zero" not in capsys.readouterr().err


def test_mixfreq_writes_both_joins_side_by_side(tmp_path, capsys):
    p = _periods_csv(tmp_path / "periods.csv")
    out = tmp_path / "joined.csv"
    assert main(["mixfreq", p, "--interval", str(43_200_000_000_000),
                 "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["label"] == "1.5"
    assert rows[0]["knowable"] == ""        # the point of the whole module
    assert any(r["knowable"] for r in rows)


def test_mixfreq_rejects_an_empty_file(tmp_path, capsys):
    p = tmp_path / "empty.csv"
    p.write_text("start,end,value\n")
    assert main(["mixfreq", str(p), "--interval", "1"]) == 1
    assert "no periods" in capsys.readouterr().err


def test_mixfreq_reports_a_missing_column(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    p.write_text("start,value\n0,1.5\n")
    assert main(["mixfreq", str(p), "--interval", "1"]) == 1
    assert "error:" in capsys.readouterr().err


# -- membership -------------------------------------------------------------


def _changes_csv(path, rows=None):
    rows = rows if rows is not None else [
        ("A", "add", 0, ""),
        ("B", "add", 0, ""),
        ("B", "delete", 86_400_000_000_000 * 10, 86_400_000_000_000 * 7),
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instrument_id", "action", "effective", "announced"])
        for r in rows:
            w.writerow(r)
    return str(path)


def test_membership_reports_the_file(tmp_path, capsys):
    p = _changes_csv(tmp_path / "idx.csv")
    assert main(["membership", p]) == 0
    err = capsys.readouterr().err
    assert "instruments          2" in err
    assert "  deletions          1" in err


def test_membership_flags_a_today_list(tmp_path, capsys):
    p = _changes_csv(tmp_path / "idx.csv",
                     [("A", "add", 0, ""), ("B", "add", 0, "")])
    assert main(["membership", p]) == 0
    err = capsys.readouterr().err
    assert "nothing ever left this index" in err
    assert "no change carries an announcement date" in err


def test_membership_sizes_the_survivorship_gap(tmp_path, capsys):
    p = _changes_csv(tmp_path / "idx.csv")
    assert main(["membership", p, "--at", str(86_400_000_000_000)]) == 0
    err = capsys.readouterr().err
    assert "a today-list would drop  1" in err
    assert "dropped: B" in err


def test_membership_announced_basis_moves_the_composition(tmp_path, capsys):
    p = _changes_csv(tmp_path / "idx.csv")
    at = str(86_400_000_000_000 * 8)          # after announcement, before effect
    assert main(["membership", p, "--at", at, "--basis", "announced"]) == 0
    assert "on the announced basis: 1" in capsys.readouterr().err
    assert main(["membership", p, "--at", at, "--basis", "effective"]) == 0
    assert "on the effective basis: 2" in capsys.readouterr().err


def test_membership_output_requires_a_moment(tmp_path, capsys):
    p = _changes_csv(tmp_path / "idx.csv")
    out = tmp_path / "m.csv"
    assert main(["membership", p, "-o", str(out)]) == 1
    assert "needs --at" in capsys.readouterr().err


def test_membership_rejects_an_empty_file(tmp_path, capsys):
    p = tmp_path / "e.csv"
    p.write_text("instrument_id,action,effective\n")
    assert main(["membership", str(p)]) == 1
    assert "no changes" in capsys.readouterr().err


# -- reconcile --------------------------------------------------------------


def _series_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_ns", "value"])
        for r in rows:
            w.writerow(r)
    return str(path)


def test_reconcile_separates_coverage_from_content(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(0, "100"), (1, "101"), (2, "102")])
    b = _series_csv(tmp_path / "b.csv", [(0, "100"), (1, "999")])
    assert main(["reconcile", a, b]) == 0
    err = capsys.readouterr().err
    assert "shared timestamps    2" in err
    assert "  agreed             1" in err
    assert "  differed           1" in err
    assert "only in left         1" in err


def test_reconcile_warns_that_no_tolerance_means_exact(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(0, "100")])
    b = _series_csv(tmp_path / "b.csv", [(0, "100")])
    assert main(["reconcile", a, b]) == 0
    assert "no tolerance given" in capsys.readouterr().err


def test_reconcile_tolerance_admits_a_small_difference(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(0, "100.00")])
    b = _series_csv(tmp_path / "b.csv", [(0, "100.01")])
    assert main(["reconcile", a, b, "--relative", "0.001"]) == 0
    err = capsys.readouterr().err
    assert "agreement            100.00%" in err
    assert "no tolerance given" not in err


def test_reconcile_diagnoses_a_clock_offset(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(i * 10**9, "100") for i in range(5)])
    b = _series_csv(tmp_path / "b.csv",
                    [(i * 10**9 + 250 * 10**6, "100") for i in range(5)])
    assert main(["reconcile", a, b]) == 0
    err = capsys.readouterr().err
    assert "shared timestamps    0" in err
    assert "clock difference, not a disagreement" in err


def test_reconcile_applies_a_stated_shift(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(i * 10**9, "100") for i in range(5)])
    b = _series_csv(tmp_path / "b.csv",
                    [(i * 10**9 + 250 * 10**6, "100") for i in range(5)])
    assert main(["reconcile", a, b, "--shift", str(-250 * 10**6)]) == 0
    assert "agreement            100.00%" in capsys.readouterr().err


def test_reconcile_writes_the_mismatches(tmp_path):
    a = _series_csv(tmp_path / "a.csv", [(0, "100"), (1, "101")])
    b = _series_csv(tmp_path / "b.csv", [(0, "105")])
    out = tmp_path / "mm.csv"
    assert main(["reconcile", a, b, "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    kinds = {r["kind"] for r in rows}
    assert kinds == {"value", "only_left"}
    value_row = next(r for r in rows if r["kind"] == "value")
    assert value_row["difference"] == "5"


def test_reconcile_rejects_an_empty_input(tmp_path, capsys):
    a = _series_csv(tmp_path / "a.csv", [(0, "100")])
    b = tmp_path / "b.csv"
    b.write_text("ts_ns,value\n")
    assert main(["reconcile", a, str(b)]) == 1
    assert "must contain observations" in capsys.readouterr().err


# -- calendar ---------------------------------------------------------------


def _calendar_csv(path, rows):
    path.write_text("date,kind,close,name\n" + "".join(rows))
    return str(path)


US = ["--session", "09:30-16:00", "--tz", "America/New_York"]


def _us_2026(tmp_path):
    return _calendar_csv(tmp_path / "cal.csv", [
        "2026-01-01,holiday,,New Year's Day\n",
        "2026-07-03,holiday,,Independence Day\n",
        "2026-11-26,holiday,,Thanksgiving\n",
        "2026-11-27,early_close,13:00,Day after Thanksgiving\n",
        "2026-12-25,holiday,,Christmas Day\n",
    ])


def test_calendar_counts_the_sessions_it_was_given(tmp_path, capsys):
    assert main(["calendar", _us_2026(tmp_path)] + US) == 0
    err = capsys.readouterr().err
    assert "calendar covers      2026-01-01..2026-12-31" in err
    assert "trading days         257" in err     # 365 - 104 weekends - 4
    assert "early closes         1" in err


def test_calendar_reports_the_year_rather_than_the_convention(tmp_path, capsys):
    assert main(["calendar", _us_2026(tmp_path)] + US) == 0
    err = capsys.readouterr().err
    assert "--sessions-per-year 257" in err
    assert "not the conventional 252" in err


def test_calendar_prices_the_early_closes(tmp_path, capsys):
    assert main(["calendar", _us_2026(tmp_path)] + US) == 0
    err = capsys.readouterr().err
    assert "early closes cost 180 minute(s)" in err


def test_calendar_narrowed_to_a_week(tmp_path, capsys):
    assert main(["calendar", _us_2026(tmp_path)] + US +
                ["--from", "2026-11-23", "--to", "2026-11-27"]) == 0
    err = capsys.readouterr().err
    assert "trading days         4" in err         # Thanksgiving is shut
    assert "trading minutes      1380" in err      # three full days plus 210
    assert "not a whole year" in err


def test_calendar_writes_the_days_and_marks_the_short_ones(tmp_path):
    out = tmp_path / "days.csv"
    assert main(["calendar", _us_2026(tmp_path)] + US +
                ["--from", "2026-11-23", "--to", "2026-11-27",
                 "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["date"] for r in rows] == ["2026-11-23", "2026-11-24",
                                         "2026-11-25", "2026-11-27"]
    assert [r["early"] for r in rows] == ["0", "0", "0", "1"]


def test_calendar_refuses_a_day_it_does_not_cover(tmp_path, capsys):
    assert main(["calendar", _us_2026(tmp_path)] + US +
                ["--to", "2027-01-04"]) == 1
    assert "outside this calendar" in capsys.readouterr().err


def test_calendar_accepts_a_stated_range_for_an_empty_file(tmp_path, capsys):
    path = _calendar_csv(tmp_path / "empty.csv", [])
    assert main(["calendar", path] + US +
                ["--first-day", "2026-01-01", "--last-day", "2026-01-31"]) == 0
    err = capsys.readouterr().err
    assert "trading days         22" in err
    assert "holidays           0" in err


# -- fx ---------------------------------------------------------------------


def _rates_csv(path, rows):
    path.write_text("pair,ts_ns,rate\n" + "".join(f"{p},{t},{r}\n"
                                                  for p, t, r in rows))
    return str(path)


HOUR_NS = 3600 * 10**9


def _fx_fixture(tmp_path):
    prices = _series_csv(tmp_path / "px.csv",
                         [(0, "100"), (HOUR_NS, "100")])
    rates = _rates_csv(tmp_path / "fx.csv",
                       [("EUR/USD", 0, "1.10"), ("EUR/USD", HOUR_NS, "1.20")])
    return prices, rates


def test_fx_converts_each_point_at_its_own_rate(tmp_path, capsys):
    px, fx = _fx_fixture(tmp_path)
    out = tmp_path / "usd.csv"
    assert main(["fx", px, fx, "--from", "EUR", "--to", "USD",
                 "--max-age", "1d", "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert [r["value"] for r in rows] == ["110.00", "120.00"]


def test_fx_reports_the_move_the_single_rate_would_have_hidden(tmp_path, capsys):
    px, fx = _fx_fixture(tmp_path)
    assert main(["fx", px, fx, "--from", "EUR", "--to", "USD",
                 "--max-age", "1d"]) == 0
    err = capsys.readouterr().err
    assert "the rate moved +9.09%" in err
    assert "did not exist until the end" in err


def test_fx_drops_and_counts_what_it_cannot_convert(tmp_path, capsys):
    px = _series_csv(tmp_path / "px.csv", [(0, "100"), (10**18, "100")])
    fx = _rates_csv(tmp_path / "fx.csv", [("EUR/USD", 0, "1.10")])
    assert main(["fx", px, fx, "--from", "EUR", "--to", "USD",
                 "--max-age", "1h"]) == 0
    err = capsys.readouterr().err
    assert "no usable rate       1" in err
    assert "dropped rather than converted against an older one" in err


def test_fx_says_when_it_used_a_rate_upside_down(tmp_path, capsys):
    px, fx = _fx_fixture(tmp_path)
    assert main(["fx", px, fx, "--from", "USD", "--to", "EUR",
                 "--max-age", "1d"]) == 0
    assert "upside-down" in capsys.readouterr().err


def test_fx_can_be_told_not_to_invert(tmp_path, capsys):
    px, fx = _fx_fixture(tmp_path)
    assert main(["fx", px, fx, "--from", "USD", "--to", "EUR",
                 "--max-age", "1d", "--no-inverse"]) == 1
    assert "state a vehicle currency" in capsys.readouterr().err


def test_fx_refuses_a_cross_without_a_vehicle(tmp_path, capsys):
    px = _series_csv(tmp_path / "px.csv", [(0, "100")])
    fx = _rates_csv(tmp_path / "fx.csv",
                    [("EUR/USD", 0, "1.10"), ("USD/JPY", 0, "150")])
    assert main(["fx", px, fx, "--from", "EUR", "--to", "JPY",
                 "--max-age", "1d"]) == 1
    assert "state a vehicle currency" in capsys.readouterr().err


def test_fx_crosses_when_the_vehicle_is_stated(tmp_path, capsys):
    px = _series_csv(tmp_path / "px.csv", [(0, "100")])
    fx = _rates_csv(tmp_path / "fx.csv",
                    [("EUR/USD", 0, "1.10"), ("USD/JPY", 0, "150")])
    out = tmp_path / "jpy.csv"
    assert main(["fx", px, fx, "--from", "EUR", "--to", "JPY",
                 "--max-age", "1d", "--via", "USD", "-o", str(out)]) == 0
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["value"] == "16500.00"
    assert "both legs' spreads" in capsys.readouterr().err


def test_fx_requires_a_max_age(tmp_path, capsys):
    px, fx = _fx_fixture(tmp_path)
    with pytest.raises(SystemExit):
        main(["fx", px, fx, "--from", "EUR", "--to", "USD"])


def test_fx_rejects_an_empty_input(tmp_path, capsys):
    px = tmp_path / "px.csv"
    px.write_text("ts_ns,value\n")
    fx = _rates_csv(tmp_path / "fx.csv", [("EUR/USD", 0, "1.10")])
    assert main(["fx", str(px), fx, "--from", "EUR", "--to", "USD",
                 "--max-age", "1h"]) == 1
    assert "no observations" in capsys.readouterr().err
