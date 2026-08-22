"""CLI tests (argument parsing and end-to-end file conversions)."""
import argparse
import csv

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
