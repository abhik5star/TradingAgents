"""The demo: one command, real engine, bars that admit what they are.

Two things are worth testing here. First, that the demo's generator is the *same*
generator as the test fixtures — the demo exists to show numbers a reader can
reproduce, and reproducibility is worthless if the series quietly drifts. Second,
that ``research demo`` really delegates to ``run``/``paper``/``serve`` with every
option filled in: calling a Typer command as a plain function silently passes the
``OptionInfo`` defaults, which is exactly the kind of bug that would turn a demo
into a stack trace.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

sys.path.insert(0, str(Path(__file__).parent))

from research_fixtures import synthetic_bars  # noqa: E402  (path set above)

from tradingagents.research import demo as demo_module  # noqa: E402
from tradingagents.research.demo import (  # noqa: E402
    DEMO_DISCLAIMER,
    DEMO_SYMBOLS,
    DEMO_WINDOW,
    demo_bars,
    demo_summary,
    write_demo_csvs,
)


def test_demo_bars_are_the_test_fixtures():
    """Same seed, same bars: the demo can reproduce the numbers in the docs."""
    for seed, days, start in ((51, 1700, "2015-01-02"), (5, 520, "2022-01-03"),
                              (23, 520, "2022-01-03")):
        expected = synthetic_bars(n=days, seed=seed, start=start)
        actual = demo_bars(days=days, seed=seed, start=start)
        assert list(actual.columns) == list(expected.columns)
        assert actual.index.equals(expected.index)
        assert actual.to_numpy() == pytest.approx(expected.to_numpy(), rel=0, abs=0)


def test_demo_bars_are_deterministic_and_well_formed():
    first = demo_bars(days=120, seed=7, start="2023-01-02")
    second = demo_bars(days=120, seed=7, start="2023-01-02")
    assert first.equals(second)

    other = demo_bars(days=120, seed=8, start="2023-01-02")
    assert not first["close"].equals(other["close"])

    assert len(first) == 120
    assert first.index.is_monotonic_increasing
    assert str(first.index.tz) == "UTC"
    assert (first["high"] >= first[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (first["low"] <= first[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (first[["open", "high", "low", "close"]] > 0).all().all()


def test_write_demo_csvs_writes_every_symbol_once(tmp_path):
    written = write_demo_csvs(tmp_path)
    assert set(written) == set(DEMO_SYMBOLS)
    for symbol, path in written.items():
        assert path.name == f"{symbol}-1d.csv"
        header = path.read_text().splitlines()[0]
        assert header.split(",")[0] == "date"
        assert header.split(",")[1:] == ["open", "high", "low", "close", "volume"]

    # A second call reuses the files instead of rewriting them.
    stamp = tmp_path / "SYNTH-1d.csv"
    before = stamp.stat().st_mtime_ns
    write_demo_csvs(tmp_path)
    assert stamp.stat().st_mtime_ns == before

    # ... unless asked, which is the only way to pick up a generator change.
    write_demo_csvs(tmp_path, force=True)
    assert stamp.stat().st_mtime_ns != before


def test_demo_csvs_load_through_the_real_data_source(tmp_path):
    """The CSV the demo writes is a CSV the research engine accepts unmodified."""
    from tradingagents.research.contracts import Timeframe
    from tradingagents.research.data import CsvDataSource

    write_demo_csvs(tmp_path)
    frame = CsvDataSource(tmp_path).load("SYNTH", Timeframe.D1, DEMO_WINDOW["start"],
                                         DEMO_WINDOW["paper_end"])
    assert len(frame) == 1700
    assert {"open", "high", "low", "close", "volume"} <= set(frame.columns)


def test_demo_summary_reports_bars_and_prices(tmp_path):
    rows = demo_summary(write_demo_csvs(tmp_path))
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["SYNTH"]["bars"] == DEMO_SYMBOLS["SYNTH"]["days"]
    assert by_symbol["ALPHA"]["bars"] == DEMO_SYMBOLS["ALPHA"]["days"]
    assert by_symbol["SYNTH"]["first"] > 0
    assert all(row["role"] for row in rows)


def test_demo_summary_survives_an_unreadable_file(tmp_path):
    broken = tmp_path / "SYNTH-1d.csv"
    broken.write_text("not,a,bar\n1,2,3\n")
    rows = demo_summary({"SYNTH": broken})
    assert rows[0]["symbol"] == "SYNTH" and rows[0]["bars"] in (0, 2)


def test_demo_disclaimer_says_what_the_bars_are():
    lowered = DEMO_DISCLAIMER.lower()
    assert "synthetic" in lowered
    assert "not a market" in lowered
    assert "guarantee" not in lowered


def test_demo_window_leaves_forward_bars_after_the_research_sample():
    """Paper trading needs bars the research window never saw."""
    assert DEMO_WINDOW["start"] < DEMO_WINDOW["research_end"] < DEMO_WINDOW["paper_end"]


# --------------------------------------------------------------------------------------
# the command itself
# --------------------------------------------------------------------------------------


class _Champion:
    version = "V01"
    dataset = {"first": "2015-01-02", "last": "2020-06-30", "ticker": "SYNTH"}


class _Store:
    root = "/tmp/demo-store"


def test_demo_passes_every_option_to_the_commands_it_calls(monkeypatch, tmp_path):
    """A Typer command called as a function passes ``OptionInfo`` unless told otherwise."""
    from cli import research as cli_research

    calls: dict[str, dict] = {}
    signatures = {name: inspect.signature(getattr(cli_research, name))
                  for name in ("init", "run", "paper", "serve")}
    state: dict[str, object] = {"champion": None}

    def spy(name):
        def record(**kwargs):
            assert not [key for key, value in kwargs.items()
                        if isinstance(value, typer.models.OptionInfo)]
            calls[name] = kwargs
        return record

    def registry(store):
        return SimpleNamespace(champion_version=lambda: state["champion"])

    monkeypatch.setattr(cli_research, "_store", lambda research_dir: _Store())
    monkeypatch.setattr(cli_research, "StrategyRegistry", registry)
    monkeypatch.setattr(cli_research, "paper_history", lambda store, *args, **kwargs: [])
    for name in ("init", "run", "paper", "serve"):
        monkeypatch.setattr(cli_research, name, spy(name))
    monkeypatch.setattr(demo_module, "write_demo_csvs", lambda directory, force=False: {})

    # A fresh store: no champion, so the demo must run the loop.
    cli_research.demo(directory=str(tmp_path), port=32909, host="127.0.0.1", hypotheses=1,
                      rebuild=False, store_only=False)
    assert "run" in calls, "a fresh demo did not run the hypothesis loop"
    assert calls["run"]["ticker"] == "SYNTH"
    assert calls["run"]["end"] == DEMO_WINDOW["research_end"]
    assert set(signatures["run"].parameters) <= set(calls["run"])
    # The loop seeds V01 with the window it measured on; `init` would seed one without
    # a window, and such a baseline cannot be paper traded afterwards.
    assert "init" not in calls

    # A store that already has a champion with a window: paper, then serve.
    state["champion"] = _Champion()
    calls.clear()
    cli_research.demo(directory=str(tmp_path), port=32909, host="127.0.0.1", hypotheses=1,
                      rebuild=False, store_only=False)
    for name in ("paper", "serve"):
        assert name in calls, f"{name} was never called"
        missing = set(signatures[name].parameters) - set(calls[name])
        assert not missing, f"{name} was called without {sorted(missing)}"
    assert calls["paper"]["end"] == DEMO_WINDOW["paper_end"]
    assert calls["serve"]["ticker"] == "SYNTH"
    assert calls["serve"]["feed"] == ["replay:ALPHA", "replay:BETA", "replay:GAMMA"]


def test_demo_store_only_does_not_start_a_server(monkeypatch, tmp_path):
    from cli import research as cli_research

    started: list[str] = []
    monkeypatch.setattr(cli_research, "_store", lambda research_dir: _Store())
    monkeypatch.setattr(cli_research, "StrategyRegistry",
                        lambda store: type("R", (), {"champion_version": lambda self: _Champion()})())
    monkeypatch.setattr(cli_research, "paper_history", lambda store, *args, **kwargs: [{}])
    monkeypatch.setattr(cli_research, "serve", lambda **kwargs: started.append("serve"))
    monkeypatch.setattr(demo_module, "write_demo_csvs", lambda directory, force=False: {})

    cli_research.demo(directory=str(tmp_path), port=32909, host="127.0.0.1", hypotheses=1,
                      rebuild=False, store_only=True)
    assert not started


def test_one_command_end_to_end(tmp_path, capsys):
    """The acceptance criterion: a fresh directory, one call, a paper-verified version."""
    from cli import research as cli_research
    from tradingagents.research.store import ResearchStore

    cli_research.demo(directory=str(tmp_path), port=32909, host="127.0.0.1", hypotheses=1,
                      rebuild=False, store_only=True)

    store = ResearchStore(root=tmp_path / "store")
    sessions = store.read("paper")
    assert sessions, "the demo did not record a paper session"
    assert sessions[-1]["trades"] > 0
    assert sessions[-1]["status"] in ("passed", "failed", "pending")

    output = capsys.readouterr().out
    assert "synthetic" in output.lower()
    assert "research serve" in output

    # Evidence, not a promise: the demo cannot enable live trading for the user.
    from tradingagents.research.versions import StrategyRegistry

    assert StrategyRegistry(store).live_trading_enabled() is False
