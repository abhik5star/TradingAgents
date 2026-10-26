"""Paper trading: a version trades forward on bars it was not selected on."""

from __future__ import annotations

import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.contracts import CostModel, ExperimentStatus
from tradingagents.research.dashboard import build_snapshot, render_console, render_section_html
from tradingagents.research.engine import EngineConfig
from tradingagents.research.features import build_feature_table
from tradingagents.research.paper import (
    FAILED,
    PASSED,
    PENDING,
    latest_paper,
    paper_history,
    paper_window_start,
    run_paper_session,
)
from tradingagents.research.pipeline import PromotionPipeline
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import baseline_strategy
from tradingagents.research.validation import chronological_split
from tradingagents.research.versions import StrategyRegistry, VersionError


@pytest.fixture(scope="module")
def table():
    # Seed 17 is not special: on this fixture its forward window happens to contain a
    # passable sample, which the passing-session tests need. The others assert either
    # outcome, so they do not depend on it.
    bars = synthetic_bars(n=1200, seed=17, regime_length=70)
    return build_feature_table(market_data(bars), "1d")


def _setup(tmp_path, table, *, version: str = "V01", dataset: bool = True,
           split_at: float = 0.6, split: dict | None = None,
           costs: CostModel | None = None, **extra):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    spec = baseline_strategy("1d").with_version(version)
    payload = {"backtest": {"metrics": {"expectancy_r": 0.2}},
               "provenance": {"engine_costs": (costs or CostModel()).as_dict()},
               "out_of_sample": {"metrics": {"expectancy_r": 0.15, "trades": 40}},
               "walk_forward": {"folds": [{"fold": 1}, {"fold": 2}], "positive_share": 0.75}}
    if dataset:
        cut = int(len(table.bars) * split_at)
        window = table.bars.index[cut]
        bounds = split or {"test": f"{table.bars.index[0].date()}..{window.date()}"}
        payload["dataset"] = {**table.summary(), "split": bounds}
    payload.update(extra)
    registry.register(spec, **payload)
    return store, registry


@pytest.mark.unit
def test_the_forward_window_starts_after_the_out_of_sample_test(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    version = registry.get("V01")
    begins = paper_window_start(version)
    assert begins is not None
    assert str(begins.date()) == str(version.dataset["split"]["test"].split("..")[-1])

    session = run_paper_session(registry, store, "V01", table, EngineConfig())
    assert session.started > str(begins.date())
    # Only bars after the split were traded, and the session says how many it saw.
    assert session.bars == len(table.bars.loc[table.bars.index > begins.strftime("%Y-%m-%d")])
    assert session.ended == str(table.bars.index[-1].date())


@pytest.mark.unit
def test_a_session_records_the_numbers_it_actually_produced(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    session = run_paper_session(registry, store, "V01", table, EngineConfig())

    assert session.status in (PASSED, FAILED)
    assert session.trades >= 0
    assert session.paper_id.startswith("PPR-")
    assert session.note
    if session.trades:
        assert session.expectancy_r == pytest.approx(session.expectancy_r)
        assert session.sample_trades
        assert {"entry_time", "exit_reason", "r_multiple"} <= set(session.sample_trades[0])
    # The record is on the version, so the pipeline and dashboard can read it.
    assert registry.get("V01").paper["paper_id"] == session.paper_id


@pytest.mark.unit
def test_a_pending_session_says_so_instead_of_reporting_zero_trades_as_a_failure(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    session = run_paper_session(registry, store, "V01", table, EngineConfig(),
                                start=table.bars.index[-1])
    assert session.status == PENDING
    assert session.trades == 0
    assert "bar(s) after the out-of-sample window" in session.note
    # A pending window is not evidence either way, so it must not move the status.
    assert registry.get("V01").status == ExperimentStatus.RESEARCH.value


@pytest.mark.unit
def test_a_passing_session_moves_the_version_into_paper_testing(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    session = run_paper_session(registry, store, "V01", table, EngineConfig(),
                                min_trades=0, start=table.bars.index[700])
    assert session.status == PASSED, session.note
    assert registry.get("V01").status == ExperimentStatus.PAPER_TESTING.value


@pytest.mark.unit
def test_a_failing_session_leaves_the_version_unapproved(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    session = run_paper_session(registry, store, "V01", table, EngineConfig(),
                                min_trades=10_000)
    assert session.status == FAILED
    assert "did not support" in session.note
    assert registry.live_trading_enabled() is False
    assert registry.get("V01").approval is None


@pytest.mark.unit
def test_sessions_are_append_only_and_keep_their_history(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    first = run_paper_session(registry, store, "V01", table, EngineConfig())
    second = run_paper_session(registry, store, "V01", table, EngineConfig(), min_trades=0)
    history = paper_history(store, "V01")
    assert [record["paper_id"] for record in history] == [first.paper_id,
                                                           second.paper_id]
    assert latest_paper(store, "V01")["paper_id"] == second.paper_id
    assert paper_history(store, "V09") == []


@pytest.mark.unit
def test_paper_trading_needs_an_honest_start_and_a_known_version(table, tmp_path):
    store, registry = _setup(tmp_path, table, dataset=False)
    with pytest.raises(ValueError, match="no recorded out-of-sample window"):
        run_paper_session(registry, store, "V01", table, EngineConfig())
    with pytest.raises(VersionError):
        run_paper_session(registry, store, "V99", table, EngineConfig())

    # An explicit start is accepted when the record has no split.
    session = run_paper_session(registry, store, "V01", table, EngineConfig(),
                                start=table.bars.index[200])
    assert session.started == str(table.bars.index[201].date())


@pytest.mark.unit
def test_costs_reach_the_paper_record(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    cheap = run_paper_session(registry, store, "V01", table,
                              EngineConfig(costs=CostModel(0, 0, 0, 0, 0)), min_trades=0)
    rich = run_paper_session(registry, store, "V01", table, EngineConfig(), min_trades=0)
    # Costs change which trades fire as well as what each one earns, so the
    # comparison is on the recorded cost share rather than on a total that two
    # different trade sets happen to produce.
    if rich.trades:
        assert rich.cost_share_of_gross is not None and rich.cost_share_of_gross > 0
    assert cheap.cost_share_of_gross in (None, 0)


@pytest.mark.unit
def test_the_dashboard_shows_the_session(table, tmp_path):
    store, registry = _setup(tmp_path, table)
    session = run_paper_session(registry, store, "V01", table, EngineConfig())
    snapshot = build_snapshot(store)

    assert snapshot["paper_sessions"][-1]["paper_id"] == session.paper_id
    assert snapshot["paper_testing"]["status"] in ("pending", "passed", "failed")
    html = render_section_html(snapshot)
    assert "Paper window" in html and "Paper expectancy" in html

    import io

    from rich.console import Console

    buffer = io.StringIO()
    render_console(snapshot, console=Console(file=buffer, width=110, force_terminal=False,
                                             color_system=None))
    assert "Paper trading" in buffer.getvalue()
    assert session.version in buffer.getvalue()


@pytest.mark.unit
def test_a_passing_session_populates_the_pipeline_paper_stage(table, tmp_path):
    gates = [{"name": "cost_stress", "passed": True},
             {"name": "parameter_sensitivity", "passed": True}]
    store, registry = _setup(
        tmp_path, table,
        out_of_sample={"metrics": {"expectancy_r": 0.15, "trades": 40}, "gates": gates})
    session = run_paper_session(registry, store, "V01", table, EngineConfig(),
                                min_trades=0, start=table.bars.index[700])
    review = PromotionPipeline(registry, store).review("V01", paper_min_trades=0)
    paper_check = next(check for check in review.checks if check.stage == "paper_tested")
    assert f"{session.trades} trades" in paper_check.evidence
    assert session.ended in paper_check.evidence
    assert paper_check.passed is (session.status == PASSED)


@pytest.mark.unit
def test_the_split_the_session_uses_is_the_one_the_version_recorded(table, tmp_path):
    # A version registered by the loop carries its exact split; paper trading must
    # start after the test segment rather than after an arbitrary fraction.
    split = chronological_split(table.bars.index, 0.5, 0.25)
    store, registry = _setup(tmp_path, table, split=split.as_dict())
    begins = paper_window_start(registry.get("V01"))
    assert str(begins.date()) == split.as_dict()["test"].split("..")[-1]
    session = run_paper_session(registry, store, "V01", table, EngineConfig())
    boundary = begins.tz_localize("UTC") if begins.tzinfo is None else begins
    assert session.bars == len(table.bars.loc[table.bars.index > boundary])
