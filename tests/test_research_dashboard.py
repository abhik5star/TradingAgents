"""The Strategy Intelligence dashboard: snapshot, console, HTML and report integration."""

from __future__ import annotations

import io
import re

import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.report_html import render_report
from tradingagents.reporting import write_report_tree
from tradingagents.research.dashboard import (
    build_snapshot,
    publish_snapshot,
    render_console,
    render_section_html,
    write_dashboard_html,
)
from tradingagents.research.features import build_feature_table
from tradingagents.research.hypothesis import HypothesisLoop, LoopPlan, recent_hypotheses
from tradingagents.research.store import ResearchStore
from tradingagents.research.versions import StrategyRegistry

SETTINGS = {"version": "0.7.0", "llm_provider": "openai", "deep_think_llm": "gpt-6-sol",
            "quick_think_llm": "gpt-6-luna", "analysts": ["market"],
            "max_debate_rounds": 1, "max_risk_discuss_rounds": 1,
            "data_vendors": {"core_stock_apis": "yfinance"}}


def _state(**overrides):
    state = {"trade_date": "2026-09-26",
             "market_report": "# Market view\n\nClose | 181.2\n--- | ---\nTrend | up\n",
             "final_trade_decision": "**Rating**: Hold\n\nNo edge today."}
    state.update(overrides)
    return state


@pytest.fixture(scope="module")
def table():
    return build_feature_table(market_data(synthetic_bars(n=560, seed=13)), "1d")


@pytest.fixture()
def populated(tmp_path, table):
    """A store with one real research session in it — no fabricated numbers."""
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    plan = LoopPlan(ticker="SYNTH", timeframes=["1d"], start="2022-01-03",
                    end="2024-12-31", max_hypotheses=1, walk_forward_folds=3)
    summary = HypothesisLoop(store, registry, plan, table=table).run()
    return store, summary


@pytest.mark.unit
def test_an_empty_store_still_renders_an_honest_dashboard(tmp_path):
    store = ResearchStore(root=tmp_path)
    snapshot = publish_snapshot(store)
    assert snapshot["champion"] is None
    assert snapshot["live_enabled"] is False
    assert snapshot["counts"]["experiments_completed"] == 0
    assert snapshot["paper_testing"]["status"] == "not started"
    assert snapshot["no_trade_reason"] == ""
    assert (tmp_path / "snapshots" / "dashboard.json").exists()
    html = render_section_html(snapshot)
    assert "Strategy Intelligence" in html
    assert "live trading" in html.lower()


@pytest.mark.unit
def test_the_snapshot_reports_the_stored_evidence_not_an_estimate(populated):
    store, summary = populated
    snapshot = build_snapshot(store, summary=summary)
    registry = StrategyRegistry(store)
    champion = registry.champion_version()

    assert snapshot["champion"]["version"] == champion.version
    assert snapshot["champion"]["rules"] == champion.spec.describe()
    assert snapshot["counts"]["versions"] == len(registry.versions())
    assert snapshot["counts"]["hypotheses"] == len(recent_hypotheses(store))
    assert snapshot["live_enabled"] is False
    assert snapshot["latest_hypothesis"]["hypothesis_id"].startswith("H-")
    assert snapshot["last_run"]
    assert snapshot["holdout_windows"] == store.load_holdouts()["windows"]

    # Reported metrics come from the stored documents, not from a fresh guess.
    metrics = snapshot["out_of_sample"]["candidate"]
    if metrics:
        assert metrics["expectancy_r"] == pytest.approx(
            (registry.get(champion.version).out_of_sample.get("metrics") or {})
            .get("expectancy_r", metrics["expectancy_r"]))


@pytest.mark.unit
def test_a_rejection_is_visible_as_a_no_trade_reason(tmp_path, table):
    from tradingagents.research.validation import AcceptanceCriteria

    store = ResearchStore(root=tmp_path)
    plan = LoopPlan(ticker="SYNTH", timeframes=["1d"], start="2022-01-03", end="2024-12-31",
                    max_hypotheses=1, walk_forward_folds=3,
                    criteria=AcceptanceCriteria(min_trades=10_000, min_oos_trades=10_000))
    summary = HypothesisLoop(store, StrategyRegistry(store), plan, table=table).run()
    snapshot = build_snapshot(store, summary=summary)
    assert snapshot["no_trade_reason"].startswith("NO TRADE")
    assert "NO TRADE" in render_section_html(snapshot)


@pytest.mark.unit
def test_the_console_view_shows_every_panel(populated):
    from rich.console import Console

    store, summary = populated
    snapshot = build_snapshot(store, summary=summary)
    buffer = io.StringIO()
    console = Console(file=buffer, width=120, record=True, force_terminal=False,
                      color_system=None)
    render_console(snapshot, console=console)
    text = buffer.getvalue()
    for expected in ("Strategy Intelligence", "Research", "Health", "Decision",
                     "live trading", "risk blocks"):
        assert expected.lower() in text.lower()
    assert "V01" in text
    # The console must not print an em dash or an invented win rate.
    assert "win rate" not in text.lower()


@pytest.mark.unit
def test_hostile_text_from_a_record_is_escaped_in_the_html(tmp_path):
    store = ResearchStore(root=tmp_path)
    store.append("hypothesis", {
        "hypothesis_id": "H-evil", "ticker": "X", "title": "<script>alert(1)</script>",
        "decision": "rejected", "status": "Rejected", "reason": "<img src=x onerror=1>",
        "evidence": ["</dd><script>bad()</script>"], "explanation": "<b>not bold</b>",
        "confidence": "n/a", "created_at": "2026-01-01T00:00:00Z"})
    snapshot = build_snapshot(store)
    html = render_section_html(snapshot)
    assert "<script>" not in html.lower()
    assert "&lt;script&gt;" in html
    assert "onerror=1" not in html
    standalone = write_dashboard_html(snapshot, tmp_path / "dash.html")
    with open(standalone, encoding="utf-8") as handle:
        page = handle.read()
    assert "<script>" not in page.lower()
    assert not re.search(r'<(img|link|iframe|object|embed)\b', page, re.IGNORECASE)


@pytest.mark.unit
def test_the_standalone_page_uses_the_existing_report_theme(populated, tmp_path):
    store, summary = populated
    path = write_dashboard_html(build_snapshot(store, summary=summary),
                                tmp_path / "dashboard.html")
    with open(path, encoding="utf-8") as handle:
        page = handle.read()
    assert page.startswith("<!doctype html>")
    assert "Content-Security-Policy" in page
    assert "<style>" in page
    # No remote asset, no fetch, no second design system.
    assert not re.search(r"https?://", page)
    assert "--ink" in page or "var(--" in page


@pytest.mark.unit
def test_the_existing_report_gains_a_strategy_intelligence_section(populated, tmp_path):
    store, summary = populated
    section = render_section_html(build_snapshot(store, summary=summary))
    write_report_tree(_state(), "NVDA", tmp_path / "with", settings=SETTINGS, html=True,
                      research_section=section)
    page = (tmp_path / "with" / "complete_report.html").read_text(encoding="utf-8")
    assert 'id="strategy-intelligence"' in page
    assert 'href="#strategy-intelligence"' in page
    assert "Strategy Intelligence" in page
    # The rest of the report is untouched by the addition.
    assert "Market view" in page and "181.2" in page
    assert "Hold" in page

    write_report_tree(_state(), "NVDA", tmp_path / "without", settings=SETTINGS, html=True)
    plain = (tmp_path / "without" / "complete_report.html").read_text(encoding="utf-8")
    assert "strategy-intelligence" not in plain
    assert "Strategy Intelligence" not in plain


@pytest.mark.unit
def test_render_report_places_the_section_after_the_agent_sections(populated):
    store, summary = populated
    section = render_section_html(build_snapshot(store, summary=summary))
    parts = [("I. Analyst Team", "1_analysts",
              [("Market Analyst", "market_report.md", "text")])]
    page = render_report("NVDA", _state(), SETTINGS, parts, research_section=section)
    assert page.index("Market Analyst") < page.index("Strategy Intelligence")
    assert page.count("<section") >= 2
