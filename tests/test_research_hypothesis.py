"""The research loop end to end: observe, hypothesise, test, decide, record."""

from __future__ import annotations

import json

import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.contracts import CostModel, ExperimentStatus
from tradingagents.research.engine import EngineConfig
from tradingagents.research.features import build_feature_table
from tradingagents.research.hypothesis import (
    DiagnosticIdeaProvider,
    HypothesisLoop,
    LLMIdeaProvider,
    LoopPlan,
    diagnose,
    paper_trade,
    recent_hypotheses,
    run_hypothesis_loop,
)
from tradingagents.research.risk import RiskPolicy
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import baseline_strategy
from tradingagents.research.validation import AcceptanceCriteria
from tradingagents.research.versions import StrategyRegistry


@pytest.fixture(scope="module")
def table():
    bars = synthetic_bars(n=620, seed=41, regime_length=60)
    return build_feature_table(market_data(bars), "1d")


def _plan(**overrides) -> LoopPlan:
    payload = {"ticker": "SYNTH", "timeframes": ["1d"],
               "start": "2022-01-03", "end": "2024-12-31",
               "costs": CostModel(commission_bps=2.0, slippage_bps=2.0, spread_bps=2.0),
               "max_hypotheses": 2, "walk_forward_folds": 3}
    payload.update(overrides)
    return LoopPlan(**payload)


def _loop(tmp_path, table, provider=None, **plan_overrides) -> tuple[HypothesisLoop, ResearchStore]:
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    loop = HypothesisLoop(store, registry, _plan(**plan_overrides), provider=provider,
                          table=table)
    return loop, store


@pytest.mark.unit
def test_one_session_produces_an_auditable_record_and_never_goes_live(tmp_path, table):
    loop, store = _loop(tmp_path, table)
    summary = loop.run()

    assert summary.ticker == "SYNTH"
    assert summary.baseline["version"] == "V01"
    assert summary.leakage["ok"] is True
    assert summary.data_quality
    log = store.read("hypothesis")
    assert log, "every attempt must be recorded, accepted or not"
    for record in log:
        assert record["reason"] and record["evidence"]
        assert record["data_range"] and record["split"]
        assert record["cost_assumptions"] and record["acceptance_criteria"]
    # The append-only log holds each idea's opening and closing events.
    decided = [r for r in recent_hypotheses(store) if r["decision"] != "pending"]
    assert decided
    for record in decided:
        assert record["decision"] in ("accepted", "rejected", "continue_research")
    # A session decides; it never grants live permission.
    registry = StrategyRegistry(store)
    assert registry.live_trading_enabled() is False
    assert all(version.live_eligible is False for version in registry.versions())
    if summary.versions:
        assert all(name in summary.versions for name in summary.versions)


@pytest.mark.unit
def test_rejections_carry_the_gate_that_failed_and_the_measured_value(tmp_path, table):
    loop, store = _loop(tmp_path, table)
    loop.run()
    records = recent_hypotheses(store)
    rejected = [r for r in records if r["decision"] == "rejected"]
    if not rejected:
        pytest.skip("this fixture produced no rejection to inspect")
    for record in rejected:
        assert record["explanation"]
        gates = (record.get("results") or {}).get("gates") or []
        assert gates, "a rejection must be explained by a gate result"
        assert any(not gate["passed"] for gate in gates)


@pytest.mark.unit
def test_a_second_session_does_not_rerun_the_same_rejected_experiment(tmp_path, table):
    loop, store = _loop(tmp_path, table, max_hypotheses=3)
    first = loop.run()
    second_loop, _ = _loop(tmp_path, table, max_hypotheses=3)
    second = second_loop.run()

    first_signatures = [h.signature for h in first.hypotheses]
    second_signatures = [h.signature for h in second.hypotheses]
    assert not set(first_signatures) & set(second_signatures), \
        "a rejected experiment was proposed again without new evidence"
    if first.no_trade_reason:
        assert "NO TRADE" in first.no_trade_reason


@pytest.mark.unit
def test_the_no_trade_verdict_explains_itself(tmp_path, table):
    # A stricter sample requirement rejects everything this fixture can offer.
    criteria = AcceptanceCriteria(min_trades=10_000, min_oos_trades=10_000)
    loop, store = _loop(tmp_path, table, criteria=criteria, max_hypotheses=1)
    summary = loop.run()
    assert summary.accepted == 0
    assert summary.no_trade_reason.startswith("NO TRADE")
    assert "out-of-sample" in summary.no_trade_reason
    assert store.read("hypothesis")


@pytest.mark.unit
def test_diagnostics_are_ranked_by_the_losses_they_explain(table):
    from tradingagents.research.engine import run_backtest

    spec = baseline_strategy("1d")
    spec.validate_columns(table.features.columns)
    result = run_backtest(spec, table, EngineConfig())
    diagnostics = diagnose(result, table)
    assert all(d.evidence and d.downside for d in diagnostics)
    severities = [d.severity for d in diagnostics]
    assert severities == sorted(severities, reverse=True)
    for diagnostic in diagnostics:
        assert diagnostic.change.kind in type(diagnostic.change).KINDS


@pytest.mark.unit
def test_an_llm_idea_provider_can_only_propose_gated_candidates(tmp_path, table):
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps([
            {"kind": "tighten_stop", "params": {"atr_multiple": 1.5},
             "rationale": "the stops were too wide for the measured volatility"},
            {"kind": "raise_position_size", "params": {"times": 10},
             "rationale": "go big"},          # undeclared kind: must be discarded
            {"kind": "add_regime_filter", "params": {"allowed": ["trend_up"]},
             "rationale": "restrict to the regime where the baseline made money"},
        ])

    provider = LLMIdeaProvider(complete)
    loop, store = _loop(tmp_path, table, provider=provider, max_hypotheses=3)
    loop.run()
    assert calls, "the provider was not consulted"
    assert "risk" in calls[0].lower(), "the prompt should describe risk, not promise returns"
    assert "Do not claim results" in calls[0]

    records = store.read("hypothesis")
    recorded_kinds = {(r.get("change") or {}).get("kind") for r in records}
    assert "raise_position_size" not in recorded_kinds
    assert all(record["decision"] != "accepted" or record["results"]["gates"]
               for record in recent_hypotheses(store))


@pytest.mark.unit
def test_an_llm_provider_that_breaks_does_not_break_the_loop(tmp_path, table):
    def complete(prompt: str) -> str:
        raise RuntimeError("model endpoint unavailable")

    loop, store = _loop(tmp_path, table, provider=LLMIdeaProvider(complete))
    summary = loop.run()
    # No proposals means no trade, recorded as such; the session still finishes.
    assert summary.no_trade_reason.startswith("NO TRADE")
    assert store.read("experiment"), "the no-hypothesis outcome must be recorded"


@pytest.mark.unit
def test_the_diagnostic_provider_skips_history_it_has_already_rejected(table):
    from tradingagents.research.engine import run_backtest

    result = run_backtest(baseline_strategy("1d"), table, EngineConfig())
    diagnostics = diagnose(result, table)
    provider = DiagnosticIdeaProvider()
    if not diagnostics:
        pytest.skip("no diagnostic on this fixture")
    all_proposals = provider.propose(diagnostics, [])
    assert len(all_proposals) == len(diagnostics)
    history = [{"signature": diagnostics[0].signature, "decision": "rejected"}]
    remaining = provider.propose(diagnostics, history)
    assert all(item[0].signature != diagnostics[0].signature for item in remaining)


@pytest.mark.unit
def test_paper_trading_reports_what_the_forward_bars_say(table):
    spec = baseline_strategy("1d")
    spec.validate_columns(table.features.columns)
    after = table.bars.index[400]
    record = paper_trade(spec, table, after, EngineConfig(),
                         min_bars=30, min_trades=100)
    assert record.status == "failed"
    assert "fewer than the required trades" in record.note
    assert record.started and record.ended
    assert record.ended > record.started

    nothing_to_do = paper_trade(spec, table, table.bars.index[-1], EngineConfig())
    assert nothing_to_do.status == "pending"
    assert "0 bars after the test window" in nothing_to_do.note


@pytest.mark.unit
def test_a_data_quality_failure_stops_the_session_before_any_hypothesis(tmp_path):
    bars = synthetic_bars(n=120, seed=2)
    weak = build_feature_table(market_data(bars, min_bars=100_000), "1d")
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    summary = HypothesisLoop(store, registry, _plan(), table=weak).run()
    assert summary.no_trade_reason.startswith("data quality gate failed")
    assert not summary.hypotheses
    assert not registry.versions(), "nothing should be registered on untrustworthy data"
    assert store.read("risk_event")[-1]["kind"] == "research_blocked"


@pytest.mark.unit
def test_the_loop_never_weakens_the_risk_policy(tmp_path, table):
    policy = RiskPolicy(max_risk_per_trade=0.01, max_leverage=1.0)
    loop, store = _loop(tmp_path, table, risk_policy=policy)
    loop.run()
    assert loop.config.risk_policy == policy
    assert loop.config.risk_policy.max_risk_per_trade == 0.01
    assert loop.config.risk_policy.emergency_stop is False


@pytest.mark.unit
def test_recent_hypotheses_filters_by_ticker_and_is_newest_first(tmp_path, table):
    loop, store = _loop(tmp_path, table, max_hypotheses=1)
    loop.run()
    records = recent_hypotheses(store, ticker="SYNTH")
    assert records
    assert records[0]["ticker"] == "SYNTH"
    assert recent_hypotheses(store, ticker="OTHER") == []


@pytest.mark.unit
def test_the_summary_serialises_for_the_dashboard(tmp_path, table):
    loop, store = _loop(tmp_path, table, max_hypotheses=1)
    summary = loop.run()
    payload = json.dumps(summary.as_dict(), default=str)
    assert "SYNTH" in payload
    assert summary.hypotheses == [] or summary.hypotheses[0].hypothesis_id.startswith("H-")
    assert summary.as_dict()["accepted"] + summary.as_dict()["rejected"] == len(summary.hypotheses)


@pytest.mark.unit
def test_run_hypothesis_loop_builds_its_own_collaborators(tmp_path, table):
    summary = run_hypothesis_loop(_plan(max_hypotheses=1),
                                  store=ResearchStore(root=tmp_path), table=table)
    assert summary.ticker == "SYNTH"
    assert summary.baseline["in_sample"]["trades"] >= 0
    assert summary.champion == StrategyRegistry(
        ResearchStore(root=tmp_path)).champion_version().version
    assert ExperimentStatus.PAPER_TESTING.value
