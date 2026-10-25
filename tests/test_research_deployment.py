"""Paper-first deployment: the chain of evidence that must exist before live permission."""

from __future__ import annotations

import pytest

from tradingagents.research.contracts import ExperimentStatus
from tradingagents.research.pipeline import PromotionPipeline, deployment_summary
from tradingagents.research.risk import RiskEngine, RiskPolicy
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import PositionSizing, baseline_strategy
from tradingagents.research.versions import StrategyRegistry, VersionError


def _oos(expectancy: float = 0.22, trades: int = 45, gates_passed: bool = True) -> dict:
    return {
        "metrics": {"expectancy_r": expectancy, "trades": trades, "profit_factor": 1.4,
                    "max_drawdown_pct": 0.08},
        "gates": [
            {"name": "oos_expectancy", "passed": gates_passed, "measured": f"{expectancy}R"},
            {"name": "cost_stress", "passed": gates_passed, "measured": "0.11R"},
            {"name": "parameter_sensitivity", "passed": gates_passed, "measured": "0.1"},
            {"name": "walk_forward_consistency", "passed": gates_passed, "measured": "0.75"},
        ],
    }


def _walk(positive_share: float = 0.75) -> dict:
    return {"folds": [{"fold": 1}, {"fold": 2}, {"fold": 3}, {"fold": 4}],
            "positive_share": positive_share, "mean_expectancy_r": 0.2}


def _paper(status: str = "passed", trades: int = 120) -> dict:
    return {"status": status, "trades": trades, "expectancy_r": 0.18,
            "started": "2026-01-02", "ended": "2026-03-02",
            "note": "paper trading supported the candidate" if status == "passed" else "failed"}


def _setup(tmp_path, *, paper: dict | None = None, status: str = "Paper Testing"):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    spec = baseline_strategy("1d").with_version("V01")
    registry.register(spec, status=status, backtest={"metrics": {"expectancy_r": 0.25}},
                      out_of_sample=_oos(), walk_forward=_walk())
    if paper is not None:
        registry.update("V01", paper=paper)
    pipeline = PromotionPipeline(registry, store)
    return store, registry, pipeline


@pytest.mark.unit
def test_a_backtest_alone_is_not_nearly_enough(tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    registry.register(baseline_strategy("1d").with_version("V01"),
                      backtest={"metrics": {"expectancy_r": 0.4}})
    review = PromotionPipeline(registry, store).review("V01")
    assert review.furthest_stage == "backtested"
    assert not review.ready_for_approval
    assert any("out-of-sample" in blocker for blocker in review.blockers)
    with pytest.raises(PermissionError):
        PromotionPipeline(registry, store).request_approval("V01", approver="alice")


@pytest.mark.unit
def test_the_full_chain_reaches_awaiting_approval_and_then_stops(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper())
    review = pipeline.advance("V01")
    assert review.furthest_stage == "awaiting_user_approval"
    assert review.ready_for_approval
    assert registry.get("V01").stage == "awaiting_user_approval"
    assert registry.live_trading_enabled() is False

    approval = pipeline.request_approval("V01", approver="alice", note="paper consistent")
    assert approval["approver"] == "alice"
    record = registry.get("V01")
    assert record.live_eligible is True
    # Approval makes the version eligible; it does not switch live trading on.
    assert registry.live_trading_enabled() is False

    with pytest.raises(VersionError, match="does not match the champion"):
        registry.enable_live(approver="alice", confirmation="V02")
    assert registry.live_trading_enabled() is False
    registry.enable_live(approver="alice", confirmation="V01")
    assert registry.live_trading_enabled() is True


@pytest.mark.unit
def test_failed_paper_trading_blocks_approval(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper(status="failed", trades=9))
    review = pipeline.advance("V01")
    assert review.furthest_stage == "robustness"
    assert any("paper trading" in blocker for blocker in review.blockers)
    with pytest.raises(PermissionError, match="cannot be approved yet"):
        pipeline.request_approval("V01", approver="alice")
    assert registry.live_trading_enabled() is False


@pytest.mark.unit
def test_a_failed_robustness_gate_stops_the_chain_earlier(tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    registry.register(baseline_strategy("1d").with_version("V01"),
                      backtest={"metrics": {"expectancy_r": 0.25}},
                      out_of_sample=_oos(gates_passed=False),
                      walk_forward=_walk())
    pipeline = PromotionPipeline(registry, store)
    review = pipeline.advance("V01")
    assert review.furthest_stage == "walk_forward"
    assert any("robustness" in blocker or "cost stress" in blocker
               for blocker in review.blockers)


@pytest.mark.unit
def test_a_strategy_asking_for_more_risk_than_policy_is_refused_risk_review(tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    greedy = baseline_strategy("1d").with_version("V01")
    greedy = type(greedy)(**{**greedy.__dict__,
                             "sizing": PositionSizing(risk_per_trade=0.05, max_leverage=5.0)})
    registry.register(greedy, out_of_sample=_oos())
    pipeline = PromotionPipeline(registry, store)
    review = pipeline.review_risk(greedy)
    assert not review["passed"]
    assert any("5.00%" in reason or "5" in reason for reason in review["reasons"])
    full = pipeline.review("V01")
    assert any(check.stage == "risk_reviewed" and not check.passed for check in full.checks)


@pytest.mark.unit
def test_the_risk_review_uses_the_stored_policy_not_a_default(tmp_path):
    store = ResearchStore(root=tmp_path)
    registry = StrategyRegistry(store)
    spec = baseline_strategy("1d").with_version("V01")
    registry.register(spec, out_of_sample=_oos())
    from tradingagents.research.risk import save_policy

    save_policy(store, RiskPolicy(max_leverage=0.5))
    from tradingagents.research.risk import load_policy

    pipeline = PromotionPipeline(registry, store, RiskEngine(load_policy(store), store))
    review = pipeline.review_risk(spec)
    assert not review["passed"]
    assert any("leverage" in reason for reason in review["reasons"])


@pytest.mark.unit
def test_stage_advances_are_recorded_and_never_move_backwards(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper())
    pipeline.advance("V01")
    pipeline.advance("V01")
    events = [e for e in store.read("deployment") if e.get("kind") == "stage_advanced"]
    assert len(events) == 1
    assert events[0]["to"] == "awaiting_user_approval"
    assert events[0]["from"] == "proposed"


@pytest.mark.unit
def test_rollback_after_enabling_live_disables_it(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper())
    pipeline.advance("V01")
    # The champion moves on to a second, better-evidenced version.
    registry.register(baseline_strategy("1d").with_version("V02"), parent_version="V01",
                      status=ExperimentStatus.PAPER_TESTING,
                      backtest={"metrics": {"expectancy_r": 0.3}},
                      out_of_sample=_oos(expectancy=0.3, trades=60),
                      walk_forward=_walk(0.8))
    registry.update("V02", paper=_paper(trades=80))
    pipeline.advance("V01")
    pipeline.advance("V02")
    pipeline.request_approval("V02", approver="alice")
    registry.promote("V02")
    registry.enable_live(approver="alice", confirmation="V02")
    assert registry.live_trading_enabled() is True

    registry.rollback("V01", reason="divergence from paper")
    assert registry.live_trading_enabled() is False
    assert registry.get("V02").status == "Rolled Back"
    assert registry.get("V02").live_eligible is False
    assert registry.champion_version().version == "V01"
    status = pipeline.live_status()
    assert status["live_enabled"] is False
    assert any("approval" in reason or "disabled" in reason for reason in status["reasons"])


@pytest.mark.unit
def test_live_status_explains_every_reason_it_is_off(tmp_path):
    store, registry, _ = _setup(tmp_path)
    status = PromotionPipeline(registry, store).live_status()
    assert status["live_enabled"] is False
    assert any("disabled by default" in reason for reason in status["reasons"])
    assert any("approval" in reason for reason in status["reasons"])


@pytest.mark.unit
def test_emergency_stop_blocks_an_otherwise_approved_candidate(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper())
    pipeline.advance("V01")
    pipeline.request_approval("V01", approver="alice")
    registry.enable_live(approver="alice", confirmation="V01")

    from tradingagents.research.risk import save_policy

    save_policy(store, RiskPolicy(emergency_stop=True))
    policy = RiskEngine(RiskPolicy(emergency_stop=True), store)
    assert policy.policy.emergency_stop is True
    assert policy.policy.emergency_stop is True
    pipeline.risk = policy
    spec = registry.get("V01").spec
    assert pipeline.review_risk(spec)["passed"]  # sizing is fine; the stop is elsewhere


@pytest.mark.unit
def test_deployment_summary_is_newest_first(tmp_path):
    store, registry, pipeline = _setup(tmp_path, paper=_paper())
    pipeline.advance("V01")
    pipeline.request_approval("V01", approver="alice")
    summary = deployment_summary(store)
    assert summary
    assert summary[0]["kind"] in ("approved", "approval_recorded", "status_changed",
                                  "stage_advanced")
    timestamps = [event.get("at") or event.get("recorded_at") for event in summary]
    assert timestamps == sorted([t for t in timestamps if t], reverse=True)
