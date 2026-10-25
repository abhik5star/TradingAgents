"""Versions: immutability, comparison, promotion, approval and rollback."""

from __future__ import annotations

import pytest

from tradingagents.research.contracts import ExperimentStatus
from tradingagents.research.pipeline import PromotionPipeline
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import (
    RuleChange,
    apply_rule_change,
    baseline_strategy,
)
from tradingagents.research.versions import StrategyRegistry, VersionError, compare_specs


@pytest.fixture()
def registry(tmp_path):
    store = ResearchStore(root=tmp_path)
    return store, StrategyRegistry(store)


def _register(registry, spec, **kwargs):
    return registry.register(spec, **kwargs)


@pytest.mark.unit
def test_versions_are_immutable_and_numbered_in_sequence(registry):
    store, reg = registry
    first = _register(reg, baseline_strategy("1d").with_version("V01"))
    assert reg.next_version_name() == "V02"
    second = _register(reg, baseline_strategy("1d").with_version("V02"),
                       parent_version=first.version)
    assert [v.version for v in reg.versions()] == ["V01", "V02"]
    with pytest.raises(VersionError, match="immutable"):
        _register(reg, baseline_strategy("1d").with_version("V01"))
    assert second.parent_version == "V01"


@pytest.mark.unit
def test_measured_results_cannot_be_overwritten(registry):
    store, reg = registry
    record = _register(reg, baseline_strategy("1d").with_version("V01"),
                       backtest={"metrics": {"expectancy_r": 0.1}},
                       out_of_sample={"metrics": {"expectancy_r": 0.05}})
    with pytest.raises(VersionError, match="immutable field"):
        reg.update(record.version, out_of_sample={"metrics": {"expectancy_r": 9.9}})
    with pytest.raises(VersionError, match="immutable field"):
        reg.update(record.version, backtest={})
    # Status, paper records and stage may still move.
    reg.update(record.version, paper={"status": "passed"})
    assert reg.get("V01").paper["status"] == "passed"
    assert reg.get("V01").backtest["metrics"]["expectancy_r"] == 0.1


@pytest.mark.unit
def test_status_history_records_each_transition(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"))
    reg.set_status("V01", ExperimentStatus.VALIDATION, reason="testing", by="tester")
    record = reg.get("V01")
    assert record.status == "Validation"
    assert [entry["status"] for entry in record.status_history][-1] == "Validation"
    assert record.status_history[-1]["by"] == "tester"
    assert store.read("deployment")[-1]["kind"] == "status_changed"


@pytest.mark.unit
def test_promotion_requires_an_accepted_version(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"))
    _register(reg, baseline_strategy("1d").with_version("V02"))
    with pytest.raises(VersionError, match="only an accepted or paper-tested"):
        reg.promote("V02")
    reg.set_status("V02", ExperimentStatus.PAPER_TESTING)
    assert reg.promote("V02").version == "V02"
    assert reg.champion_version().version == "V02"


@pytest.mark.unit
def test_rollback_restores_the_previous_version_and_keeps_its_results(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"),
              backtest={"metrics": {"expectancy_r": 0.05}})
    _register(reg, baseline_strategy("1d").with_version("V02"),
              backtest={"metrics": {"expectancy_r": 0.30}})
    reg.set_status("V02", ExperimentStatus.APPROVED)
    reg.promote("V02")
    reg.rollback("V01", reason="live results diverged")
    assert reg.champion_version().version == "V01"
    rolled = reg.get("V02")
    assert rolled.status == "Rolled Back"
    assert rolled.live_eligible is False
    # The measured numbers survive the rollback untouched.
    assert rolled.backtest["metrics"]["expectancy_r"] == 0.30
    assert any(event["kind"] == "rolled_back" for event in store.read("deployment"))


@pytest.mark.unit
def test_approval_requires_completed_paper_testing_and_a_named_user(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"))
    with pytest.raises(VersionError, match="must complete paper testing"):
        reg.approve("V01", approver="alice")
    reg.set_status("V01", ExperimentStatus.PAPER_TESTING)
    with pytest.raises(VersionError, match="must name who gave it"):
        reg.approve("V01", approver="  ")
    approval = reg.approve("V01", approver="alice", note="paper looked consistent",
                           paper_trades=120)
    record = reg.get("V01")
    assert approval["approval_id"].startswith("APR-")
    assert record.status == "Approved"
    assert record.live_eligible is True
    assert record.approval["approver"] == "alice"


@pytest.mark.unit
def test_live_trading_stays_disabled_without_approval_and_confirmation(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"))
    assert reg.live_trading_enabled() is False
    with pytest.raises(VersionError, match="no recorded user approval"):
        reg.enable_live(approver="alice", confirmation="V01")
    reg.set_status("V01", ExperimentStatus.PAPER_TESTING)
    reg.approve("V01", approver="alice")
    with pytest.raises(VersionError, match="does not match the champion"):
        reg.enable_live(approver="alice", confirmation="V99")
    result = reg.enable_live(approver="alice", confirmation="v01")
    assert result["live_enabled"] is True
    assert reg.live_trading_enabled() is True
    reg.disable_live(by="alice", reason="done")
    assert reg.live_trading_enabled() is False


@pytest.mark.unit
def test_comparison_states_rule_changes_and_metric_deltas(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"),
              backtest={"metrics": {"expectancy_r": 0.1, "trades": 100}})
    candidate = apply_rule_change(
        baseline_strategy("1d").with_version("V02"),
        RuleChange("add_filter", {"condition": {"left": "regime_adx", "op": ">=", "right": 20.0,
                                                "label": "ADX at least 20"}}))
    _register(reg, candidate, backtest={"metrics": {"expectancy_r": 0.25, "trades": 70}})
    comparison = reg.compare("V01", "V02")
    assert any("ADX at least 20" in change for change in comparison["rule_changes"])
    assert comparison["metrics"]["expectancy_r"]["backtest"]["difference"] == pytest.approx(0.15)
    assert comparison["signature"]["baseline"] != comparison["signature"]["candidate"]


@pytest.mark.unit
def test_compare_specs_reports_exit_and_regime_differences():
    baseline = baseline_strategy("1d")
    candidate = apply_rule_change(baseline, RuleChange("add_trailing_stop",
                                                       {"trailing_atr": 1.0}))
    changes = compare_specs(baseline, candidate)
    assert any("exits:" in change for change in changes)
    # Neither spec restricts regimes, so no regime change should be claimed.
    assert not any(change.startswith("regimes:") for change in changes)


@pytest.mark.unit
def test_pipeline_refuses_approval_before_paper_testing(registry):
    store, reg = registry
    _register(reg, baseline_strategy("1d").with_version("V01"),
              backtest={"metrics": {"expectancy_r": 0.1}},
              out_of_sample={"metrics": {"expectancy_r": 0.05, "trades": 40}})
    pipeline = PromotionPipeline(reg, store)
    review = pipeline.review("V01")
    assert not review.ready_for_approval
    assert review.blockers
    with pytest.raises(PermissionError):
        pipeline.request_approval("V01", approver="alice")


@pytest.mark.unit
def test_pipeline_advances_only_through_passing_stages(registry):
    store, reg = registry
    record = _register(reg, baseline_strategy("1d").with_version("V01"),
                       backtest={"metrics": {"expectancy_r": 0.1}},
                       out_of_sample={"metrics": {"expectancy_r": 0.05, "trades": 40},
                                      "gates": [{"name": "cost_stress", "passed": True},
                                                {"name": "parameter_sensitivity",
                                                 "passed": True}]},
                       walk_forward={"folds": [{"fold": 1}], "positive_share": 0.6})
    pipeline = PromotionPipeline(reg, store)
    review = pipeline.advance("V01")
    # Paper testing has not happened, so that is exactly where the path stops.
    assert review.furthest_stage == "robustness"
    assert reg.get("V01").stage == "robustness"
    assert record.paper == {}
    assert review.blockers and "paper trading" in review.blockers[0]
