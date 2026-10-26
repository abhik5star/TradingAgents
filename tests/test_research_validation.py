"""Validation: splits, walk-forward, look-ahead probing, holdout discipline and gates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.contracts import GateResult, Metrics, TradeRecord, TradeSide
from tradingagents.research.engine import EngineConfig
from tradingagents.research.features import build_feature_table
from tradingagents.research.store import ResearchStore
from tradingagents.research.strategies import baseline_strategy
from tradingagents.research.validation import (
    AcceptanceCriteria,
    HoldoutLedger,
    LeakageReport,
    acceptance_gates,
    chronological_split,
    gates_passed,
    multiple_testing_summary,
    parameter_sensitivity,
    probe_lookahead,
    walk_forward,
)


@pytest.fixture(scope="module")
def table():
    bars = synthetic_bars(n=700, seed=21)
    data = market_data(bars)
    return build_feature_table(data, "1d")


@pytest.mark.unit
def test_split_is_chronological_and_leaves_gaps_between_segments(table):
    split = chronological_split(table.bars.index, 0.5, 0.25, purge_bars=5, embargo_bars=2)
    train = split.subset(table, "train")
    validation = split.subset(table, "validation")
    test = split.subset(table, "test")
    assert train.bars.index[-1] < validation.bars.index[0]
    assert validation.bars.index[-1] < test.bars.index[0]
    overlap = set(train.bars.index) & set(test.bars.index)
    assert not overlap
    gap = (validation.bars.index[0] - train.bars.index[-1]).days
    assert gap >= 1


@pytest.mark.unit
def test_split_rejects_a_sample_too_small_or_a_test_window_too_small():
    import pandas as pd

    with pytest.raises(ValueError, match="at least 60 bars"):
        chronological_split(pd.date_range("2026-01-01", periods=40, freq="1D", tz="UTC"))
    with pytest.raises(ValueError, match="untouched test window"):
        chronological_split(pd.date_range("2026-01-01", periods=400, freq="1D", tz="UTC"),
                            0.8, 0.19)


@pytest.mark.unit
def test_walk_forward_reports_each_fold_and_its_consistency(table):
    result = walk_forward(baseline_strategy("1d"), table, EngineConfig(), folds=4)
    assert result.folds_run >= 3
    assert 0.0 <= result.positive_share <= 1.0
    assert result.consistency_note
    for fold in result.folds:
        assert fold.test_bounds and fold.train_bounds
        assert fold.metrics.trades >= 0


@pytest.mark.unit
def test_walk_forward_refuses_an_impossible_request(table):
    with pytest.raises(ValueError, match="at least two folds"):
        walk_forward(baseline_strategy("1d"), table, EngineConfig(), folds=1)
    with pytest.raises(ValueError, match="cannot fill"):
        walk_forward(baseline_strategy("1d"), table, EngineConfig(), folds=40)


@pytest.mark.unit
def test_lookahead_probe_passes_on_an_honest_feature_table(table):
    report = probe_lookahead(table.data, "1d", probes=2)
    assert report.ok, report.differences[:3]
    assert report.probes >= 1


@pytest.mark.unit
def test_lookahead_probe_catches_a_planted_future_leak(monkeypatch, table):
    import tradingagents.research.validation as validation

    original = validation.build_feature_table

    def leaky(*args, **kwargs):
        built = original(*args, **kwargs)
        # A recipe that peeks one bar ahead is exactly the bug the probe exists for.
        built.features["peek"] = built.bars["close"].shift(-1)
        return built

    monkeypatch.setattr(validation, "build_feature_table", leaky)
    report = probe_lookahead(table.data, "1d", probes=2)
    assert not report.ok
    assert any("peek" in difference for difference in report.differences)


@pytest.mark.unit
def test_holdout_ledger_counts_uses_and_enforces_the_budget(tmp_path):
    store = ResearchStore(root=tmp_path)
    ledger = HoldoutLedger(store)
    assert ledger.uses("fp", "window") == 0
    first, within = ledger.record("fp", "window", "H-1", max_uses=2)
    assert (first, within) == (1, True)
    second, within = ledger.record("fp", "window", "H-2", max_uses=2)
    assert (second, within) == (2, True)
    third, within = ledger.record("fp", "window", "H-3", max_uses=2)
    assert (third, within) == (3, False)
    # A restart does not reset the count.
    assert HoldoutLedger(ResearchStore(root=tmp_path)).uses("fp", "window") == 3


def _trade(r_multiple: float) -> TradeRecord:
    return TradeRecord(
        trade_id=f"T{abs(hash(r_multiple)) % 1000}", strategy_id="s", version="V",
        ticker="X", timeframe="1d", side=TradeSide.LONG, entry_time="2026-01-01",
        exit_time="2026-01-02", entry_price=100.0, exit_price=101.0, quantity=10.0,
        stop_price=99.0, target_price=102.0, gross_pnl=10.0, cost=1.0, net_pnl=9.0,
        r_multiple=r_multiple, bars_held=1, exit_reason="target", regime="trend_up",
        session="london", planned_risk=10.0)


def _result(*, trades: int = 80, expectancy_r: float = 0.25, profit_factor: float = 1.6,
            drawdown: float = 0.10, cost_share: float = 0.20,
            r_multiples: list[float] | None = None) -> SimpleNamespace:
    r_values = r_multiples if r_multiples is not None else [0.4, -0.2] * (trades // 2)
    return SimpleNamespace(
        metrics=Metrics(trades=trades, expectancy_r=expectancy_r, profit_factor=profit_factor,
                        max_drawdown_pct=drawdown, cost_share_of_gross=cost_share),
        trades=[_trade(value) for value in r_values],
    )


def _gates(**overrides):
    candidate = _result()
    baseline = _result(trades=90, expectancy_r=0.10, profit_factor=1.2, drawdown=0.12)
    oos = overrides.pop("oos", _result(r_multiples=[0.8, -0.2] * 40))
    oos_baseline = overrides.pop("oos_baseline", _result(
        trades=60, expectancy_r=0.05, profit_factor=1.1, drawdown=0.12,
        r_multiples=[-0.2, -0.2, 0.4] * 20))
    stress = overrides.pop("stress", _result(expectancy_r=0.10))
    walk = overrides.pop("walk", SimpleNamespace(folds_run=4, positive_share=0.75,
                                                 mean_expectancy_r=0.2,
                                                 consistency_note="positive"))
    sensitivity = overrides.pop("sensitivity", {"fragile": 0.0, "variants": 16,
                                                "profitable_share": 1.0, "spread": 0.1})
    criteria = overrides.pop("criteria", AcceptanceCriteria())
    holdout_uses = overrides.pop("holdout_uses", 1)
    data_quality_ok = overrides.pop("data_quality_ok", True)
    leakage = overrides.pop("leakage", LeakageReport(ok=True, probes=3))
    return acceptance_gates(candidate, baseline, oos=oos, oos_baseline=oos_baseline,
                            stress=stress, walk=walk, sensitivity=sensitivity,
                            criteria=criteria, family_size=1, holdout_uses=holdout_uses,
                            data_quality_ok=data_quality_ok, leakage=leakage, **overrides)


@pytest.mark.unit
def test_all_gates_pass_for_a_genuinely_better_candidate():
    gates = _gates()
    assert gates_passed(gates), [g.name for g in gates if not g.passed]
    names = {gate.name for gate in gates}
    assert {"data_quality", "sample_size", "oos_expectancy", "beats_baseline",
            "drawdown_vs_baseline", "cost_stress", "walk_forward_consistency",
            "parameter_sensitivity", "holdout_budget", "statistical_significance"} <= names


@pytest.mark.unit
def test_thin_samples_and_negative_out_of_sample_expectancy_fail():
    gates = _gates(oos=_result(trades=12, expectancy_r=-0.1, profit_factor=0.8,
                               r_multiples=[-1.0, 0.5] * 6),
                   stress=_result(expectancy_r=-0.05))
    failed = {gate.name for gate in gates if not gate.passed}
    assert {"oos_sample", "oos_expectancy", "beats_baseline", "cost_stress"} <= failed


@pytest.mark.unit
def test_a_bigger_drawdown_fails_even_with_a_better_edge():
    gates = _gates(oos=_result(drawdown=0.40))
    failed = {gate.name for gate in gates if not gate.passed}
    assert "drawdown_vs_baseline" in failed


@pytest.mark.unit
def test_cost_dominated_and_fragile_results_are_refused():
    gates = _gates(oos=_result(cost_share=0.8),
                   sensitivity={"fragile": 0.75, "variants": 16,
                                "profitable_share": 0.4, "spread": 2.0})
    failed = {gate.name for gate in gates if not gate.passed}
    assert {"cost_share", "parameter_sensitivity"} <= failed


@pytest.mark.unit
def test_identical_distributions_are_not_reported_as_an_improvement():
    same = [0.4, -0.2] * 30
    gates = _gates(oos=_result(expectancy_r=0.10, r_multiples=same),
                   oos_baseline=_result(trades=60, expectancy_r=0.10, r_multiples=same))
    failed = {gate.name for gate in gates if not gate.passed}
    assert "statistical_significance" in failed


@pytest.mark.unit
def test_exhausted_holdout_budget_and_leakage_block_acceptance():
    gates = _gates(holdout_uses=9)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "holdout_budget" in failed
    leaked = acceptance_gates(_result(), _result(), oos=_result(), oos_baseline=_result(),
                              stress=None, walk=None, sensitivity=None,
                              criteria=AcceptanceCriteria(), leakage=LeakageReport(
                                  ok=False, probes=3, differences=["close differs"]))
    assert any(gate.name == "no_lookahead" and not gate.passed for gate in leaked)

    bad_data = acceptance_gates(_result(), _result(), oos=_result(), oos_baseline=_result(),
                                stress=None, walk=None, sensitivity=None,
                                criteria=AcceptanceCriteria(), data_quality_ok=False)
    assert any(gate.name == "data_quality" and not gate.passed for gate in bad_data)


@pytest.mark.unit
def test_gate_objects_explain_requirement_measurement_and_detail():
    for gate in _gates():
        assert isinstance(gate, GateResult)
        assert gate.required and gate.measured is not None
        assert gate.name


@pytest.mark.unit
def test_parameter_sensitivity_runs_on_the_test_window_only(table):
    split = chronological_split(table.bars.index, 0.5, 0.25)
    stability, detail = parameter_sensitivity(baseline_strategy("1d"), table,
                                              EngineConfig(), oos_only=split)
    assert stability["variants"] == len(detail) == 16
    assert 0.0 <= stability["profitable_share"] <= 1.0
    assert "fragile" in stability


@pytest.mark.unit
def test_multiple_testing_summary_controls_the_family():
    summary = multiple_testing_summary([0.001, 0.02, 0.04, 0.5, 0.8])
    assert summary["tests"] == 5
    assert isinstance(summary["rejected"], list)
    assert summary["note"]


@pytest.mark.unit
def test_rejection_is_explainable_without_running_an_experiment():
    gates = [GateResult("oos_expectancy", False, "expectancy > 0", "-0.12R",
                        "the candidate lost money out of sample")]
    assert not gates_passed(gates)
    assert np.isfinite(len([g for g in gates if not g.passed]))
