"""Autopsy: a loss is not automatically a mistake, and a win proves nothing."""

from __future__ import annotations

import pytest

from tradingagents.research.autopsy import (
    autopsy_batch,
    autopsy_trade,
    recurring_weaknesses,
)
from tradingagents.research.contracts import MistakeCategory, TradeRecord, TradeSide
from tradingagents.research.risk import RiskPolicy


def _trade(**overrides) -> TradeRecord:
    payload = {
        "trade_id": "T-1", "strategy_id": "base", "version": "V01", "ticker": "TEST",
        "timeframe": "1d", "side": TradeSide.LONG, "entry_time": "2026-01-02T00:00:00Z",
        "exit_time": "2026-01-05T00:00:00Z", "entry_price": 100.0, "exit_price": 98.0,
        "quantity": 100.0, "stop_price": 98.0, "target_price": 104.0, "gross_pnl": -200.0,
        "cost": 2.0, "net_pnl": -202.0, "r_multiple": -1.0, "bars_held": 3,
        "exit_reason": "stop", "regime": "trend_up", "session": "london",
        "planned_risk": 200.0, "mae_r": 1.0, "mfe_r": 0.7,
        "signals": ["ema_20 > ema_50", "close > vwap"],
    }
    payload.update(overrides)
    return TradeRecord(**payload)


@pytest.mark.unit
def test_a_clean_loss_is_labelled_ordinary_variance():
    record = autopsy_trade(_trade(), atr_at_entry=2.0)
    assert record.verdict is MistakeCategory.NORMAL_LOSS
    assert record.is_loss
    assert not record.repeatable
    assert "not evidence against a strategy" in record.note


@pytest.mark.unit
def test_a_clean_win_is_still_not_proof():
    record = autopsy_trade(_trade(exit_price=104.0, net_pnl=398.0, r_multiple=2.0,
                                  exit_reason="target", mfe_r=2.0, mae_r=0.1,
                                  gross_pnl=400.0),
                           atr_at_entry=2.0)
    assert not record.is_loss
    assert record.verdict is MistakeCategory.UNKNOWN
    assert "does not validate" in record.note


@pytest.mark.unit
def test_a_trade_without_rule_evidence_cannot_be_audited():
    record = autopsy_trade(_trade(signals=[]), atr_at_entry=2.0)
    assert MistakeCategory.STRATEGY_LOGIC in {f.category for f in record.findings}


@pytest.mark.unit
def test_a_tight_stop_is_a_risk_finding_not_a_bad_luck_story():
    record = autopsy_trade(_trade(), atr_at_entry=4.0)  # stop is 0.5 ATR away
    assert any(f.category is MistakeCategory.RISK_MANAGEMENT for f in record.findings)
    assert "0.50 ATR" in " ".join(f.description for f in record.findings)


@pytest.mark.unit
def test_a_wide_stop_is_also_flagged():
    record = autopsy_trade(_trade(stop_price=90.0), atr_at_entry=2.0)  # 5 ATR
    assert any("wider than the setup" in f.description for f in record.findings)


@pytest.mark.unit
def test_giving_back_a_profitable_position_is_an_exit_problem():
    record = autopsy_trade(_trade(mfe_r=1.6), atr_at_entry=2.0)
    assert any("gave back" in f.description for f in record.findings)


@pytest.mark.unit
def test_immediate_adverse_move_suggests_a_late_entry():
    record = autopsy_trade(_trade(mfe_r=0.05, mae_r=1.1), atr_at_entry=2.0)
    assert any("premature" in f.question or "late" in f.question
               for f in record.findings)


@pytest.mark.unit
def test_costs_that_dominate_the_move_are_an_execution_finding():
    record = autopsy_trade(_trade(cost=70.0, gross_pnl=-200.0, net_pnl=-270.0),
                           atr_at_entry=2.0)
    assert any(f.category is MistakeCategory.EXECUTION for f in record.findings)


@pytest.mark.unit
def test_an_unplanned_exit_reason_is_a_logic_finding():
    record = autopsy_trade(_trade(exit_reason="manual_override"), atr_at_entry=2.0)
    assert any("not part of the tested exits" in f.description for f in record.findings)


@pytest.mark.unit
def test_a_conflicting_higher_timeframe_and_a_disallowed_regime_are_findings():
    record = autopsy_trade(_trade(), atr_at_entry=2.0, htf_conflict=True,
                           regime_allowed=False)
    categories = {f.category for f in record.findings}
    assert MistakeCategory.REGIME_CLASSIFICATION in categories
    assert MistakeCategory.STRATEGY_LOGIC in categories


@pytest.mark.unit
def test_a_breached_risk_budget_is_serious_and_wins_the_verdict():
    policy = RiskPolicy(max_risk_per_trade=0.01)
    record = autopsy_trade(_trade(planned_risk=5_000.0), atr_at_entry=2.0,
                           equity_at_entry=100_000.0, policy=policy)
    assert record.verdict is MistakeCategory.RISK_MANAGEMENT
    assert any(f.severity == "serious" for f in record.findings)


@pytest.mark.unit
def test_bad_input_data_is_reported_not_excused():
    record = autopsy_trade(_trade(), atr_at_entry=2.0, data_quality_ok=False)
    assert record.verdict is MistakeCategory.DATA_QUALITY


@pytest.mark.unit
def test_unknown_evidence_stays_unknown():
    # No ATR at entry: the stop cannot be judged, so no guess is made.
    record = autopsy_trade(_trade(mfe_r=0.3), atr_at_entry=None)
    assert not any(f.category is MistakeCategory.RISK_MANAGEMENT for f in record.findings)
    assert record.verdict is MistakeCategory.NORMAL_LOSS


@pytest.mark.unit
def test_a_weakness_only_becomes_repeatable_when_it_recurs():
    healthy = _trade(trade_id="T-ok", exit_price=104.0, net_pnl=398.0, r_multiple=2.0,
                     exit_reason="target", gross_pnl=400.0)
    losses = [_trade(trade_id=f"T-{i}", mfe_r=1.5) for i in range(4)]
    records = autopsy_batch([healthy, *losses], atr_at_entry=2.0, min_occurrences=3)
    assert records[0].repeatable is False
    repeated = [r for r in records[1:] if r.repeatable]
    assert repeated and "repeatable weakness" in repeated[0].note


@pytest.mark.unit
def test_recurring_weaknesses_are_ranked_by_how_often_they_appear():
    losses = ([_trade(trade_id=f"A{i}", mfe_r=1.5, cost=70.0, net_pnl=-270.0)
               for i in range(3)]
              + [_trade(trade_id=f"B{i}", mfe_r=0.05, mae_r=1.2) for i in range(2)])
    records = autopsy_batch(losses, atr_at_entry=2.0, min_occurrences=2)
    weaknesses = recurring_weaknesses(records)
    assert weaknesses
    assert weaknesses[0]["trades"] >= weaknesses[-1]["trades"]
    assert all({"category", "trades", "example", "action"} <= set(w) for w in weaknesses)


@pytest.mark.unit
def test_ordinary_variance_is_never_reported_as_a_repeatable_weakness():
    # Four clean losses: repeating "this was normal variance" would be nonsense.
    losses = [_trade(trade_id=f"N{i}") for i in range(4)]
    records = autopsy_batch(losses, atr_at_entry=2.0, min_occurrences=3)
    assert all(record.verdict is MistakeCategory.NORMAL_LOSS for record in records)
    assert not any(record.repeatable for record in records)
    assert recurring_weaknesses(records) == []


@pytest.mark.unit
def test_a_real_defect_is_listed_even_when_variance_dominates_the_batch():
    losses = [_trade(trade_id=f"N{i}") for i in range(3)]
    broken = [_trade(trade_id=f"B{i}", exit_reason="manual_override") for i in range(3)]
    records = autopsy_batch([*losses, *broken], atr_at_entry=2.0, min_occurrences=2)
    weaknesses = recurring_weaknesses(records)
    assert [entry["category"] for entry in weaknesses] == ["strategy_logic"]
    assert weaknesses[0]["repeatable"] is True
    assert weaknesses[0]["trades"] == 3
