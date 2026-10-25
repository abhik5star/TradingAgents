"""The risk engine: independent, tightening-only and impossible to weaken from AI output."""

from __future__ import annotations

import pytest

from tradingagents.research.contracts import TradeSide
from tradingagents.research.risk import (
    OrderIntent,
    RiskEngine,
    RiskPolicy,
    RiskState,
    load_policy,
    save_policy,
    tightened,
)
from tradingagents.research.store import ResearchStore


def _intent(**overrides) -> OrderIntent:
    payload = {"ticker": "TEST", "timeframe": "1d", "side": TradeSide.LONG, "price": 100.0,
               "quantity": 10.0, "stop_price": 95.0, "risk_amount": 50.0, "equity": 100_000.0}
    payload.update(overrides)
    return OrderIntent(**payload)


def _state(**overrides) -> RiskState:
    payload = {"equity": 100_000.0, "peak_equity": 100_000.0, "day_start_equity": 100_000.0}
    payload.update(overrides)
    return RiskState(**payload)


@pytest.mark.unit
def test_a_clean_order_is_allowed_and_not_resized():
    decision = RiskEngine(RiskPolicy()).evaluate(_intent(), _state())
    assert decision.allowed
    assert decision.size_multiplier == 1.0
    assert all(decision.checks.values())


@pytest.mark.unit
def test_an_order_without_a_stop_is_refused():
    decision = RiskEngine().evaluate(_intent(stop_price=None), _state())
    assert not decision.allowed
    assert any("stop" in reason for reason in decision.reasons)


@pytest.mark.unit
@pytest.mark.parametrize("change", [
    {"max_risk_per_trade": 0.05},
    {"max_daily_loss_pct": 0.5},
    {"max_drawdown_pct": 0.9},
    {"max_leverage": 10.0},
    {"max_open_positions": 5},
    {"max_trades_per_day": 100},
    {"require_data_quality": False},
    {"drift_size_multiplier": 1.0},
    {"min_equity_fraction": 0.1},
])
def test_every_loosening_change_is_refused_and_named(change):
    engine = RiskEngine(RiskPolicy())
    proposed = RiskPolicy(**{**engine.policy.as_dict(), **change})
    review = engine.review_policy_change(proposed)
    assert not review.allowed
    assert review.weakened
    # The live policy is untouched by a refused change.
    assert engine.policy == RiskPolicy()


@pytest.mark.unit
def test_a_tightening_change_is_accepted_and_applied():
    engine = RiskEngine(RiskPolicy())
    proposed = RiskPolicy(**{**engine.policy.as_dict(), "max_risk_per_trade": 0.005,
                             "max_leverage": 0.5})
    review = engine.apply_policy_change(proposed)
    assert review.allowed
    assert engine.policy.max_risk_per_trade == 0.005


@pytest.mark.unit
def test_an_engaged_emergency_stop_cannot_be_released_without_a_new_policy_review():
    policy = RiskPolicy(emergency_stop=True)
    engine = RiskEngine(policy)
    assert not engine.evaluate(_intent(), _state()).allowed
    proposed = RiskPolicy(**{**policy.as_dict(), "emergency_stop": False})
    assert not engine.review_policy_change(proposed).allowed
    # Releasing it through a policy that starts released is a new, allowed policy.
    assert RiskEngine(RiskPolicy(emergency_stop=False)).evaluate(_intent(), _state()).allowed


@pytest.mark.unit
def test_data_quality_failure_blocks_entries():
    decision = RiskEngine().evaluate(_intent(), _state(), quality_ok=False)
    assert not decision.allowed
    assert "quality gate" in decision.blocked_reason


@pytest.mark.unit
def test_daily_loss_and_drawdown_limits_halt_new_entries():
    state = _state(day_pnl=-3_500.0)
    decision = RiskEngine().evaluate(_intent(), state)
    assert not decision.allowed
    assert "daily loss" in decision.blocked_reason

    state = _state(equity=84_000.0, peak_equity=100_000.0)
    decision = RiskEngine().evaluate(_intent(), state)
    assert not decision.allowed
    assert "drawdown" in decision.blocked_reason


@pytest.mark.unit
def test_exposure_and_trade_frequency_limits_are_enforced():
    decision = RiskEngine().evaluate(_intent(), _state(open_positions=1))
    assert not decision.allowed
    decision = RiskEngine().evaluate(_intent(), _state(trades_today=6))
    assert not decision.allowed
    decision = RiskEngine().evaluate(_intent(quantity=2_000.0, risk_amount=10_000.0), _state())
    assert not decision.allowed
    assert "notional" in decision.blocked_reason


@pytest.mark.unit
def test_oversized_risk_is_capped_not_approved_at_the_requested_size():
    policy = RiskPolicy(max_risk_per_trade=0.01)
    decision = RiskEngine(policy).evaluate(_intent(risk_amount=5_000.0), _state())
    assert decision.allowed
    assert decision.size_multiplier == pytest.approx(0.2)
    assert any("per-trade cap" in reason for reason in decision.reasons)


@pytest.mark.unit
def test_model_drift_only_ever_reduces_size():
    policy = RiskPolicy(require_no_model_drift=False, drift_size_multiplier=0.5)
    decision = RiskEngine(policy).evaluate(_intent(), _state(), model_drift=True)
    assert decision.allowed
    assert decision.size_multiplier <= 0.5

    strict = RiskPolicy(require_no_model_drift=True)
    blocked = RiskEngine(strict).evaluate(_intent(), _state(), model_drift=True)
    assert not blocked.allowed
    assert "drift" in blocked.blocked_reason


@pytest.mark.unit
def test_blocks_are_recorded_with_reasons_and_state(tmp_path):
    store = ResearchStore(root=tmp_path)
    state = _state(open_positions=1)
    RiskEngine(RiskPolicy(), store).evaluate(_intent(), state)
    events = store.read("risk_event")
    assert events and events[-1]["kind"] == "entry_blocked"
    assert events[-1]["reasons"]
    assert events[-1]["state"]["open_positions"] == 1
    assert state.blocked == 1


@pytest.mark.unit
def test_policy_round_trips_through_the_store(tmp_path):
    store = ResearchStore(root=tmp_path)
    assert load_policy(store) == RiskPolicy()  # nothing stored: safe default
    tightened_policy = RiskPolicy(max_risk_per_trade=0.002, max_leverage=0.5)
    save_policy(store, tightened_policy)
    assert load_policy(store).max_risk_per_trade == 0.002
    assert load_policy(store).max_leverage == 0.5


@pytest.mark.unit
def test_tightened_helper_reports_without_applying():
    policy = RiskPolicy()
    proposed, review = tightened(policy, max_leverage=0.25)
    assert review.allowed and proposed.max_leverage == 0.25
    proposed, review = tightened(policy, max_leverage=4.0)
    assert not review.allowed


@pytest.mark.unit
def test_risk_state_rolls_the_day_and_reports_its_own_limits():
    state = _state(day_pnl=-1_000.0)
    state.roll_day(state.day, 100_000.0)
    assert state.daily_loss_pct == pytest.approx(0.01)
    state.roll_day(__import__("datetime").date(2026, 1, 2), 95_000.0)
    assert state.day_pnl == 0.0
    assert state.day_start_equity == 95_000.0
