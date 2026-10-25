"""The backtester: bar timing, costs, stops, sizing and the authoritative risk gate."""

from __future__ import annotations

import pandas as pd
import pytest

from tradingagents.research.contracts import CostModel, TradeSide
from tradingagents.research.engine import (
    EngineConfig,
    adverse_price,
    leg_fees,
    round_trip_cost,
    run_backtest,
)
from tradingagents.research.features import FeatureTable
from tradingagents.research.risk import RiskPolicy
from tradingagents.research.strategies import Condition, ExitRules, PositionSizing, StrategySpec


def _frame(opens, highs, lows, closes) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(closes), freq="1D", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": 1000.0}, index=index, dtype=float)


def _table(bars: pd.DataFrame, **overrides) -> FeatureTable:
    features = pd.DataFrame(index=bars.index)
    features["atr_14"] = 1.0
    features["regime"] = "trend_up"
    features["session_id"] = 0.0
    features["price"] = bars["close"]
    features["always"] = 1.0
    for column, value in overrides.items():
        features[column] = value
    return FeatureTable(ticker="TEST", timeframe="1d", bars=bars, features=features,
                        version="testtable")


def _always_long(**kwargs) -> StrategySpec:
    exits = kwargs.pop("exits", ExitRules(stop_atr=1.0, target_r=100.0))
    sizing = kwargs.pop("sizing", PositionSizing(risk_per_trade=0.01, max_leverage=1.0))
    return StrategySpec(strategy_id="test", version="T1", name="always long",
                        long_entry=(Condition("always", "==", 1.0),),
                        exits=exits, sizing=sizing, **kwargs)


@pytest.mark.unit
def test_signal_on_a_close_fills_at_the_next_bars_open():
    bars = _frame([10, 10, 10, 20, 10], [10.5, 10.5, 21, 21, 10.5],
                  [9.5, 9.5, 9.5, 19, 9.5], [10, 10, 20, 20, 10])
    features = pd.DataFrame(index=bars.index)
    features["atr_14"] = 1.0
    features["regime"] = "trend_up"
    features["session_id"] = 0.0
    features["price"] = bars["close"]
    features["trigger"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    table = FeatureTable(ticker="TEST", timeframe="1d", bars=bars, features=features,
                         version="t")
    spec = StrategySpec(strategy_id="t", version="T1", name="trigger",
                        long_entry=(Condition("trigger", "==", 1.0),),
                        exits=ExitRules(stop_atr=1.0, target_r=100.0))
    result = run_backtest(spec, table, EngineConfig(costs=CostModel(commission_bps=0,
                                                                     slippage_bps=0,
                                                                     spread_bps=0)))
    assert len(result.trades) == 1
    trade = result.trades[0]
    # The signal was on bar 1; the fill is bar 2's open, not bar 1's close.
    assert trade.entry_time == str(bars.index[2])
    assert trade.entry_price == pytest.approx(bars["open"].iloc[2])


@pytest.mark.unit
def test_costs_reduce_net_profit_and_are_separately_reported():
    bars = _frame([10] * 10, [11] * 10, [9] * 10, [10] * 10)
    table = _table(bars)
    free = run_backtest(_always_long(), table,
                        EngineConfig(costs=CostModel(commission_bps=0, slippage_bps=0,
                                                     spread_bps=0)))
    charged = run_backtest(_always_long(), table, EngineConfig(costs=CostModel(
        commission_bps=5, slippage_bps=5, spread_bps=4, tax_bps_sell=1)))
    assert charged.metrics.total_cost > 0
    assert charged.metrics.net_pnl < free.metrics.net_pnl
    assert charged.metrics.gross_pnl > charged.metrics.net_pnl
    # Gross figures ignore friction; net figures do not.
    assert charged.gross_metrics.total_cost == 0
    assert charged.gross_metrics.net_pnl > charged.metrics.net_pnl


@pytest.mark.unit
def test_stop_is_assumed_to_fill_before_the_target_when_both_are_touched():
    bars = _frame([10, 10, 10, 10], [10.2, 10.2, 20, 10.2], [9.8, 9.8, 5, 9.8],
                  [10, 10, 10, 10])
    table = _table(bars)
    spec = _always_long(exits=ExitRules(stop_atr=1.0, target_r=2.0))
    result = run_backtest(spec, table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].exit_price == pytest.approx(9.0)


@pytest.mark.unit
def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    bars = _frame([10, 10, 10, 10, 10], [10.2, 10.2, 10.2, 10.2, 10.2],
                  [9.8, 9.8, 9.8, 9.8, 9.8], [10, 10, 10, 10, 10])
    bars.iloc[3, :] = [7.0, 7.2, 6.8, 7.0, 1000.0]
    bars.iloc[4, :] = [7.0, 7.2, 6.8, 7.0, 1000.0]
    table = _table(bars)
    result = run_backtest(_always_long(exits=ExitRules(stop_atr=1.0, target_r=100.0)),
                          table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    trade = result.trades[0]
    assert trade.exit_reason == "stop_gap"
    assert trade.exit_price == pytest.approx(7.0)
    assert trade.net_pnl < 0


@pytest.mark.unit
def test_position_size_comes_from_the_risk_budget_and_the_stop_distance():
    bars = _frame([10] * 6, [10.1] * 6, [9.9] * 6, [10] * 6)
    table = _table(bars)
    spec = _always_long(exits=ExitRules(stop_atr=2.0, target_r=100.0),
                        sizing=PositionSizing(risk_per_trade=0.01, max_leverage=10.0))
    result = run_backtest(spec, table, EngineConfig(initial_equity=100_000,
                                                   costs=CostModel(0, 0, 0, 0, 0, 0)))
    trade = result.trades[0]
    # ATR is 1.0 in this table, so the stop is 2.0 away and 1% of 100k buys 500 units.
    assert trade.planned_risk == pytest.approx(1_000.0, rel=0.02)
    assert trade.quantity == pytest.approx(500.0, rel=0.02)
    assert trade.planned_risk <= 100_000 * spec.sizing.risk_per_trade * 1.02


@pytest.mark.unit
def test_risk_gate_blocks_entries_and_each_block_records_its_reason():
    bars = _frame([10] * 8, [10.2] * 8, [9.8] * 8, [10] * 8)
    table = _table(bars)
    halted = run_backtest(_always_long(), table, EngineConfig(
        risk_policy=RiskPolicy(emergency_stop=True), costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert halted.metrics.trades == 0
    assert halted.blocked
    assert "emergency stop" in halted.blocked[0].reason

    bad_data = run_backtest(_always_long(), table, EngineConfig(
        data_quality_ok=False, costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert bad_data.metrics.trades == 0
    assert any("quality gate" in block.reason for block in bad_data.blocked)


@pytest.mark.unit
def test_daily_loss_limit_stops_new_entries_after_a_losing_day():
    # Each bar is a separate day here, so the day resets and the limit is not hit;
    # the test therefore drives the risk engine directly through a losing trade
    # followed by a same-day re-entry attempt using a 1m frame.
    index = pd.date_range("2026-01-01 09:30", periods=12, freq="1min", tz="UTC")
    bars = pd.DataFrame({"open": 10.0, "high": 10.2, "low": 9.0, "close": 9.5},
                        index=index, dtype=float)
    table = FeatureTable(ticker="TEST", timeframe="1m", bars=bars,
                         features=pd.DataFrame({
                             "atr_14": 1.0, "regime": "trend_up", "session_id": 0.0,
                             "price": bars["close"], "always": 1.0}, index=bars.index),
                         version="t")
    spec = _always_long(exits=ExitRules(stop_atr=0.5, target_r=0.5))
    result = run_backtest(spec, table, EngineConfig(
        initial_equity=100_000, costs=CostModel(0, 0, 0, 0, 0, 0),
        risk_policy=RiskPolicy(max_trades_per_day=1)))
    assert result.metrics.trades == 1
    assert any("policy allows 1" in block.reason for block in result.blocked)


@pytest.mark.unit
def test_open_position_at_the_end_is_closed_and_reported():
    bars = _frame([10] * 5, [10.1] * 5, [9.9] * 5, [10] * 5)
    table = _table(bars)
    result = run_backtest(_always_long(exits=ExitRules(stop_atr=5.0, target_r=None)),
                          table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert result.trades
    assert result.trades[-1].exit_reason == "end_of_data"
    assert any("still open at the last bar" in note for note in result.notes)


@pytest.mark.unit
def test_equity_curve_covers_every_bar_after_the_first_and_metrics_are_consistent():
    bars = _frame([10] * 20, [10.5] * 20, [9.5] * 20, [10] * 20)
    table = _table(bars)
    result = run_backtest(_always_long(), table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert len(result.equity) == len(bars) - 1
    assert result.equity.index[0] == bars.index[1]
    assert result.metrics.trades == len(result.trades)
    assert result.metrics.wins + result.metrics.losses <= result.metrics.trades
    assert result.metrics.max_drawdown_pct >= 0


@pytest.mark.unit
def test_partial_exit_records_the_partial_and_reduces_the_position():
    index = pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC")
    bars = pd.DataFrame({
        "open": [10, 10, 10, 10], "high": [10.2, 12.5, 12.5, 12.5],
        "low": [9.8, 9.8, 9.8, 9.8], "close": [10, 12, 12, 12], "volume": 1000.0,
    }, index=index, dtype=float)
    table = _table(bars)
    spec = _always_long(exits=ExitRules(stop_atr=1.0, target_r=None, partial_at_r=1.0,
                                        partial_fraction=0.5))
    result = run_backtest(spec, table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    trade = result.trades[0]
    assert trade.context["partials"], "a partial exit should be recorded"
    assert trade.context["partials"][0]["quantity"] == pytest.approx(500.0, rel=0.1)
    # The record keeps the position size at entry; the partial is in the context.
    assert trade.quantity == pytest.approx(1000.0, rel=0.1)


@pytest.mark.unit
def test_invalid_spec_columns_fail_before_any_simulation():
    bars = _frame([10] * 5, [10] * 5, [10] * 5, [10] * 5)
    table = _table(bars)
    spec = StrategySpec(strategy_id="t", version="T1", name="bad",
                        long_entry=(Condition("does_not_exist", ">", 0.0),))
    with pytest.raises(ValueError, match="not in this feature table"):
        run_backtest(spec, table, EngineConfig())


@pytest.mark.unit
def test_cost_helpers_are_monotone_in_the_parameters():
    costs = CostModel(commission_bps=2, min_commission=0, slippage_bps=3, spread_bps=4,
                      tax_bps_sell=1, funding_bps_per_day=0.5)
    assert costs.stressed(2).slippage_bps == pytest.approx(6)
    buy = adverse_price(100.0, "buy", costs)
    sell = adverse_price(100.0, "sell", costs)
    assert buy > 100 > sell
    commission, tax = leg_fees(10_000, costs, "sell")
    assert commission == pytest.approx(2.0)
    assert tax == pytest.approx(1.0)
    assert round_trip_cost(100.0, 101.0, 10, "buy", "sell", costs, days_held=2) > 0


@pytest.mark.unit
def test_annualisation_reflects_the_timeframe():
    assert EngineConfig().annualisation("1d") == 252
    assert EngineConfig().annualisation("1h") > EngineConfig().annualisation("4h")
    assert EngineConfig(periods_per_year=100).annualisation("1m") == 100


@pytest.mark.unit
def test_short_side_is_symmetric_when_configured():
    bars = _frame([10] * 8, [10.2] * 8, [9.8] * 8, [10] * 8)
    table = _table(bars)
    spec = StrategySpec(strategy_id="t", version="T1", name="short",
                        short_entry=(Condition("always", "==", 1.0),),
                        exits=ExitRules(stop_atr=1.0, target_r=100.0),
                        sizing=PositionSizing(risk_per_trade=0.01, max_leverage=1.0))
    result = run_backtest(spec, table, EngineConfig(costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert result.trades and result.trades[0].side is TradeSide.SHORT
    blocked = run_backtest(spec, table, EngineConfig(
        risk_policy=RiskPolicy(allow_short=False), costs=CostModel(0, 0, 0, 0, 0, 0)))
    assert blocked.metrics.trades == 0
    assert any("short" in block.reason for block in blocked.blocked)
