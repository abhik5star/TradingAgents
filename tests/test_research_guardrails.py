"""Guardrails: nothing computed before the cut may depend on anything after it.

The strongest available test of look-ahead bias is counterfactual: build the
features and the backtest twice, once with the full history and once with the
history truncated at a cut, and require the two to agree on every bar that both
runs could see. Any feature, alignment or fill rule that peeks forward changes
the past the moment the future changes, and fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.research_fixtures import market_data, synthetic_bars
from tradingagents.research.contracts import CostModel
from tradingagents.research.engine import EngineConfig, run_backtest
from tradingagents.research.features import build_feature_table
from tradingagents.research.strategies import (
    Condition,
    ExitRules,
    PositionSizing,
    StrategySpec,
)


def _spec() -> StrategySpec:
    return StrategySpec(strategy_id="guardrail", version="V01", name="trend follow",
                        long_entry=(Condition("ema_21", ">", "ema_50"),),
                        exits=ExitRules(stop_atr=2.0, target_r=2.0),
                        sizing=PositionSizing(risk_per_trade=0.01),
                        allowed_regimes=("trend_up", "trend_down", "breakout"))

CUT = 260
LOOKAHEAD = 200


@pytest.fixture(scope="module")
def frames():
    bars = synthetic_bars(n=CUT + LOOKAHEAD, seed=17)
    full = build_feature_table(market_data(bars), "1d")
    truncated = build_feature_table(market_data(bars.iloc[:CUT]), "1d")
    return bars, full, truncated


@pytest.mark.unit
def test_no_feature_changes_when_the_unseen_future_changes(frames):
    bars, full, truncated = frames
    # The truncation point is early enough that the warm-up window is complete.
    comparison_end = CUT - 60
    left = full.features.loc[:full.features.index[comparison_end]]
    right = truncated.features.loc[:truncated.features.index[comparison_end]]
    numeric = [c for c in left.columns if pd.api.types.is_numeric_dtype(left[c])
               and pd.api.types.is_numeric_dtype(right[c])]
    shared = [c for c in numeric if c in right.columns]
    assert len(shared) > 100
    mismatches: list[str] = []
    for column in shared:
        a = left[column].to_numpy(dtype=float)
        b = right[column].to_numpy(dtype=float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=False)
        if not (both_nan | close).all():
            worst = int(np.argmax(~(both_nan | close)))
            mismatches.append(f"{column}@{left.index[worst]}: {a[worst]} != {b[worst]}")
    assert not mismatches, "future data leaked into the past: " + "; ".join(mismatches[:5])


@pytest.mark.unit
def test_multi_timeframe_context_is_also_causal(frames):
    # A coarser context frame ends later than the primary bar it informs, so this
    # is where an as-of merge without a known-at column would leak.
    intraday = synthetic_bars(n=700, seed=23, timeframe="1h")
    full_table = build_feature_table(market_data(intraday, timeframes=["1h", "4h"]), "1h",
                                      context_timeframe="4h")
    cut = 520
    truncated_table = build_feature_table(
        market_data(intraday.iloc[:cut], timeframes=["1h", "4h"]), "1h",
        context_timeframe="4h")
    shared_index = truncated_table.features.index[:-10]
    left = full_table.features.loc[shared_index]
    right = truncated_table.features.loc[shared_index]
    context_columns = [c for c in left.columns if c.startswith("htf_")]
    assert context_columns
    for column in context_columns:
        a = left[column].to_numpy(dtype=float)
        b = right[column].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"{column} used a future htf bar"


@pytest.mark.unit
def test_the_backtest_decisions_before_the_cut_are_identical(frames):
    bars, full, truncated = frames
    spec = _spec()
    costs = CostModel(commission_bps=5.0, slippage_bps=5.0, spread_bps=2.0,
                      tax_bps_sell=2.0, funding_bps_per_day=0.0)
    long_run = run_backtest(spec, full, EngineConfig(costs=costs, initial_equity=100_000.0))
    short_run = run_backtest(spec, truncated,
                             EngineConfig(costs=costs, initial_equity=100_000.0))
    last_common_bar = truncated.bars.index[-2]
    # Entries that fired well before the cut must be identical, bar for bar.
    comparison_cut = truncated.bars.index[CUT - 60]
    left = [t for t in long_run.trades if pd.Timestamp(t.entry_time) <= comparison_cut]
    right = [t for t in short_run.trades if pd.Timestamp(t.entry_time) <= comparison_cut]
    assert left or right, "the fixture produced no trades to compare"
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert (a.entry_time, a.side, a.entry_price) == (b.entry_time, b.side, b.entry_price)
        assert a.exit_reason == b.exit_reason
        assert a.r_multiple == pytest.approx(b.r_multiple, abs=1e-9)
    assert last_common_bar > comparison_cut


@pytest.mark.unit
def test_a_planted_lookahead_is_caught_by_the_counterfactual(frames):
    bars, full, _ = frames
    leaky = full.features.copy()
    # This is the mistake the probe exists for: normalising by the whole sample.
    leaky["z_close"] = ((full.bars["close"] - full.bars["close"].mean())
                        / full.bars["close"].std())
    truncated = build_feature_table(market_data(bars.iloc[:CUT]), "1d")
    honest = ((truncated.bars["close"] - truncated.bars["close"].mean())
              / truncated.bars["close"].std())
    shared = truncated.features.index[:-1]
    assert not np.allclose(leaky.loc[shared, "z_close"], honest.loc[shared]), \
        "the counterfactual check must flag whole-sample normalisation"


@pytest.mark.unit
def test_cost_stress_can_only_reduce_performance(frames):
    _, full, _ = frames
    spec = _spec()
    cheap = run_backtest(spec, full, EngineConfig(costs=CostModel(0.0, 0.0, 0.0, 0.0, 0.0)))
    expensive = run_backtest(spec, full,
                             EngineConfig(costs=CostModel(0.002, 0.002, 0.001, 0.001, 0.0)))
    assert expensive.metrics.expectancy_r <= cheap.metrics.expectancy_r
    assert expensive.metrics.net_pnl <= cheap.metrics.net_pnl
