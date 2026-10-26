"""The event-driven backtester: rules in, trades and evidence out.

Timing contract, and it never varies:

1. A bar closes. Features for that bar become knowable at its close.
2. ``StrategySpec.evaluate`` reads those features and produces a signal.
3. The order fills at the **next bar's open**, never at the signal bar's close.
4. Inside the next bar, exits are checked against the high/low with the
   pessimistic assumption that the stop fills before the target when both are
   touched, and that a gap through the stop fills at the open.

Costs are charged on both legs from an explicit
:class:`~tradingagents.research.contracts.CostModel` (commission, half-spread,
slippage, sell-side tax, carry). Nothing is marked to a mid price it could not
have traded at.

The risk gate is external and authoritative: the engine asks
:class:`tradingagents.research.risk.RiskEngine` before every entry, and a blocked
entry is recorded with its reason instead of being quietly dropped. Position size
can only be reduced by the gate, never raised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tradingagents.research.contracts import (
    CostModel,
    Metrics,
    TradeRecord,
    TradeSide,
    stable_hash,
    utc_now,
)
from tradingagents.research.features import FeatureTable
from tradingagents.research.quant import compute_metrics
from tradingagents.research.risk import OrderIntent, RiskEngine, RiskPolicy, RiskState
from tradingagents.research.strategies import StrategySpec

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}

#: Session id → name, matching the order of the ``Session`` enum.
SESSION_NAMES = {0.0: "asia", 1.0: "london", 2.0: "london_ny_overlap",
                 3.0: "new_york", 4.0: "off_hours"}


@dataclass
class EngineConfig:
    """Execution assumptions for one run. Defaults are deliberately pessimistic."""

    initial_equity: float = 100_000.0
    costs: CostModel = field(default_factory=CostModel)
    max_bars_in_trade: int | None = None
    periods_per_year: int | None = None
    risk_policy: RiskPolicy = field(default_factory=RiskPolicy)
    enforce_risk_gate: bool = True
    data_quality_ok: bool = True
    model_drift: bool = False
    one_position_at_a_time: bool = True

    def annualisation(self, timeframe: str) -> int:
        if self.periods_per_year:
            return self.periods_per_year
        minutes = TIMEFRAME_MINUTES.get(timeframe, 1440)
        if minutes >= 1440:
            return 252
        # Cash-session instruments: ~6.5h of 60m bars, scaled by bar size.
        bars_per_session = max(1.0, 390 / minutes)
        return int(bars_per_session * 252)


@dataclass
class BlockedEntry:
    """An entry the risk gate refused, with the reason it gave."""

    timestamp: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"timestamp": self.timestamp, "reason": self.reason}


@dataclass
class BacktestResult:
    """Everything one deterministic run produced, including what it refused to do."""

    spec: StrategySpec
    ticker: str
    timeframe: str
    trades: list[TradeRecord]
    metrics: Metrics
    equity: pd.Series
    gross_metrics: Metrics
    config: EngineConfig
    signals: int = 0
    entries: int = 0
    blocked: list[BlockedEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dataset_fingerprint: str = ""
    feature_version: str = ""
    started_at: str = field(default_factory=utc_now)
    bars: int = 0
    risk_state: dict[str, Any] = field(default_factory=dict)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=["trade_id", "entry_time", "side", "net_pnl"])
        return pd.DataFrame([t.as_dict() for t in self.trades])

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.spec.as_dict(),
            "ticker": self.ticker,
            "timeframe": self.timeframe,
            "bars": self.bars,
            "signals": self.signals,
            "entries": self.entries,
            "blocked": [b.as_dict() for b in self.blocked],
            "metrics": self.metrics.as_dict(),
            "gross_metrics": self.gross_metrics.as_dict(),
            "costs": self.config.costs.as_dict(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_version": self.feature_version,
            "notes": self.notes,
            "risk_state": self.risk_state,
            "started_at": self.started_at,
        }

    def summary_line(self) -> str:
        m = self.metrics
        profit_factor = f"{m.profit_factor:.2f}" if m.profit_factor is not None else "n/a"
        return (f"{self.spec.version} {self.ticker} {self.timeframe}: {m.trades} trades, "
                f"expectancy {m.expectancy_r:+.2f}R, PF {profit_factor}, "
                f"max DD {m.max_drawdown_pct:.1%}, net {m.net_pnl:+.0f}")


def adverse_price(raw: float, side: str, costs: CostModel) -> float:
    """The price a market order actually gets: through the half-spread and slippage."""
    offset = (costs.spread_bps / 2 + costs.slippage_bps) / 1e4
    return raw * (1 + offset) if side == "buy" else raw * (1 - offset)


def leg_fees(notional: float, costs: CostModel, side: str) -> tuple[float, float]:
    """Commission and tax in cash for one leg of ``notional``."""
    commission = max(costs.min_commission, abs(notional) * costs.commission_bps / 1e4)
    tax = abs(notional) * costs.tax_bps_sell / 1e4 if side == "sell" else 0.0
    return commission, tax


def round_trip_cost(entry_raw: float, exit_raw: float, quantity: float, entry_side: str,
                    exit_side: str, costs: CostModel, days_held: float = 0.0) -> float:
    """Every cash friction for closing ``quantity`` units, in the account currency.

    The slippage/spread term is the difference between the adverse fill prices and
    the raw prices, which is exactly what the engine used to fill the orders.
    """
    offset = (costs.spread_bps / 2 + costs.slippage_bps) / 1e4
    friction = offset * (abs(entry_raw) + abs(exit_raw)) * abs(quantity)
    entry_commission, _ = leg_fees(entry_raw * abs(quantity), costs, entry_side)
    exit_commission, exit_tax = leg_fees(exit_raw * abs(quantity), costs, exit_side)
    funding = (abs(entry_raw * quantity) * costs.funding_bps_per_day / 1e4 * max(0.0, days_held))
    return friction + entry_commission + exit_commission + exit_tax + funding


def run_backtest(spec: StrategySpec, table: FeatureTable,
                 config: EngineConfig | None = None) -> BacktestResult:
    """Simulate ``spec`` over ``table`` exactly once. Pure function of its inputs."""
    config = config or EngineConfig()
    bars = table.bars
    features = table.features
    if len(bars) != len(features):
        raise ValueError("bars and features are not aligned; refusing to guess an alignment")
    spec.validate_columns(features.columns)

    annualisation = config.annualisation(table.timeframe)
    signals = spec.evaluate(features)
    # The signal on bar i-1 is the order for bar i's open. This single shift is
    # what separates a backtest from a plausible-looking fiction.
    desired = signals.shift(1).fillna(False)

    risk = RiskEngine(config.risk_policy) if config.enforce_risk_gate else None
    state = RiskState(equity=config.initial_equity, peak_equity=config.initial_equity,
                      day=None, day_start_equity=config.initial_equity)
    realized = config.initial_equity
    equity_points: list[float] = []
    trades: list[TradeRecord] = []
    gross_trades: list[TradeRecord] = []
    blocked: list[BlockedEntry] = []
    notes: list[str] = []

    position: dict[str, Any] | None = None
    in_market_bars = 0
    signal_count = 0

    index = bars.index
    open_, high, low, close = (bars[c].to_numpy(dtype=float) for c in
                               ("open", "high", "low", "close"))
    atr_values = features["atr_14"].to_numpy(dtype=float) if "atr_14" in features.columns else None
    regime_values = features["regime"].astype(str).to_numpy()
    session_values = (features["session_id"].astype(float).to_numpy()
                      if "session_id" in features.columns else np.zeros(len(bars)))

    for i in range(1, len(bars)):
        timestamp = index[i]
        day = timestamp.tz_convert("UTC").date() if timestamp.tzinfo else timestamp.date()
        state.roll_day(day, realized)

        # ---------------- manage an open position on this bar ----------------
        if position is not None:
            in_market_bars += 1
            exit_info = _manage_position(position, i, high, low, close, open_, timestamp, config)
            if exit_info is not None:
                trade, gross_trade, pnl = _close_position(
                    position, exit_info, spec, table, timestamp, config.costs)
                realized += pnl
                state.equity = realized
                state.peak_equity = max(state.peak_equity, realized)
                state.open_positions = max(0, state.open_positions - 1)
                state.day_pnl = realized - state.day_start_equity
                state.consecutive_losses = (state.consecutive_losses + 1 if pnl < 0 else 0)
                trades.append(trade)
                gross_trades.append(gross_trade)
                position = None
                if risk is not None:
                    risk.record_exit(trade.net_pnl, state)
                # A bar that closed a position is not reused to open another one.
                equity_points.append(realized)
                continue

        # ---------------- consider a new entry at this bar's open ----------------
        want_long = bool(desired["long"].iloc[i])
        want_short = bool(desired["short"].iloc[i])
        if want_long or want_short:
            signal_count += 1
        if position is None and (want_long or want_short) and atr_values is not None:
            previous = i - 1
            atr_at_signal = float(atr_values[previous])
            if not math.isfinite(atr_at_signal) or atr_at_signal <= 0:
                blocked.append(BlockedEntry(str(timestamp), "no ATR at the signal bar; cannot size"))
            else:
                entry_raw = float(open_[i])
                stop_distance = spec.exits.stop_atr * atr_at_signal
                if entry_raw <= 0 or stop_distance <= 0:
                    blocked.append(BlockedEntry(str(timestamp), "invalid price or stop distance"))
                else:
                    side = TradeSide.LONG if want_long else TradeSide.SHORT
                    equity_now = max(realized, 1e-9)
                    quantity = equity_now * spec.sizing.risk_per_trade / stop_distance
                    leverage_cap = equity_now * min(spec.sizing.max_leverage,
                                                    config.risk_policy.max_leverage)
                    quantity = min(quantity, leverage_cap / entry_raw)
                    if quantity <= max(0.0, spec.sizing.min_quantity):
                        blocked.append(BlockedEntry(str(timestamp), "size rounds to zero"))
                    else:
                        intent = OrderIntent(
                            ticker=table.ticker, timeframe=table.timeframe, side=side,
                            price=entry_raw, quantity=quantity,
                            stop_price=(entry_raw - stop_distance if side is TradeSide.LONG
                                        else entry_raw + stop_distance),
                            risk_amount=quantity * stop_distance,
                            equity=equity_now, regime=str(regime_values[previous]),
                        )
                        decision = (risk.evaluate(intent, state, quality_ok=config.data_quality_ok,
                                                  model_drift=config.model_drift)
                                    if risk is not None else None)
                        if decision is not None and not decision.allowed:
                            blocked.append(BlockedEntry(str(timestamp),
                                                        "; ".join(decision.reasons) or "risk block"))
                        else:
                            multiplier = decision.size_multiplier if decision is not None else 1.0
                            quantity *= multiplier
                            entry_side = "buy" if side is TradeSide.LONG else "sell"
                            entry_fill = adverse_price(entry_raw, entry_side, config.costs)
                            stop_price = (entry_fill - stop_distance if side is TradeSide.LONG
                                          else entry_fill + stop_distance)
                            target_price = None
                            if spec.exits.target_r is not None:
                                target_price = (entry_fill + spec.exits.target_r * stop_distance
                                                if side is TradeSide.LONG
                                                else entry_fill - spec.exits.target_r * stop_distance)
                            position = {
                                "side": side,
                                "entry_time": timestamp,
                                "entry_index": i,
                                "entry_raw": entry_raw,
                                "entry_fill": entry_fill,
                                "entry_side": entry_side,
                                "quantity": quantity,
                                "initial_quantity": quantity,
                                "stop": stop_price,
                                "target": target_price,
                                "stop_distance": stop_distance,
                                "atr_at_entry": atr_at_signal,
                                "regime": str(regime_values[previous]),
                                "session": SESSION_NAMES.get(float(session_values[previous]),
                                                             "unknown"),
                                "signals": _fired_labels(spec, features, previous),
                                "mfe_r": 0.0,
                                "mae_r": 0.0,
                                "bars": 0,
                                "partial_done": False,
                                "realized_pnl": 0.0,
                                "realized_gross": 0.0,
                                "realized_cost": 0.0,
                                "partials": [],
                                "trailing_atr": spec.exits.trailing_atr,
                                "trail_after_r": spec.exits.trail_after_r,
                                "partial_at_r": spec.exits.partial_at_r,
                                "partial_fraction": spec.exits.partial_fraction,
                                "breakeven_at_r": spec.exits.move_stop_to_breakeven_at_r,
                                "time_stop_bars": spec.exits.time_stop_bars,
                                "decision": decision.as_dict() if decision is not None else None,
                            }
                            state.open_positions += 1
                            state.trades_today += 1

        # ---------------- mark to market ----------------
        if position is not None:
            side_sign = 1.0 if position["side"] is TradeSide.LONG else -1.0
            equity_points.append(realized + side_sign * (close[i] - position["entry_fill"])
                                 * position["quantity"])
        else:
            equity_points.append(realized)

    if position is not None:
        # Close whatever is still open at the last bar's close, and say so: an
        # open position at the end is a real cost, not a win.
        exit_info = {"price": float(close[-1]), "reason": "end_of_data", "index": len(bars) - 1}
        trade, gross_trade, pnl = _close_position(position, exit_info, spec, table, index[-1],
                                                  config.costs)
        realized += pnl
        trades.append(trade)
        gross_trades.append(gross_trade)
        equity_points[-1] = realized
        notes.append("a position was still open at the last bar and was closed at its close")

    equity = pd.Series(equity_points, index=index[1:], name="equity")
    metrics = compute_metrics(trades, equity.to_numpy(), config.initial_equity,
                              bars_in_market=in_market_bars, total_bars=len(bars),
                              periods_per_year=annualisation)
    gross_metrics = compute_metrics(gross_trades, equity.to_numpy(), config.initial_equity,
                                    bars_in_market=in_market_bars, total_bars=len(bars),
                                    periods_per_year=annualisation)

    return BacktestResult(
        spec=spec, ticker=table.ticker, timeframe=table.timeframe, trades=trades,
        metrics=metrics, equity=equity, gross_metrics=gross_metrics, config=config,
        signals=signal_count, entries=len(trades), blocked=blocked, notes=notes,
        dataset_fingerprint=stable_hash(table.summary())[:16],
        feature_version=table.version, bars=len(bars),
        risk_state=state.as_dict(),
    )


def _manage_position(position: dict[str, Any], i: int, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, open_: np.ndarray, timestamp: Any,
                     config: EngineConfig) -> dict[str, Any] | None:
    """Apply stops, targets, partials, trailing and time stops to bar ``i``."""
    side = position["side"]
    sign = 1.0 if side is TradeSide.LONG else -1.0
    entry = position["entry_fill"]
    stop_distance = position["stop_distance"]
    position["bars"] += 1

    # Favorable/adverse excursion, in R, from this bar's extremes.
    favorable = (high[i] - entry) if sign > 0 else (entry - low[i])
    adverse = (entry - low[i]) if sign > 0 else (high[i] - entry)
    position["mfe_r"] = max(position["mfe_r"], favorable / stop_distance)
    position["mae_r"] = max(position["mae_r"], adverse / stop_distance)

    # A gap through the stop fills at the open, which is worse than the stop.
    if (sign > 0 and open_[i] <= position["stop"]) or (sign < 0 and open_[i] >= position["stop"]):
        return {"price": float(open_[i]), "reason": "stop_gap", "index": i}

    stop_hit = low[i] <= position["stop"] if sign > 0 else high[i] >= position["stop"]
    target = position["target"]
    target_hit = target is not None and (high[i] >= target if sign > 0 else low[i] <= target)
    if stop_hit:
        # Both inside the same bar: the stop is assumed to have been first.
        return {"price": float(position["stop"]), "reason": "stop", "index": i}
    if target_hit:
        return {"price": float(target), "reason": "target", "index": i}

    # Partial exit at a configured R multiple, once.
    partial_r = position.get("partial_at_r")
    if partial_r is not None and not position["partial_done"]:
        partial_price = (entry + partial_r * stop_distance if sign > 0
                         else entry - partial_r * stop_distance)
        reached = high[i] >= partial_price if sign > 0 else low[i] <= partial_price
        if reached:
            fraction = position.get("partial_fraction", 0.5)
            closed = position["quantity"] * fraction
            gross = sign * (partial_price - entry) * closed
            cost = round_trip_cost(entry, partial_price, closed,
                                   position["entry_side"],
                                   "sell" if sign > 0 else "buy", config.costs)
            position["realized_pnl"] += gross - cost
            position["realized_gross"] = position.get("realized_gross", 0.0) + gross
            position["realized_cost"] = position.get("realized_cost", 0.0) + cost
            position["partials"].append({"price": float(partial_price),
                                         "quantity": float(closed),
                                         "r_multiple": float(partial_r),
                                         "time": str(timestamp)})
            position["quantity"] -= closed
            position["partial_done"] = True
            if position["quantity"] <= 1e-12:
                return {"price": float(partial_price), "reason": "partial_full", "index": i}

    # Trailing stop and break-even moves happen at the close, using information
    # from this bar only.
    trail_atr = position.get("trailing_atr")
    if trail_atr is not None and position["mfe_r"] >= position.get("trail_after_r", 1.0):
        candidate = (close[i] - trail_atr * position["atr_at_entry"]) if sign > 0 else (
            close[i] + trail_atr * position["atr_at_entry"])
        position["stop"] = (max(position["stop"], candidate) if sign > 0
                            else min(position["stop"], candidate))
    breakeven_r = position.get("breakeven_at_r")
    if breakeven_r is not None and position["mfe_r"] >= breakeven_r:
        position["stop"] = (max(position["stop"], entry) if sign > 0
                            else min(position["stop"], entry))

    time_stop = position.get("time_stop_bars")
    if time_stop is not None and position["bars"] >= time_stop:
        return {"price": float(close[i]), "reason": "time_stop", "index": i}
    if config.max_bars_in_trade and position["bars"] >= config.max_bars_in_trade:
        return {"price": float(close[i]), "reason": "max_bars", "index": i}
    return None


def _close_position(position: dict[str, Any], exit_info: dict[str, Any], spec: StrategySpec,
                    table: FeatureTable, timestamp: Any,
                    costs: CostModel) -> tuple[TradeRecord, TradeRecord, float]:
    """Turn a position into one net trade record, one gross record and a cash delta."""
    side = position["side"]
    sign = 1.0 if side is TradeSide.LONG else -1.0
    entry = position["entry_fill"]
    entry_raw = position["entry_raw"]
    exit_price = float(exit_info["price"])
    quantity = float(position["quantity"])
    exit_side = "sell" if sign > 0 else "buy"

    gross_remaining = sign * (exit_price - entry_raw) * quantity
    cost_remaining = round_trip_cost(entry_raw, exit_price, quantity, position["entry_side"],
                                     exit_side, costs,
                                     days_held=_days_between(position["entry_time"], timestamp,
                                                             table.timeframe))
    gross_total = position.get("realized_gross", 0.0) + gross_remaining
    cost_total = position.get("realized_cost", 0.0) + cost_remaining
    net_total = gross_total - cost_total

    planned_risk = position["initial_quantity"] * position["stop_distance"]
    bars_held = max(1, int(position["bars"]))
    trade_id = stable_hash({"spec": spec.signature(), "ticker": table.ticker,
                            "entry": str(position["entry_time"]),
                            "version": spec.version})[:12]

    def build(net: float, cost: float, gross: float) -> TradeRecord:
        r_multiple = net / planned_risk if planned_risk else 0.0
        return TradeRecord(
            trade_id=trade_id, strategy_id=spec.strategy_id, version=spec.version,
            ticker=table.ticker, timeframe=table.timeframe, side=side,
            entry_time=str(position["entry_time"]), exit_time=str(timestamp),
            entry_price=entry, exit_price=exit_price, quantity=position["initial_quantity"],
            stop_price=position["stop"], target_price=position["target"],
            gross_pnl=float(gross), cost=float(cost), net_pnl=float(net),
            r_multiple=float(r_multiple),
            bars_held=bars_held, exit_reason=str(exit_info["reason"]),
            regime=position["regime"], session=position["session"],
            planned_risk=float(planned_risk),
            mae_r=float(position["mae_r"]), mfe_r=float(position["mfe_r"]),
            signals=list(position["signals"]),
            context={"atr_at_entry": position["atr_at_entry"],
                     "partials": position["partials"],
                     "risk_decision": position["decision"],
                     "bars_held": bars_held},
        )

    net_trade = build(net_total, cost_total, gross_total)
    gross_trade = build(gross_total, 0.0, gross_total)
    return net_trade, gross_trade, net_total


def _days_between(start: Any, end: Any, timeframe: str) -> float:
    try:
        delta = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400
    except (TypeError, ValueError):  # pragma: no cover - timestamps are always parseable here
        delta = 0.0
    if delta <= 0:
        delta = TIMEFRAME_MINUTES.get(timeframe, 1440) / 1440
    return float(delta)


def _fired_labels(spec: StrategySpec, features: pd.DataFrame, row: int) -> list[str]:
    """Which of the strategy's conditions were true at the signal bar."""
    labels: list[str] = []
    for condition in (*spec.long_entry, *spec.short_entry, *spec.filters):
        try:
            if bool(condition.evaluate(features).iloc[row]):
                labels.append(condition.label or condition.describe())
        except (KeyError, IndexError):
            continue
    return labels


def to_frame(trades: list[TradeRecord]) -> pd.DataFrame:
    return pd.DataFrame([t.as_dict() for t in trades])


def equity_frame(result: BacktestResult) -> pd.DataFrame:
    return result.equity.rename("equity").to_frame()
